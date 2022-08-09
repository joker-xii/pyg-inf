import argparse
import os
import threading
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

import time

######################
# Import From Quiver
######################
from quiver_feature import DistHelper, Range, DistTensorServerParam, DistTensorDeviceParam
from quiver_feature import DistTensorPGAS, serve_tensor_for_remote_access, PipeParam

######################
# Import Config File
######################
import config


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        time0 = time.time()
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x)

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        time1 = time.time() - time0
        print(time1, x_all.size(), time1 / x_all.size(0))
        return x_all


export_model = None
export_dist_tensor = None
export_edge_indx = None


def run(rank, process_rank_base, world_size, data_split, edge_index, dist_tensor, y, num_features, num_classes):
    os.environ['MASTER_ADDR'] = config.MASTER_IP
    os.environ['MASTER_PORT'] = "11421"
    # os.environ['NCCL_DEBUG'] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["TP_SOCKET_IFNAME"] = "eth0"
    os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
    os.environ["TP_VERBOSE_LOGGING"] = "0"

    device_rank = rank
    process_rank = process_rank_base + rank

    dist.init_process_group('nccl', rank=process_rank, world_size=world_size)

    torch.torch.cuda.set_device(device_rank)

    # train_mask, val_mask, test_mask = data_split
    # train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    # train_idx = train_idx.split(train_idx.size(0) // world_size)[process_rank]

    # train_loader = NeighborSampler(edge_index, node_idx=train_idx,
    #                                sizes=config.SAMPLE_PARAM, batch_size=config.BATCH_SIZE,
    #                                shuffle=True, persistent_workers=True,
    #                                num_workers=3)
    if process_rank == 0:
        subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                                          sizes=[-1], batch_size=config.BATCH_SIZE,
                                          shuffle=False, num_workers=0)

    torch.manual_seed(12345)
    model = SAGE(num_features, 256, num_classes).to(device_rank)
    model = DistributedDataParallel(model, device_ids=[device_rank])
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Simulate cases those data can not be fully stored by GPU memory
    y = y.to(device_rank)

    # if process_rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
    model.eval()
    global export_model, export_edge_indx, export_dist_tensor
    export_model = model
    export_edge_indx = edge_index
    export_dist_tensor = dist_tensor
    # subgraph_loader = NeighborSampler(edge_index, node_idx=None,
    #                                   sizes=[-1], batch_size=config.BATCH_SIZE,
    #                                   shuffle=False, num_workers=0)
    # with torch.no_grad():
    #     out = model.module.inference(dist_tensor, device_rank, subgraph_loader)
    # res = out.argmax(dim=-1) == y
    # acc1 = int(res[train_mask].sum()) / int(train_mask.sum())
    # acc2 = int(res[val_mask].sum()) / int(val_mask.sum())
    # acc3 = int(res[test_mask].sum()) / int(test_mask.sum())
    # print(f'Process_Rank: {process_rank}:\tTrain: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

    # dist.barrier()
    # dist.destroy_process_group()


def load_partitioned_data(args, data, node_count):
    """
    Return local data partition, cache information and local tensor range information
    """

    cached_range = Range(0, int(args.cache_ratio * node_count))

    UNCACHED_NUM_ELEMENT = (node_count - cached_range.end) // args.server_world_size

    range_list = []
    for idx in range(args.server_world_size):
        range_item = Range(cached_range.end + UNCACHED_NUM_ELEMENT * idx,
                           cached_range.end + UNCACHED_NUM_ELEMENT * (idx + 1))
        range_list.append(range_item)

    local_tensor = torch.cat([data.x[: cached_range.end], data.x[range_list[args.server_rank].start: range_list[
        args.server_rank].end]]).pin_memory()
    local_tensor.share_memory_()
    return local_tensor, cached_range, range_list[args.server_rank]


"""
Suppose we have 2 servers and Server 0 is the master node.

On Server 0:
    python3 distribue_training.py -server_rank 0 -device_per_node 1 -server_world_size 2

On Server 1:
    python3 distribute_training.py -server_rank 1 -device_per_node 1 -server_world_size 2


"""
parser = argparse.ArgumentParser(description='')
parser.add_argument('-server_rank', type=int, default=0, help='server_rank')
parser.add_argument('-device_per_node', type=int, default=1, help="how many process per server")
parser.add_argument('-server_world_size', type=int, default=1, help="world size")
parser.add_argument("-cache_ratio", type=float, default=0.0, help="how much data you want to cache")

args = parser.parse_args()

print("Loading Reddit Dataset")
dataset = Reddit('./')
data = dataset[0]

# Simulate Load Local data partition
local_tensor, cached_range, local_range = load_partitioned_data(args, data, data.x.shape[0])

print("Exchange TensorPoints Information")
dist_helper = DistHelper(config.MASTER_IP, config.HLPER_PORT, args.server_world_size, args.server_rank)
tensor_endpoints = dist_helper.exchange_tensor_endpoints_info(local_range)

print(f"[Server_Rank]: {args.server_rank}:\tBegin To Create DistTensorPGAS")
device_param = DistTensorDeviceParam(device_list=list(range(args.device_per_node)), device_cache_size="55M",
                                     cache_policy="device_replicate")
server_param = DistTensorServerParam(port_num=config.PORT_NUMBER, server_world_size=args.server_world_size)
buffer_shape = [np.prod(config.SAMPLE_PARAM) * config.BATCH_SIZE, local_tensor.shape[1]]
pipe_param = PipeParam(config.QP_NUM, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)

dist_tensor = DistTensorPGAS(args.server_rank, tensor_endpoints, pipe_param, buffer_shape, cached_range)
dist_tensor.from_cpu_tensor(local_tensor, dist_helper=dist_helper, server_param=server_param,
                            device_param=device_param)

print(f"Begin To Spawn Training Processes")
world_size = args.device_per_node * args.server_world_size
process_rank_base = args.device_per_node * args.server_rank
data_split = (data.train_mask, data.val_mask, data.test_mask)
run(0, process_rank_base, world_size, data_split, data.edge_index, dist_tensor, data.y, dataset.num_features,
    dataset.num_classes)
# mp.spawn(
#     run,
#     args=(process_rank_base, world_size, data_split, data.edge_index, dist_tensor, data.y, dataset.num_features,
#           dataset.num_classes),
#     nprocs=args.device_per_node,
#     join=True
# )
# dist_helper.

dist_helper.sync_all()
