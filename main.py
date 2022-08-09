from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.data import NeighborSampler
from model import RecurrentGCN
import torch
from tqdm import tqdm

loader = ChickenpoxDatasetLoader()
dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

model = RecurrentGCN(node_features=4)
model.eval()
data = {time: snapshot for time, snapshot in enumerate(test_dataset)}


class Predict:
    def __init__(self, model=model, data=data):
        self.model = model
        self.data = data
        self.sampler = {}

    def get_data(self, time, fanout):
        snapshot = self.data[time]
        if time not in self.sampler:
            sampler = NeighborSampler(snapshot.edge_index, [fanout], return_e_id=True)
            self.sampler[time] = sampler

        return snapshot, self.sampler[time]

    def get_predict_result(self, time, ids, fanout):
        snapshot, sampler = self.get_data(time, fanout)
        batch_data = sampler.sample(torch.tensor(ids))
        result = self.model(snapshot.x[batch_data[1]], batch_data[2].edge_index, snapshot.edge_attr[batch_data[2].e_id])
        return result[:batch_data[0]].cpu().tolist()


from qv_inference import export_dist_tensor, export_model, export_edge_indx
import config
class QuiverPredict:
    def __init__(self):
        self.model = export_model
        self.edge_index = export_edge_indx
        self.dist_tensor = export_dist_tensor
        self.sampler = NeighborSampler(export_edge_indx,
                                       sizes=config.SAMPLE_PARAM, batch_size=config.BATCH_SIZE,
                                       shuffle=True, persistent_workers=True,
                                       num_workers=3)

    def get_predict_result(self, ids, *args, **kwargs):
        sampler = self.sampler
        batch_data = sampler.sample(torch.tensor(ids))
        result = self.model(self.dist_tensor[batch_data[1]], batch_data[2])
        return result[:batch_data[0]].cpu().tolist()


# predictor = Predict(model, data)
predictor = QuiverPredict()
if __name__ == '__main__':
    pass

# y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
# cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
# cost = cost / (time + 1)
# cost = cost.item()
# print("MSE: {:.4f}".format(cost))
