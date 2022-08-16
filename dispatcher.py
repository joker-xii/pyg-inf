from main import predictor
from pylru import lrucache
from collections import defaultdict
import time


def default_node_grouper(node_id):
    return 0


class RequestItem:
    def __init__(self):
        self.server_ids = set()
        self.create_time = -1

    def attach_server(self, server_id, create_time):
        if self.create_time == -1:
            self.create_time = create_time
        self.server_ids.add(server_id)


class Dispatcher:
    def __init__(self, max_batch_size, buffer_size, max_wait, node_grouper=default_node_grouper):
        self.node_groups = defaultdict(lambda: defaultdict(RequestItem))
        self.buffer = lrucache(buffer_size)
        self.current_create_time = defaultdict(int)
        self.max_wait = max_wait
        self.max_batch_size = max_batch_size
        self.node_grouper = node_grouper
        self.result = {}
        pass

    def invalidate_buffer(self, nodes):
        for node in nodes:
            del self.buffer[node]

    def check_and_run(self):
        # TODO Async call
        for node_group_id, node_group in self.node_groups.items():
            if self.current_create_time[node_group_id] + self.max_wait >= time.time_ns() \
                    or len(node_group) >= self.max_batch_size:
                node_ids = list(node_group.keys())
                results = predictor.get_predict_result(node_ids)
                for i, r in enumerate(results):
                    self.result[node_ids[i]] = r

    def add_record(self, node, server_id):
        group_id = self.node_grouper(node)
        create_time = time.time_ns()
        self.node_groups[group_id][node].attach_server(server_id, create_time)
        if group_id not in self.current_create_time:
            self.current_create_time[group_id] = create_time
        self.check_and_run()
        pass

    def get_result(self, node, server_id):
        if node in self.buffer:
            return self.buffer[node]

        self.check_and_run()
        if node in self.result:
            self.buffer[node] = self.result[node]
            return self.result[node]
        return None


dispatcher = Dispatcher(10, 50, 100)
