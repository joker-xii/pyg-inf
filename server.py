from inference_pb2 import Inquiry, Result
from inference_pb2_grpc import InferenceServicer
import grpc
import inference_pb2_grpc

import asyncio
import logging

import json
import random

fixed_batch_size = 5


class MyService(InferenceServicer):
    async def Inference(self, request: Inquiry, context) -> Result:
        nodes = json.loads(request.nodes)
        time = int(request.time)
        fanout = int(request.fanout)
        from main import predictor
        result = predictor.get_predict_result(nodes, time, fanout)
        return Result(res=json.dumps(result))

    async def StreamingInference(self, request_iterator, context):
        from main import predictor
        for node in request_iterator:
            result = predictor.get_predict_result([node.node])
            yield Result(res=json.dumps(result))

    async def StreamingInferenceByLabel(self, request_iterator, context):
        pass

    async def SetDynamicBatchProperty(self, request, context):
        pass

    async def SetFixedBatchProperty(self, request, context):
        global fixed_batch_size
        fixed_batch_size = request.batch_size

    async def StreamingDynamicBatchInference(self, request_iterator, context):
        curr_id = random.random()
        from main import predictor
        from dispatcher import dispatcher
        added = []
        for node in request_iterator:
            dispatcher.add_record(node, curr_id)
            added.append(node)

        for node in added:
            result = None
            while result is None:
                result = dispatcher.get_result(node, curr_id)
                yield Result(res=json.dumps(result))

    async def StreamingFixedBatchInference(self, request_iterator, context):
        from main import predictor
        batches = [[]]
        for node in request_iterator:
            batches[-1].append(node.node)
            if len(batches[-1]) == fixed_batch_size:
                batches.append([])

        for batch in batches:
            result = predictor.get_predict_result(batch)
            for val in result:
                yield Result(res=json.dumps(val))


async def serve() -> None:
    server = grpc.aio.server()
    inference_pb2_grpc.add_InferenceServicer_to_server(MyService(), server)
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
