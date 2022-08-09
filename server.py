from inference_pb2 import Inquiry, Result
from inference_pb2_grpc import InferenceServicer
import grpc
import inference_pb2_grpc

import asyncio
import logging

import json


class MyService(InferenceServicer):
    async def Inference(self, request: Inquiry, context) -> Result:
        nodes = json.loads(request.nodes)
        time = int(request.time)
        fanout = int(request.fanout)
        from main import predictor
        result = predictor.get_predict_result(nodes, time, fanout)
        return Result(res=json.dumps(result))


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
