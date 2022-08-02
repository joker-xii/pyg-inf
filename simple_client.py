import asyncio
import logging

import grpc
import inference_pb2_grpc
import inference_pb2
import json


async def run() -> None:
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = inference_pb2_grpc.InferenceStub(channel)
        response = await stub.Inference(inference_pb2.Inquiry(nodes=json.dumps([0, 1, 2]), time=0, fanout=10))
    print("client received: " + response.res)


if __name__ == '__main__':
    logging.basicConfig()
    asyncio.run(run())
