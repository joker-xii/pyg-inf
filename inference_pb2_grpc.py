# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import inference_pb2 as inference__pb2


class InferenceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Inference = channel.unary_unary(
                '/Inference/Inference',
                request_serializer=inference__pb2.Inquiry.SerializeToString,
                response_deserializer=inference__pb2.Result.FromString,
                )


class InferenceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Inference(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Inference': grpc.unary_unary_rpc_method_handler(
                    servicer.Inference,
                    request_deserializer=inference__pb2.Inquiry.FromString,
                    response_serializer=inference__pb2.Result.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Inference', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Inference(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Inference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Inference/Inference',
            inference__pb2.Inquiry.SerializeToString,
            inference__pb2.Result.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
