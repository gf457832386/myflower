# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower ClientApp process."""


import gc
import time
from logging import DEBUG, ERROR, INFO
from typing import Optional

import grpc

from flwr.app.error import Error
from flwr.cli.install import install_from_fab
from flwr.client.client_app import ClientApp, LoadClientAppError
from flwr.client.clientapp.utils import get_load_client_app_fn
from flwr.common import Context, Message
from flwr.common.config import get_flwr_dir
from flwr.common.constant import ErrorCode
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log
from flwr.common.retry_invoker import _make_simple_grpc_retry_invoker, _wrap_stub
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    message_from_proto,
    message_to_proto,
    run_from_proto,
)
from flwr.common.typing import Fab, Run

# pylint: disable=E0611
from flwr.proto.clientappio_pb2 import (
    GetTokenRequest,
    GetTokenResponse,
    PullClientAppInputsRequest,
    PullClientAppInputsResponse,
    PushClientAppOutputsRequest,
    PushClientAppOutputsResponse,
)
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub


def run_clientapp(  # pylint: disable=R0914
    clientappio_api_address: str,
    run_once: bool,
    token: Optional[int] = None,
    flwr_dir: Optional[str] = None,
    certificates: Optional[bytes] = None,
) -> None:
    """Run Flower ClientApp process."""
    channel = create_channel(
        server_address=clientappio_api_address,
        insecure=(certificates is None),
        root_certificates=certificates,
    )
    channel.subscribe(on_channel_state_change)

    # Resolve directory where FABs are installed
    flwr_dir_ = get_flwr_dir(flwr_dir)
    try:
        stub = ClientAppIoStub(channel)
        _wrap_stub(stub, _make_simple_grpc_retry_invoker())

        while True:
            # If token is not set, loop until token is received from SuperNode
            while token is None:
                token = get_token(stub)
                time.sleep(1)

            # Pull Message, Context, Run and (optional) FAB from SuperNode
            message, context, run, fab = pull_clientappinputs(stub=stub, token=token)

            # Install FAB, if provided
            if fab:
                log(DEBUG, "[flwr-clientapp] Start FAB installation.")
                install_from_fab(fab.content, flwr_dir=flwr_dir_, skip_prompt=True)

            load_client_app_fn = get_load_client_app_fn(
                default_app_ref="",
                app_path=None,
                multi_app=True,
                flwr_dir=str(flwr_dir_),
            )

            try:
                # Load ClientApp
                log(DEBUG, "[flwr-clientapp] Start `ClientApp` Loading.")
                client_app: ClientApp = load_client_app_fn(
                    run.fab_id, run.fab_version, fab.hash_str if fab else ""
                )

                # Execute ClientApp
                reply_message = client_app(message=message, context=context)

            except Exception as ex:  # pylint: disable=broad-exception-caught
                # Don't update/change NodeState

                e_code = ErrorCode.CLIENT_APP_RAISED_EXCEPTION
                # Ex fmt: "<class 'ZeroDivisionError'>:<'division by zero'>"
                reason = str(type(ex)) + ":<'" + str(ex) + "'>"
                exc_entity = "ClientApp"
                if isinstance(ex, LoadClientAppError):
                    reason = (
                        "An exception was raised when attempting to load `ClientApp`"
                    )
                    e_code = ErrorCode.LOAD_CLIENT_APP_EXCEPTION

                log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)

                # Create error message
                reply_message = Message(
                    Error(code=e_code, reason=reason), reply_to=message
                )

            # Push Message and Context to SuperNode
            _ = push_clientappoutputs(
                stub=stub, token=token, message=reply_message, context=context
            )

            del client_app, message, context, run, fab, reply_message
            gc.collect()

            # Reset token to `None` to prevent flwr-clientapp from trying to pull the
            # same inputs again
            token = None

            # Stop the loop if `flwr-clientapp` is expected to process only a single
            # message
            if run_once:
                break

    except KeyboardInterrupt:
        log(INFO, "Closing connection")
    except grpc.RpcError as e:
        log(ERROR, "GRPC error occurred: %s", str(e))
    finally:
        channel.close()


def get_token(stub: grpc.Channel) -> Optional[int]:
    """Get a token from SuperNode."""
    log(DEBUG, "[flwr-clientapp] Request token")
    try:
        res: GetTokenResponse = stub.GetToken(GetTokenRequest())
        log(DEBUG, "[GetToken] Received token: %s", res.token)
        return res.token
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.FAILED_PRECONDITION:  # pylint: disable=no-member
            log(DEBUG, "[GetToken] No token available yet")
        else:
            log(ERROR, "[GetToken] gRPC error occurred: %s", str(e))
        return None


def pull_clientappinputs(
    stub: grpc.Channel, token: int
) -> tuple[Message, Context, Run, Optional[Fab]]:
    """Pull ClientAppInputs from SuperNode."""
    log(INFO, "[flwr-clientapp] Pull `ClientAppInputs` for token %s", token)
    try:
        res: PullClientAppInputsResponse = stub.PullClientAppInputs(
            PullClientAppInputsRequest(token=token)
        )
        message = message_from_proto(res.message)
        context = context_from_proto(res.context)
        run = run_from_proto(res.run)
        fab = fab_from_proto(res.fab) if res.fab else None
        return message, context, run, fab
    except grpc.RpcError as e:
        log(ERROR, "[PullClientAppInputs] gRPC error occurred: %s", str(e))
        raise e


def push_clientappoutputs(
    stub: grpc.Channel, token: int, message: Message, context: Context
) -> PushClientAppOutputsResponse:
    """Push ClientAppOutputs to SuperNode."""
    log(INFO, "[flwr-clientapp] Push `ClientAppOutputs` for token %s", token)
    proto_message = message_to_proto(message)
    proto_context = context_to_proto(context)

    try:
        res: PushClientAppOutputsResponse = stub.PushClientAppOutputs(
            PushClientAppOutputsRequest(
                token=token, message=proto_message, context=proto_context
            )
        )
        return res
    except grpc.RpcError as e:
        log(ERROR, "[PushClientAppOutputs] gRPC error occurred: %s", str(e))
        raise e
