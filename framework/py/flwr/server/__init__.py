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
"""Flower server."""


from ..compat.server.app import start_server as start_server  # Deprecated
from . import strategy
from . import workflow as workflow
from .client_manager import ClientManager as ClientManager
from .client_manager import SimpleClientManager as SimpleClientManager
from .compat import LegacyContext as LegacyContext
from .grid import Driver as Driver
from .grid import Grid as Grid
from .history import History as History
from .server import Server as Server
from .server_app import ServerApp as ServerApp
from .server_config import ServerConfig as ServerConfig
from .serverapp_components import ServerAppComponents as ServerAppComponents

__all__ = [
    "ClientManager",
    "Driver",
    "Grid",
    "History",
    "LegacyContext",
    "Server",
    "ServerApp",
    "ServerAppComponents",
    "ServerConfig",
    "SimpleClientManager",
    "start_server",
    "strategy",
    "workflow",
]
