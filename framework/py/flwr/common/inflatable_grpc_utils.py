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
"""InflatableObject utils."""


from typing import Optional, Union

from flwr.proto.fleet_pb2_grpc import FleetStub  # pylint: disable=E0611
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub  # pylint: disable=E0611

from .inflatable import (
    InflatableObject,
    get_object_head_values_from_object_content,
    get_object_id,
)
from .message import Message
from .record import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict

# Helper registry that maps names of classes to their type
inflatable_class_registry: dict[str, type[InflatableObject]] = {
    Array.__qualname__: Array,
    ArrayRecord.__qualname__: ArrayRecord,
    ConfigRecord.__qualname__: ConfigRecord,
    Message.__qualname__: Message,
    MetricRecord.__qualname__: MetricRecord,
    RecordDict.__qualname__: RecordDict,
}


def push_object_to_servicer(
    obj: InflatableObject,
    stub: Union[FleetStub, ServerAppIoStub],
    node: Node,
    run_id: int,
    object_ids_to_push: Optional[set[str]] = None,
) -> set[str]:
    """Recursively deflate an object and push it to the servicer.

    Objects with the same ID are not pushed twice. If `object_ids_to_push` is set,
    only objects with those IDs are pushed. It returns the set of pushed object
    IDs.
    """
    pushed_object_ids: set[str] = set()
    # Push children if it has any
    if children := obj.children:
        for child in children.values():
            pushed_object_ids |= push_object_to_servicer(
                child, stub, node, run_id, object_ids_to_push
            )

    # Deflate object and push
    object_content = obj.deflate()
    object_id = get_object_id(object_content)
    # Push always if no object set is specified, or if the object is in the set
    if object_ids_to_push is None or object_id in object_ids_to_push:
        _: PushObjectResponse = stub.PushObject(
            PushObjectRequest(
                node=node,
                run_id=run_id,
                object_id=object_id,
                object_content=object_content,
            )
        )
        pushed_object_ids.add(object_id)

    return pushed_object_ids


def pull_object_from_servicer(
    object_id: str,
    stub: Union[FleetStub, ServerAppIoStub],
    node: Node,
    run_id: int,
) -> InflatableObject:
    """Recursively inflate an object by pulling it from the servicer."""
    # Pull object
    object_proto: PullObjectResponse = stub.PullObject(
        PullObjectRequest(node=node, run_id=run_id, object_id=object_id)
    )
    object_content = object_proto.object_content

    # Extract object class and object_ids of children
    obj_type, children_obj_ids, _ = get_object_head_values_from_object_content(
        object_content=object_content
    )
    # Resolve object class
    cls_type = inflatable_class_registry[obj_type]

    # Pull all children objects
    children: dict[str, InflatableObject] = {}
    for child_object_id in children_obj_ids:
        children[child_object_id] = pull_object_from_servicer(
            child_object_id, stub, node, run_id
        )

    # Inflate object passing its children
    return cls_type.inflate(object_content, children=children)
