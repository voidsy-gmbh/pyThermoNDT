from __future__ import annotations

from collections import defaultdict
from collections.abc import ItemsView
from typing import TYPE_CHECKING, Any

from ..data.datacontainer.node import DataNode, GroupNode, RootNode
from ..data.units import generate_label
from .contracts import DEFAULT_PREVIEW_LIMIT, MAX_PREVIEW_LIMIT, JsonObject, JsonValue

if TYPE_CHECKING:
    from ..data import DataContainer


class DataContainerAdapter:
    """Read-only adapter exposing DataContainer content for browser visualization."""

    def __init__(self, container: DataContainer):
        self._container = container

    def list_children(self, path: str = "/") -> JsonObject:
        """List direct children of a root/group path."""
        normalized_path = self._normalize_path(path)
        parent_node = self._container.nodes(normalized_path)
        if isinstance(parent_node, DataNode):
            raise TypeError(f"Path '{normalized_path}' points to a dataset and has no children.")

        child_counts = self._direct_child_counts()
        children = []
        for node_path, node in self._container.nodes.items():
            if node_path == "/":
                continue
            if self._parent_path(node_path) != normalized_path:
                continue

            children.append(self._node_summary(node_path, node, child_counts.get(node_path, 0)))

        children.sort(key=lambda child: (self._node_order(str(child["node_type"])), str(child["name"])))
        return {"path": normalized_path, "children": children}

    def get_node_details(self, path: str) -> JsonObject:
        """Get details for a specific node path."""
        normalized_path = self._normalize_path(path)
        node = self._container.nodes(normalized_path)
        node_attributes = self._node_attributes(node)

        payload: JsonObject = {
            "path": normalized_path,
            "name": node.name,
            "node_type": node.type.value,
            "attributes": self._serialize_attributes(node_attributes),
        }

        if isinstance(node, DataNode):
            payload |= self._dataset_summary(normalized_path, node)

        return payload

    def get_dataset_preview(self, path: str, offset: int = 0, limit: int = DEFAULT_PREVIEW_LIMIT) -> JsonObject:
        """Get bounded flattened preview values of a dataset."""
        normalized_path = self._normalize_path(path)
        node = self._container.nodes(normalized_path)
        if not isinstance(node, DataNode):
            raise TypeError(f"Path '{normalized_path}' is not a dataset.")

        if offset < 0:
            raise ValueError("offset must be >= 0")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if limit > MAX_PREVIEW_LIMIT:
            raise ValueError(f"limit must be <= {MAX_PREVIEW_LIMIT}")

        data = node.data
        total = int(data.numel())
        if offset > total:
            raise ValueError(f"offset must be <= dataset size ({total})")

        end = min(total, offset + limit)
        values: list[JsonValue]
        if end == offset:
            values = []
        else:
            flat = data.reshape(-1)
            values = self._tensor_to_values(flat[offset:end])

        return {
            "path": normalized_path,
            "shape": [int(dim) for dim in data.shape],
            "dtype": str(data.dtype),
            "offset": offset,
            "limit": limit,
            "returned": len(values),
            "total": total,
            "values": values,
        }

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = path.strip()
        if not normalized:
            return "/"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"

        while "//" in normalized:
            normalized = normalized.replace("//", "/")

        if normalized != "/" and normalized.endswith("/"):
            normalized = normalized[:-1]

        return normalized

    @staticmethod
    def _parent_path(path: str) -> str:
        if path == "/":
            return ""
        if path.count("/") == 1:
            return "/"
        return path.rsplit("/", 1)[0]

    def _direct_child_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for path in self._container.nodes.keys():
            if path == "/":
                continue
            counts[self._parent_path(path)] += 1
        return counts

    def _node_summary(self, path: str, node: RootNode | GroupNode | DataNode, children_count: int) -> JsonObject:
        return {
            "path": path,
            "name": node.name,
            "node_type": node.type.value,
            "has_children": children_count > 0,
            "children_count": children_count,
        }

    @staticmethod
    def _node_order(node_type: str) -> int:
        order = {
            "root": 0,
            "group": 1,
            "dataset": 2,
        }
        return order.get(node_type, 99)

    def _dataset_summary(self, path: str, node: DataNode) -> JsonObject:
        data = node.data
        unit = self._container.get_unit(path)
        return {
            "shape": [int(dim) for dim in data.shape],
            "ndim": int(data.ndim),
            "dtype": str(data.dtype),
            "device": str(data.device),
            "numel": int(data.numel()),
            "size_bytes": int(data.element_size() * data.numel()),
            "unit": self._serialize_value(unit),
            "unit_label": generate_label(unit),
        }

    @staticmethod
    def _node_attributes(node: RootNode | GroupNode | DataNode) -> ItemsView[str, Any]:
        if isinstance(node, (GroupNode, DataNode)):
            return node.attributes
        return {}.items()

    def _serialize_attributes(self, attributes: ItemsView[str, Any]) -> JsonObject:
        return {key: self._serialize_value(value) for key, value in attributes}

    def _tensor_to_values(self, tensor_slice) -> list[JsonValue]:
        values = tensor_slice.detach().cpu().numpy().tolist()
        if isinstance(values, list):
            return [self._serialize_value(value) for value in values]
        return [self._serialize_value(values)]

    def _serialize_value(self, value: Any) -> JsonValue:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, tuple):
            return [self._serialize_value(item) for item in value]

        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]

        if isinstance(value, dict):
            return {str(key): self._serialize_value(val) for key, val in value.items()}

        if hasattr(value, "to_dict") and callable(value.to_dict):
            converted = value.to_dict()
            if isinstance(converted, dict):
                return {str(key): self._serialize_value(val) for key, val in converted.items()}

        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                scalar = item_method()
                return self._serialize_value(scalar)
            except (ValueError, TypeError):
                pass

        return str(value)
