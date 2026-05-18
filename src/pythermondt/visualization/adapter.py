from __future__ import annotations

from collections import defaultdict
from collections.abc import ItemsView
from typing import TYPE_CHECKING, Any, cast

import torch

from ..data.datacontainer.node import DataNode, GroupNode, RootNode
from ..data.units import generate_label
from .contracts import (
    BINARY_DTYPE,
    DEFAULT_MATRIX_COLS,
    DEFAULT_MATRIX_ROWS,
    DEFAULT_VECTOR_COUNT,
    MAX_BINARY_RESPONSE_BYTES,
    MAX_MATRIX_COLS,
    MAX_MATRIX_ROWS,
    MAX_VECTOR_COUNT,
    JsonObject,
    JsonValue,
)

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

    def get_plot_meta(self, path: str) -> JsonObject:
        """Get plotting metadata for a dataset path."""
        normalized_path, node = self._get_dataset_node(path)
        shape = [int(dim) for dim in node.data.shape]
        ndim = int(node.data.ndim)

        available_modes: list[str] = ["line"]
        default_mode = "line"
        frame_axis_options: list[int] = []
        default_frame_axis: int | None = None
        default_frame_count: int | None = None
        default_frame_shape: list[int] | None = None

        if ndim == 1:
            available_modes = ["matrix", "line"]
            default_mode = "line"
        elif ndim == 2:
            available_modes = ["matrix", "line", "heatmap"]
            default_mode = "heatmap"
            default_frame_shape = shape
            default_frame_count = 1
        elif ndim == 3:
            available_modes = ["matrix", "line", "heatmap"]
            default_mode = "heatmap"
            frame_axis_options = [0, 1, 2]
            default_frame_axis = ndim - 1
            default_frame_count = shape[default_frame_axis]
            default_frame_shape = [shape[idx] for idx in range(ndim) if idx != default_frame_axis]

        render_mode = "frames" if default_mode == "heatmap" and ndim == 3 else default_mode
        payload: JsonObject = {
            "path": normalized_path,
            "render_mode": render_mode,
            "default_mode": default_mode,
            "available_modes": available_modes,
            "ndim": ndim,
            "shape": shape,
            "dtype": str(node.data.dtype),
            "numel": int(node.data.numel()),
            "frame_axis_options": frame_axis_options,
            "default_frame_axis": default_frame_axis,
            "default_frame_count": default_frame_count,
            "default_frame_shape": default_frame_shape,
        }

        if ndim > 3:
            payload["reason"] = "Only line mode is available for datasets with more than 3 dimensions."

        return payload

    def get_vector_binary(
        self,
        path: str,
        start: int = 0,
        count: int = DEFAULT_VECTOR_COUNT,
        stride: int = 1,
    ) -> tuple[bytes, JsonObject]:
        """Get 1D dataset values as binary float32 payload."""
        normalized_path, node = self._get_dataset_node(path)

        if int(node.data.ndim) < 1:
            raise TypeError(f"Path '{normalized_path}' must point to a non-scalar dataset for vector plotting.")
        if start < 0:
            raise ValueError("start must be >= 0")
        if count < 1:
            raise ValueError("count must be >= 1")
        if count > MAX_VECTOR_COUNT:
            raise ValueError(f"count must be <= {MAX_VECTOR_COUNT}")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        flat = node.data.reshape(-1)
        total = int(flat.shape[0])
        if start >= total:
            raise ValueError(f"start must be < vector size ({total})")

        vector = flat[start::stride][:count]
        prepared = self._prepare_float32_tensor(vector)

        self._validate_binary_size(int(prepared.numel()))
        payload = prepared.numpy().tobytes(order="C")
        metadata: JsonObject = {
            "dtype": BINARY_DTYPE,
            "shape": [int(prepared.numel())],
            "ndim": 1,
            "count": int(prepared.numel()),
        }
        return payload, metadata

    def get_frame_binary(
        self,
        path: str,
        frame_axis: int | None = None,
        frame_index: int | None = None,
    ) -> tuple[bytes, JsonObject]:
        """Get 2D frame data as binary float32 payload."""
        normalized_path, node = self._get_dataset_node(path)
        frame, selected_axis, selected_index = self._select_2d_frame(
            normalized_path,
            node.data,
            frame_axis=frame_axis,
            frame_index=frame_index,
        )

        prepared = self._prepare_float32_tensor(frame)
        self._validate_binary_size(int(prepared.numel()))

        rows, cols = [int(dim) for dim in prepared.shape]
        payload = prepared.numpy().tobytes(order="C")
        metadata: JsonObject = {
            "dtype": BINARY_DTYPE,
            "shape": [rows, cols],
            "ndim": 2,
            "count": int(prepared.numel()),
            "frame_axis": selected_axis,
            "frame_index": selected_index,
        }
        return payload, metadata

    def get_matrix_data(
        self,
        path: str,
        frame_axis: int | None = None,
        frame_index: int | None = None,
        row_start: int = 0,
        row_count: int = DEFAULT_MATRIX_ROWS,
        col_start: int = 0,
        col_count: int = DEFAULT_MATRIX_COLS,
    ) -> JsonObject:
        """Get paged matrix values for table rendering."""
        normalized_path, node = self._get_dataset_node(path)
        matrix, selected_axis, selected_index = self._select_matrix_2d(
            normalized_path,
            node.data,
            frame_axis=frame_axis,
            frame_index=frame_index,
        )

        if row_start < 0:
            raise ValueError("row_start must be >= 0")
        if col_start < 0:
            raise ValueError("col_start must be >= 0")
        if row_count < 1:
            raise ValueError("row_count must be >= 1")
        if col_count < 1:
            raise ValueError("col_count must be >= 1")
        if row_count > MAX_MATRIX_ROWS:
            raise ValueError(f"row_count must be <= {MAX_MATRIX_ROWS}")
        if col_count > MAX_MATRIX_COLS:
            raise ValueError(f"col_count must be <= {MAX_MATRIX_COLS}")

        rows, cols = [int(dim) for dim in matrix.shape]
        if row_start >= rows:
            raise ValueError(f"row_start must be < number of rows ({rows})")
        if col_start >= cols:
            raise ValueError(f"col_start must be < number of columns ({cols})")

        row_end = min(rows, row_start + row_count)
        col_end = min(cols, col_start + col_count)

        window = matrix[row_start:row_end, col_start:col_end]
        values = self._prepare_float32_tensor(window).numpy().tolist()

        return {
            "path": normalized_path,
            "shape": [rows, cols],
            "dtype": BINARY_DTYPE,
            "frame_axis": selected_axis,
            "frame_index": selected_index,
            "row_start": row_start,
            "row_end": row_end,
            "col_start": col_start,
            "col_end": col_end,
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

    def _get_dataset_node(self, path: str) -> tuple[str, DataNode]:
        normalized_path = self._normalize_path(path)
        node = self._container.nodes(normalized_path)
        if not isinstance(node, DataNode):
            raise TypeError(f"Path '{normalized_path}' is not a dataset.")
        return normalized_path, node

    def _select_2d_frame(
        self,
        path: str,
        data: torch.Tensor,
        frame_axis: int | None,
        frame_index: int | None,
    ) -> tuple[torch.Tensor, int, int]:
        ndim = int(data.ndim)
        if ndim not in (2, 3):
            raise TypeError(f"Path '{path}' must point to a 2D or 3D dataset for heatmap plotting.")

        selected_axis = -1
        selected_index = 0
        frame: torch.Tensor

        if ndim == 2:
            if frame_index not in (None, 0):
                raise ValueError("frame_index must be 0 for 2D datasets")
            frame = data
        else:
            selected_axis = int(frame_axis if frame_axis is not None else (ndim - 1))
            if selected_axis < 0:
                selected_axis += ndim
            if selected_axis < 0 or selected_axis >= ndim:
                raise ValueError(f"frame_axis must be in range [0, {ndim})")

            selected_index = int(frame_index if frame_index is not None else 0)
            frame_count = int(data.shape[selected_axis])
            if selected_index < 0 or selected_index >= frame_count:
                raise ValueError(f"frame_index must be in range [0, {frame_count})")

            slicing = cast(list[slice | int], [slice(None)] * ndim)
            slicing[selected_axis] = selected_index
            frame = data[tuple(slicing)]
            if int(frame.ndim) != 2:
                raise ValueError("Only 2D frame extraction is supported for 3D datasets.")

        return frame, selected_axis, selected_index

    def _select_matrix_2d(
        self,
        path: str,
        data: torch.Tensor,
        frame_axis: int | None,
        frame_index: int | None,
    ) -> tuple[torch.Tensor, int, int]:
        ndim = int(data.ndim)
        if ndim == 1:
            if frame_index not in (None, 0):
                raise ValueError("frame_index must be 0 for 1D datasets")
            return data.unsqueeze(0), -1, 0
        if ndim in (2, 3):
            return self._select_2d_frame(path, data, frame_axis=frame_axis, frame_index=frame_index)

        raise TypeError(f"Path '{path}' must point to a 1D, 2D or 3D dataset for matrix mode.")

    @staticmethod
    def _prepare_float32_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()

    @staticmethod
    def _validate_binary_size(num_values: int) -> None:
        total_bytes = num_values * 4
        if total_bytes > MAX_BINARY_RESPONSE_BYTES:
            raise ValueError(
                f"Requested payload is too large ({total_bytes} bytes). Maximum is {MAX_BINARY_RESPONSE_BYTES} bytes."
            )

    @staticmethod
    def _node_attributes(node: RootNode | GroupNode | DataNode) -> ItemsView[str, Any]:
        if isinstance(node, (GroupNode, DataNode)):
            return node.attributes
        return {}.items()

    def _serialize_attributes(self, attributes: ItemsView[str, Any]) -> JsonObject:
        return {key: self._serialize_value(value) for key, value in attributes}

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
