"""Weight Exporter: Export model weights as CUDA IPC handles for inheritance.

The old process (about to be replaced) exports its model weights' CUDA IPC
handles to a shared file. The new process can then open these handles and
slice the weights for the new TP configuration without any disk IO.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

RESHARD_IPC_DIR = Path(os.environ.get("SGLANG_RESHARD_IPC_DIR", "/tmp/sglang_reshard"))


@dataclass
class WeightHandle:
    """Metadata for a single exported weight tensor."""
    name: str
    shape: List[int]
    dtype: str
    ipc_handle: bytes
    device_idx: int


@dataclass
class ExportedWeights:
    """Collection of exported weight handles from a process."""
    tp_size: int
    tp_rank: int
    module_type: str  # "prefill" or "decode"
    perspective: str  # "attn" or "ffn" (for PDAF) or "full" (for PD)
    handles: Dict[str, WeightHandle] = field(default_factory=dict)


class WeightExporter:
    """Export model weights as CUDA IPC handles.

    Usage in the old process (before shutdown):
        exporter = WeightExporter(model, tp_size=1, tp_rank=0,
                                  module_type="prefill", perspective="attn")
        export_path = exporter.export()
        # ... signal new process that handles are ready ...
        # ... wait for new process to confirm receipt ...
        exporter.cleanup()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tp_size: int,
        tp_rank: int,
        module_type: str = "prefill",
        perspective: str = "full",
        export_dir: Optional[Path] = None,
    ):
        self.model = model
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.module_type = module_type
        self.perspective = perspective
        self.export_dir = export_dir or RESHARD_IPC_DIR
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._exported_tensors: List[torch.Tensor] = []

    def export(self) -> Path:
        """Export all model parameters as IPC handles.

        Returns path to the export metadata file.
        """
        handles: Dict[str, dict] = {}

        for name, param in self.model.named_parameters():
            if not param.is_cuda:
                continue
            tensor = param.data.contiguous()
            self._exported_tensors.append(tensor)

            # _share_cuda_() returns (device, handle_bytes, storage_size_bytes, storage_offset,
            #                         ref_counter_handle, ref_counter_offset, event_handle, event_sync_required)
            share_tuple = tensor.untyped_storage()._share_cuda_()

            handles[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "ipc_handle": share_tuple[1].hex(),
                "device_idx": share_tuple[0],
                "storage_size_bytes": share_tuple[2],
                "storage_offset": share_tuple[3],
                "ref_counter_handle": share_tuple[4].hex(),
                "ref_counter_offset": share_tuple[5],
                "event_handle": share_tuple[6].hex(),
                "event_sync_required": share_tuple[7],
                "numel": tensor.numel(),
                "tensor_storage_offset": tensor.storage_offset(),
            }

        metadata = {
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "module_type": self.module_type,
            "perspective": self.perspective,
            "num_params": len(handles),
            "handles": handles,
        }

        export_path = self.export_dir / f"{self.module_type}_{self.perspective}_tp{self.tp_size}_rank{self.tp_rank}.json"
        with open(export_path, "w") as f:
            json.dump(metadata, f)

        logger.info(
            "Exported %d weight tensors (tp%d rank%d %s/%s) → %s",
            len(handles), self.tp_size, self.tp_rank,
            self.module_type, self.perspective, export_path,
        )
        return export_path

    def export_raw_handles(self) -> Dict[str, Tuple[torch.Tensor, bytes]]:
        """Export raw tensor + IPC handle pairs (for in-process transfer).

        Returns dict mapping param name → (tensor, ipc_handle_bytes).
        Used when old and new processes can communicate via shared memory
        without going through file serialization.
        """
        result = {}
        for name, param in self.model.named_parameters():
            if not param.is_cuda:
                continue
            tensor = param.data.contiguous()
            self._exported_tensors.append(tensor)
            handle_bytes = tensor.storage()._share_cuda_()[1]
            result[name] = (tensor, handle_bytes)
        return result

    def cleanup(self):
        """Release references to exported tensors.

        Call this AFTER the new process confirms it has copied the weights.
        """
        self._exported_tensors.clear()
        export_path = self.export_dir / f"{self.module_type}_{self.perspective}_tp{self.tp_size}_rank{self.tp_rank}.json"
        if export_path.exists():
            export_path.unlink()
        logger.info("Weight export cleanup done.")
