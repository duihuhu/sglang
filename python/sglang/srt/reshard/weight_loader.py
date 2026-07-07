"""Fast TP Loader: Inherit weights from IPC handles and re-slice for new TP.

Instead of loading from disk (~10s), the new process opens CUDA IPC handles
exported by the old process, slices weights for the new TP config, and copies
them into its own parameter buffers. This takes ~1-2s via NVLink.

TP Slicing Rules (standard Megatron-style):
  - Column parallel (gate_proj, up_proj, q_proj, k_proj, v_proj): slice output rows
  - Row parallel (down_proj, o_proj): slice input columns
  - Embedding / lm_head: slice along vocab dimension
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

RESHARD_IPC_DIR = Path(os.environ.get("SGLANG_RESHARD_IPC_DIR", "/tmp/sglang_reshard"))

# Weight name patterns for TP slicing strategy
COLUMN_PARALLEL_PATTERNS = [
    "gate_proj", "up_proj", "q_proj", "k_proj", "v_proj",
    "gate_up_proj", "qkv_proj",
]
ROW_PARALLEL_PATTERNS = [
    "down_proj", "o_proj",
]
VOCAB_PARALLEL_PATTERNS = [
    "embed_tokens", "lm_head",
]


def _get_slice_strategy(param_name: str) -> str:
    """Determine how to slice a parameter for TP."""
    for pattern in COLUMN_PARALLEL_PATTERNS:
        if pattern in param_name:
            return "column"
    for pattern in ROW_PARALLEL_PATTERNS:
        if pattern in param_name:
            return "row"
    for pattern in VOCAB_PARALLEL_PATTERNS:
        if pattern in param_name:
            return "vocab"
    return "replicate"


def _slice_weight(
    full_weight: torch.Tensor,
    new_tp_size: int,
    new_tp_rank: int,
    strategy: str,
) -> torch.Tensor:
    """Slice a full weight tensor for the given TP rank.

    Args:
        full_weight: The complete (unsliced) weight tensor.
        new_tp_size: Target TP parallelism.
        new_tp_rank: This rank's index in the new TP group.
        strategy: One of "column", "row", "vocab", "replicate".

    Returns:
        The sliced tensor for this rank.
    """
    if strategy == "replicate" or new_tp_size == 1:
        return full_weight.clone()

    if strategy == "column":
        # PyTorch linear weights are [out_features, in_features]. Megatron
        # column parallelism shards the output features, i.e. dim 0.
        dim = 0
        chunk_size = full_weight.shape[dim] // new_tp_size
        start = new_tp_rank * chunk_size
        end = start + chunk_size
        return full_weight.narrow(dim, start, chunk_size).contiguous()

    elif strategy == "row":
        # Row parallelism shards the input features, i.e. dim 1 for matrix
        # weights. Bias and other 1-D tensors fall back to dim 0.
        dim = 1 if full_weight.dim() == 2 else 0
        chunk_size = full_weight.shape[dim] // new_tp_size
        start = new_tp_rank * chunk_size
        end = start + chunk_size
        return full_weight.narrow(dim, start, chunk_size).contiguous()

    elif strategy == "vocab":
        vocab_size = full_weight.shape[0]
        chunk_size = (vocab_size + new_tp_size - 1) // new_tp_size
        start = new_tp_rank * chunk_size
        end = min(start + chunk_size, vocab_size)
        return full_weight[start:end].contiguous()

    return full_weight.clone()


def _reconstruct_full_weight(
    slices: List[torch.Tensor],
    old_tp_size: int,
    strategy: str,
) -> torch.Tensor:
    """Reconstruct full weight from TP-sliced pieces.

    Used when the old process had TP > 1 and we need the full weight
    to re-slice for a different TP size.
    """
    if old_tp_size == 1:
        return slices[0]

    if strategy == "column":
        return torch.cat(slices, dim=0)
    elif strategy == "row":
        dim = 1 if slices[0].dim() == 2 else 0
        return torch.cat(slices, dim=dim)
    elif strategy == "vocab":
        return torch.cat(slices, dim=0)
    else:
        return slices[0]


class FastTPLoader:
    """Load model weights from IPC handles with TP re-slicing.

    Usage in the new process:
        loader = FastTPLoader(
            new_tp_size=2, new_tp_rank=0,
            old_tp_size=1,
            module_type="prefill", perspective="attn",
        )
        if loader.has_ipc_weights():
            loader.load_into_model(model)
        else:
            # fallback to standard safetensors loading
            ...
    """

    def __init__(
        self,
        new_tp_size: int,
        new_tp_rank: int,
        old_tp_size: int = 1,
        module_type: str = "prefill",
        perspective: str = "full",
        ipc_dir: Optional[Path] = None,
    ):
        self.new_tp_size = new_tp_size
        self.new_tp_rank = new_tp_rank
        self.old_tp_size = old_tp_size
        self.module_type = module_type
        self.perspective = perspective
        self.ipc_dir = ipc_dir or RESHARD_IPC_DIR

    def has_ipc_weights(self) -> bool:
        """Check if IPC weight handles are available from old process."""
        for rank in range(self.old_tp_size):
            path = self.ipc_dir / f"{self.module_type}_{self.perspective}_tp{self.old_tp_size}_rank{rank}.json"
            if not path.exists():
                return False
        return True

    def _load_handles(self) -> List[Dict]:
        """Load all rank handle files."""
        all_handles = []
        for rank in range(self.old_tp_size):
            path = self.ipc_dir / f"{self.module_type}_{self.perspective}_tp{self.old_tp_size}_rank{rank}.json"
            with open(path) as f:
                data = json.load(f)
            all_handles.append(data)
        return all_handles

    def _open_ipc_tensor(self, handle_info: dict, device: torch.device) -> torch.Tensor:
        """Open a CUDA IPC handle and reconstruct the tensor.

        Uses PyTorch's UntypedStorage._new_shared_cuda() which expects the full
        8-tuple from _share_cuda_().

        If the IPC tensor is on a different GPU than `device`, it will be copied.
        """
        shape = handle_info["shape"]
        dtype = getattr(torch, handle_info["dtype"].replace("torch.", ""))

        # Reconstruct the 8-tuple for _new_shared_cuda
        share_tuple = (
            handle_info["device_idx"],
            bytes.fromhex(handle_info["ipc_handle"]),
            handle_info["storage_size_bytes"],
            handle_info["storage_offset"],
            bytes.fromhex(handle_info["ref_counter_handle"]),
            handle_info["ref_counter_offset"],
            bytes.fromhex(handle_info["event_handle"]),
            handle_info["event_sync_required"],
        )

        storage = torch.UntypedStorage._new_shared_cuda(*share_tuple)
        src_device = torch.device(f"cuda:{handle_info['device_idx']}")
        tensor = torch.tensor([], dtype=dtype, device=src_device)
        tensor_offset = handle_info.get("tensor_storage_offset", 0)
        tensor.set_(storage, tensor_offset, shape)

        # Copy to target device if different
        if device != src_device:
            tensor = tensor.to(device)

        return tensor

    def load_into_model(self, model: torch.nn.Module, device: Optional[torch.device] = None) -> bool:
        """Load weights from IPC into model with TP re-slicing.

        Args:
            model: Target model whose parameters will be updated.
            device: Target device. If None, uses current CUDA device.

        Returns:
            True if successful, False otherwise.
        """
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")

        if not self.has_ipc_weights():
            logger.warning("No IPC weight handles found, cannot load.")
            return False

        all_handles = self._load_handles()
        loaded_count = 0
        skipped_count = 0

        model_params = dict(model.named_parameters())

        # Get param names from first rank's handles
        param_names = list(all_handles[0]["handles"].keys())

        for param_name in param_names:
            if param_name not in model_params:
                skipped_count += 1
                continue

            target_param = model_params[param_name]
            strategy = _get_slice_strategy(param_name)

            # Collect slices from all old ranks
            old_slices = []
            for rank_data in all_handles:
                if param_name not in rank_data["handles"]:
                    continue
                handle_info = rank_data["handles"][param_name]
                tensor = self._open_ipc_tensor(handle_info, device)
                old_slices.append(tensor)

            if not old_slices:
                skipped_count += 1
                continue

            # Reconstruct full weight
            full_weight = _reconstruct_full_weight(old_slices, self.old_tp_size, strategy)

            # Re-slice for new TP
            new_slice = _slice_weight(full_weight, self.new_tp_size, self.new_tp_rank, strategy)

            # Copy into model parameter
            if new_slice.shape == target_param.shape:
                target_param.data.copy_(new_slice)
                loaded_count += 1
            else:
                logger.warning(
                    "Shape mismatch for %s: got %s, expected %s",
                    param_name, new_slice.shape, target_param.shape,
                )
                skipped_count += 1

            # Free intermediate tensors
            del old_slices, full_weight, new_slice

        logger.info(
            "FastTPLoader: loaded %d params, skipped %d (old_tp=%d → new_tp=%d rank=%d)",
            loaded_count, skipped_count, self.old_tp_size, self.new_tp_size, self.new_tp_rank,
        )
        return loaded_count > 0

    def load_from_raw_handles(
        self,
        model: torch.nn.Module,
        raw_handles: Dict[str, Tuple[torch.Tensor, bytes]],
        device: Optional[torch.device] = None,
    ) -> bool:
        """Load from in-memory raw handles (for same-machine IPC).

        This is faster than the file-based approach when both processes
        can communicate directly.
        """
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")

        model_params = dict(model.named_parameters())
        loaded_count = 0

        for param_name, (tensor, _handle_bytes) in raw_handles.items():
            if param_name not in model_params:
                continue
            target_param = model_params[param_name]
            strategy = _get_slice_strategy(param_name)
            new_slice = _slice_weight(tensor, self.new_tp_size, self.new_tp_rank, strategy)

            if new_slice.shape == target_param.shape:
                target_param.data.copy_(new_slice)
                loaded_count += 1
            del new_slice

        logger.info("FastTPLoader (raw): loaded %d params", loaded_count)
        return loaded_count > 0

    def signal_done(self):
        """Signal to the old process that weights have been copied.

        Removes the IPC handle files, allowing the old process to free memory.
        """
        for rank in range(self.old_tp_size):
            path = self.ipc_dir / f"{self.module_type}_{self.perspective}_tp{self.old_tp_size}_rank{rank}.json"
            if path.exists():
                path.unlink()
        logger.info("FastTPLoader: signaled done, IPC files removed.")
