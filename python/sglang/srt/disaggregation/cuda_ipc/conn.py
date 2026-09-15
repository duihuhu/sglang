"""Single-node CUDA IPC KV transfer backend.

The decode process owns and exports the KV pools.  A matching prefill TP rank
opens those allocations and writes the requested pages directly over P2P/NVLink.
ZMQ carries only handles, destination indices and completion notifications.
"""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import socket
import struct
import threading
import time
from typing import List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)
GUARD = b"CudaIpcKV1"
IPC_HANDLE_BYTES = 64


class CudaIpcRuntime(Protocol):
    def export(self, ptr: int, length: int, device: int) -> Union[bytes, Tuple[bytes, int, int]]: ...
    def open(self, handle: bytes, device: int) -> int: ...
    def close(self, ptr: int) -> None: ...
    def copy(self, dst: int, dst_device: int, src: int, src_device: int, length: int) -> None: ...
    def synchronize(self, device: int) -> None: ...
    def can_access_peer(self, src_device: int, dst_device: int) -> bool: ...
    def driver_version(self) -> int: ...


class ExtensionCudaIpcRuntime:
    """Thin wrapper around afd_ipc_cpp raw CUDA IPC primitives."""

    def __init__(self):
        try:
            from sglang.srt.layers.afd_ipc_cpp import get_module

            self._mod = get_module()
        except Exception as exc:
            raise RuntimeError(
                "cuda_ipc PD backend requires the afd_ipc_cpp extension with "
                "CudaIpcMemory support; build sgl-kernel with CUDA enabled"
            ) from exc
        if not hasattr(self._mod, "CudaIpcMemory"):
            raise RuntimeError(
                "afd_ipc_cpp is too old for cuda_ipc PD; rebuild it after updating SGLang"
            )
        self._ipc = self._mod.CudaIpcMemory()

    def export(self, ptr: int, length: int, device: int):
        exported = self._ipc.export_handle(ptr, length, device)
        if isinstance(exported, (bytes, bytearray, memoryview)):
            return bytes(exported)
        handle, offset, allocation_size = exported
        return bytes(handle), int(offset), int(allocation_size)

    def open(self, handle: bytes, device: int) -> int:
        return int(self._ipc.open_handle(handle, device))

    def close(self, ptr: int) -> None:
        self._ipc.close_handle(ptr)

    def copy(self, dst: int, dst_device: int, src: int, src_device: int, length: int) -> None:
        self._ipc.copy_peer_async(dst, dst_device, src, src_device, length)

    def synchronize(self, device: int) -> None:
        self._ipc.synchronize(device)

    def can_access_peer(self, src_device: int, dst_device: int) -> bool:
        return bool(self._ipc.can_access_peer(src_device, dst_device))

    def driver_version(self) -> int:
        return int(self._ipc.driver_version())


@dataclasses.dataclass
class CudaIpcKVArgs(KVArgs):
    hostname: str = dataclasses.field(default_factory=socket.gethostname)
    driver_version: int = 0
    ipc_handles: List[bytes] = dataclasses.field(default_factory=list)
    ipc_offsets: List[int] = dataclasses.field(default_factory=list)
    ipc_allocation_sizes: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class CudaIpcRegistration:
    hostname: str
    endpoint: str
    driver_version: int
    gpu_id: int
    tp_rank: int
    tp_size: int
    kv_handles: Tuple[bytes, ...]
    kv_offsets: Tuple[int, ...]
    kv_allocation_sizes: Tuple[int, ...]
    kv_lens: Tuple[int, ...]
    kv_item_lens: Tuple[int, ...]
    aux_ptrs: Tuple[int, ...]

    def validate_for(self, local_hostname: str, local_driver: int, local_tp_rank: int, local_tp_size: int) -> None:
        if self.hostname != local_hostname:
            raise RuntimeError("cuda_ipc requires prefill and decode on the same host")
        if self.driver_version != local_driver:
            raise RuntimeError("cuda_ipc requires the same NVIDIA driver version")
        if self.tp_size != local_tp_size:
            raise RuntimeError("cuda_ipc phase 1 requires equal prefill/decode attention TP sizes")
        if self.tp_rank != local_tp_rank:
            raise RuntimeError("cuda_ipc requires identity TP rank mapping (P rank i -> D rank i)")
        counts = {
            len(self.kv_handles), len(self.kv_offsets),
            len(self.kv_allocation_sizes), len(self.kv_lens),
            len(self.kv_item_lens),
        }
        if len(counts) != 1 or not self.kv_handles:
            raise RuntimeError("invalid cuda_ipc KV allocation layout")
        if any(len(handle) != IPC_HANDLE_BYTES for handle in self.kv_handles):
            raise RuntimeError("invalid CUDA IPC memory handle size")
        sizes_by_handle = {}
        for handle, allocation_size in zip(self.kv_handles, self.kv_allocation_sizes):
            previous_size = sizes_by_handle.setdefault(handle, allocation_size)
            if previous_size != allocation_size:
                raise RuntimeError("inconsistent cuda_ipc allocation size for shared handle")
        for offset, allocation_size, length in zip(
            self.kv_offsets, self.kv_allocation_sizes, self.kv_lens
        ):
            if (offset < 0 or allocation_size <= 0 or length <= 0
                    or offset > allocation_size
                    or length > allocation_size - offset):
                raise RuntimeError("invalid cuda_ipc KV allocation bounds")


class CudaIpcKVManager(CommonKVManager):
    def __init__(
        self,
        args: CudaIpcKVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
        runtime: Optional[CudaIpcRuntime] = None,
    ):
        self.runtime = runtime or ExtensionCudaIpcRuntime()
        args.hostname = socket.gethostname()
        args.driver_version = self.runtime.driver_version()
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        if self.attn_cp_size != 1 or self.pp_size != 1 or self.system_dp_size != 1:
            raise RuntimeError("cuda_ipc phase 1 supports CP=1, PP=1 and DP=1 only")
        self.local_hostname = args.hostname
        local_tp_rank = args.engine_rank % self.attn_tp_size
        logger.info(
            "CUDA IPC KV backend ready hostname=%s gpu_id=%s tp_rank=%s tp_size=%s",
            self.local_hostname,
            args.gpu_id,
            local_tp_rank,
            self.attn_tp_size,
        )
        self.remote_registrations = {}
        self.opened_remote_ptrs = {}
        self.opened_remote_bases = {}
        self.transfer_infos = {}
        if disaggregation_mode == DisaggregationMode.DECODE:
            exports = [
                self.runtime.export(ptr, length, args.gpu_id)
                for ptr, length in zip(args.kv_data_ptrs, args.kv_data_lens)
            ]
            normalized = [
                (bytes(value), 0, length)
                if isinstance(value, (bytes, bytearray, memoryview)) else value
                for value, length in zip(exports, args.kv_data_lens)
            ]
            args.ipc_handles = [bytes(value[0]) for value in normalized]
            args.ipc_offsets = [int(value[1]) for value in normalized]
            args.ipc_allocation_sizes = [int(value[2]) for value in normalized]
            self._start_decode_status_thread()
        else:
            self._start_prefill_control_thread()

    def _start_prefill_control_thread(self):
        def worker():
            while True:
                msg = self.server_socket.recv_multipart()
                if not msg or msg[0] != GUARD:
                    logger.error("Ignoring foreign cuda_ipc control message")
                    continue
                kind = msg[1]
                if kind == b"REGISTER":
                    reg = decode_registration(msg[2:])
                    try:
                        self.add_registration(reg)
                    except Exception:
                        logger.exception("Failed to register cuda_ipc decode allocation")
                elif kind == b"REQUEST":
                    room = int(msg[2])
                    peer = msg[3].decode()
                    dst_indices = np.frombuffer(msg[4], dtype=np.int32).copy()
                    aux_index = int(msg[5])
                    required = int(msg[6])
                    self.transfer_infos.setdefault(room, {})[peer] = (dst_indices, aux_index)
                    if len(self.transfer_infos[room]) == required:
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=worker, daemon=True).start()

    def _start_decode_status_thread(self):
        def worker():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg and msg[0] == GUARD and msg[1] == b"DONE":
                    room, status = int(msg[2]), int(msg[3])
                    if status == KVPoll.Success:
                        aux_index = int(msg[4])
                        for buffer_index, payload in enumerate(msg[5:]):
                            dst = self.kv_args.aux_data_ptrs[buffer_index]
                            dst += aux_index * self.kv_args.aux_item_lens[buffer_index]
                            ctypes.memmove(dst, payload, len(payload))
                    self.update_status(room, status)
        threading.Thread(target=worker, daemon=True).start()

    def add_registration(self, reg: CudaIpcRegistration):
        local_rank = self.kv_args.engine_rank % self.attn_tp_size
        reg.validate_for(self.local_hostname, self.runtime.driver_version(), local_rank, self.attn_tp_size)
        if not self.runtime.can_access_peer(self.kv_args.gpu_id, reg.gpu_id):
            raise RuntimeError(
                f"CUDA peer access unavailable from prefill GPU {self.kv_args.gpu_id} to decode GPU {reg.gpu_id}"
            )
        key = self._registration_key(reg.hostname, reg.gpu_id)
        self.opened_remote_ptrs.pop(key, None)
        old_bases = self.opened_remote_bases.pop(key, ())
        for ptr in old_bases:
            self.runtime.close(ptr)
        bases_by_handle = {}
        for handle in reg.kv_handles:
            if handle not in bases_by_handle:
                bases_by_handle[handle] = self.runtime.open(handle, self.kv_args.gpu_id)
        ptrs = tuple(
            bases_by_handle[handle] + offset
            for handle, offset in zip(reg.kv_handles, reg.kv_offsets)
        )
        self.remote_registrations[key] = reg
        self.opened_remote_ptrs[key] = ptrs
        self.opened_remote_bases[key] = tuple(bases_by_handle.values())
        logger.info(
            "CUDA IPC KV registration complete remote_gpu_id=%s handles=%s",
            reg.gpu_id,
            len(reg.kv_handles),
        )

    @staticmethod
    def _registration_key(hostname: str, gpu_id: int) -> str:
        # gpu_id is the CUDA ordinal in the process's shared full-node
        # visibility, and therefore identifies the actual peer device.  A TP
        # rank alone is only logical and can collide across server instances.
        return f"{hostname}:{gpu_id}"

    def close(self):
        for bases in self.opened_remote_bases.values():
            for ptr in bases:
                self.runtime.close(ptr)
        self.opened_remote_ptrs.clear()
        self.opened_remote_bases.clear()

    def copy_kv(self, reg: CudaIpcRegistration, src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]):
        key = self._registration_key(reg.hostname, reg.gpu_id)
        dst_ptrs = self.opened_remote_ptrs[key]
        src_blocks, dst_blocks = group_concurrent_contiguous(src_indices, dst_indices)
        src_ptrs, selected_dst_ptrs, layers = (
            self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, list(dst_ptrs))
            if self.is_mla_backend
            else (None, None, None)
        )
        if self.is_mla_backend:
            pairs = zip(src_ptrs[:layers], selected_dst_ptrs[:layers], self.kv_args.kv_item_lens[:layers])
        else:
            sk, sv, dk, dv, layers = self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, list(dst_ptrs))
            pairs = zip(sk[:layers] + sv[:layers], dk[:layers] + dv[:layers], self.kv_args.kv_item_lens[: 2 * layers])
        copied_bytes = 0
        for src_base, dst_base, item_len in pairs:
            for src_block, dst_block in zip(src_blocks, dst_blocks):
                length = len(src_block) * item_len
                self.runtime.copy(
                    dst_base + int(dst_block[0]) * item_len,
                    reg.gpu_id,
                    src_base + int(src_block[0]) * item_len,
                    self.kv_args.gpu_id,
                    length,
                )
                copied_bytes += length
        self.runtime.synchronize(self.kv_args.gpu_id)
        return copied_bytes


class CudaIpcKVSender(CommonKVSender):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.conclude_state = None
        self.transferred_bytes = 0
        self.transfer_started = False

    def send(self, kv_indices, state_indices=None):
        if state_indices:
            raise RuntimeError("cuda_ipc phase 1 does not support hybrid state pools")
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        if self.bootstrap_room not in self.kv_mgr.transfer_infos:
            return
        try:
            for key, (dst_indices, dst_aux_index) in self.kv_mgr.transfer_infos[
                self.bootstrap_room
            ].items():
                reg = self.kv_mgr.remote_registrations[key]
                copied_bytes = self.kv_mgr.copy_kv(
                    reg, kv_indices, dst_indices[index_slice]
                )
                self.transferred_bytes += copied_bytes
                if copied_bytes and not self.transfer_started:
                    logger.info(
                        "CUDA IPC KV transfer started room=%s", self.bootstrap_room
                    )
                    self.transfer_started = True
                if self.curr_idx == self.num_kv_indices:
                    self._notify_done(reg, KVPoll.Success, dst_aux_index)
            if self.curr_idx == self.num_kv_indices:
                logger.info(
                    "CUDA IPC KV transfer complete bytes=%s room=%s",
                    self.transferred_bytes,
                    self.bootstrap_room,
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                self.kv_mgr.transfer_infos.pop(self.bootstrap_room, None)
        except Exception as exc:
            self.kv_mgr.record_failure(self.bootstrap_room, str(exc))
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            raise

    def _notify_done(self, reg, status, dst_aux_index):
        # Decode status endpoint is carried in aux_ptrs as (port,), avoiding another protocol object.
        endpoint = f"tcp://{reg.endpoint}:{reg.aux_ptrs[0]}"
        aux_payloads = []
        if status == KVPoll.Success:
            if self.aux_index is None:
                raise RuntimeError("cuda_ipc requires a metadata source index on the final chunk")
            for ptr, item_len in zip(
                self.kv_mgr.kv_args.aux_data_ptrs,
                self.kv_mgr.kv_args.aux_item_lens,
            ):
                src = ptr + self.aux_index * item_len
                aux_payloads.append(bytes((ctypes.c_byte * item_len).from_address(src)))
        self.kv_mgr._connect(endpoint).send_multipart(
            [
                GUARD,
                b"DONE",
                str(self.bootstrap_room).encode(),
                str(status).encode(),
                str(dst_aux_index).encode(),
                *aux_payloads,
            ]
        )

    def poll(self):
        return self.conclude_state or self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        self.conclude_state = KVPoll.Failed
        raise RuntimeError(self.kv_mgr.failure_records.get(self.bootstrap_room, "cuda_ipc KV transfer failed"))


class CudaIpcKVReceiver(CommonKVReceiver):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room=None, prefill_dp_rank=None):
        self.conclude_state = None
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _register_kv_args(self):
        reg = CudaIpcRegistration(
            hostname=self.kv_mgr.local_hostname,
            endpoint=self.kv_mgr.local_ip,
            driver_version=self.kv_mgr.kv_args.driver_version,
            gpu_id=self.kv_mgr.kv_args.gpu_id,
            tp_rank=self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size,
            tp_size=self.kv_mgr.attn_tp_size,
            kv_handles=tuple(self.kv_mgr.kv_args.ipc_handles),
            kv_offsets=tuple(self.kv_mgr.kv_args.ipc_offsets),
            kv_allocation_sizes=tuple(self.kv_mgr.kv_args.ipc_allocation_sizes),
            kv_lens=tuple(self.kv_mgr.kv_args.kv_data_lens),
            kv_item_lens=tuple(self.kv_mgr.kv_args.kv_item_lens),
            aux_ptrs=(self.kv_mgr.rank_port,),
        )
        payload = encode_registration(reg)
        for info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(info)
            with lock:
                sock.send_multipart([GUARD, b"REGISTER", *payload])

    def init(self, kv_indices, aux_index=None, state_indices=None):
        if state_indices:
            raise RuntimeError("cuda_ipc phase 1 does not support hybrid state pools")
        for info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(info)
            with lock:
                sock.send_multipart([
                    GUARD, b"REQUEST", str(self.bootstrap_room).encode(),
                    self.kv_mgr._registration_key(
                        self.kv_mgr.local_hostname, self.kv_mgr.kv_args.gpu_id
                    ).encode(),
                    kv_indices.tobytes(), str(aux_index).encode(), str(self.required_dst_info_num).encode(),
                ])
        self.init_time = time.time()

    def poll(self):
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        elif self.init_time and time.time() - self.init_time >= self.kv_mgr.waiting_timeout:
            self.conclude_state = KVPoll.Failed
        return self.conclude_state or status

    def failure_exception(self):
        raise RuntimeError(self.kv_mgr.failure_records.get(self.bootstrap_room, "cuda_ipc KV transfer failed"))


class CudaIpcKVBootstrapServer(CommonKVBootstrapServer):
    pass

def encode_registration(reg: CudaIpcRegistration) -> List[bytes]:
    return [
        reg.hostname.encode(), reg.endpoint.encode(),
        str(reg.driver_version).encode(), str(reg.gpu_id).encode(),
        str(reg.tp_rank).encode(), str(reg.tp_size).encode(),
        b"".join(reg.kv_handles),
        struct.pack(f"<{len(reg.kv_offsets)}Q", *reg.kv_offsets),
        struct.pack(f"<{len(reg.kv_allocation_sizes)}Q", *reg.kv_allocation_sizes),
        struct.pack(f"<{len(reg.kv_lens)}Q", *reg.kv_lens),
        struct.pack(f"<{len(reg.kv_item_lens)}Q", *reg.kv_item_lens),
        struct.pack(f"<{len(reg.aux_ptrs)}Q", *reg.aux_ptrs),
    ]


def decode_registration(parts: Sequence[bytes]) -> CudaIpcRegistration:
    if len(parts) != 12:
        raise RuntimeError("invalid cuda_ipc registration field count")
    handles_blob, offsets_blob, sizes_blob, lens_blob, item_blob, aux_blob = parts[6:12]
    if len(handles_blob) % IPC_HANDLE_BYTES:
        raise RuntimeError("invalid CUDA IPC memory handle size")
    handles = tuple(
        handles_blob[i : i + IPC_HANDLE_BYTES]
        for i in range(0, len(handles_blob), IPC_HANDLE_BYTES)
    )
    def unpack_q(blob):
        if len(blob) % 8:
            raise RuntimeError("invalid cuda_ipc registration integer payload")
        return tuple(struct.unpack(f"<{len(blob) // 8}Q", blob)) if blob else ()
    return CudaIpcRegistration(
        hostname=parts[0].decode(), endpoint=parts[1].decode(),
        driver_version=int(parts[2]), gpu_id=int(parts[3]),
        tp_rank=int(parts[4]), tp_size=int(parts[5]), kv_handles=handles,
        kv_offsets=unpack_q(offsets_blob),
        kv_allocation_sizes=unpack_q(sizes_blob),
        kv_lens=unpack_q(lens_blob), kv_item_lens=unpack_q(item_blob),
        aux_ptrs=unpack_q(aux_blob),
    )
