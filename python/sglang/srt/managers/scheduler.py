# Copyright 2023-2024 SGLang Team
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
"""A scheduler that manages a tensor parallel GPU worker."""

import faulthandler
import logging
import os
import signal
import sys
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

import psutil
import setproctitle
import torch
import torch.distributed
import zmq
from torch.cuda import Stream as CudaStream
from torch.distributed import barrier

from sglang.jit_kernel.ngram_embedding import update_token_table
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.encode_receiver import create_mm_receiver
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
    release_req_to_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.mamba.ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.layers.dp_attention import (
    compute_dp_attention_world_info,
    get_attention_cp_group,
    get_attention_tp_group,
)
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BaseBatchReq,
    BaseReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    CheckWeightsReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadsReqInput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    PauseGenerationReqInput,
    PinPrefixReqInput,
    PinPrefixReqOutput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.mm_utils import init_mm_embedding_cache, unwrap_shm_features
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    ModelWorkerBatch,
    MultimodalInputs,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_dp_attn_mixin import SchedulerDPAttnMixin
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    SchedulerRuntimeCheckerMixin,
    create_scheduler_watchdog,
)
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.managers.session_controller import SessionController
from sglang.srt.managers.utils import GenerationBatchResult, validate_input_length
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.observability.req_time_stats import (
    real_time,
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.scheduler_metrics_mixin import (
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsMixin,
)
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs, get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    get_numa_node,
    is_mps,
    kill_itself_when_parent_died,
    numa_bind_to_node,
    point_to_point_pyobj,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.srt.utils.common import is_npu
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

if is_mps():
    CudaStreamContext = nullcontext
else:
    from torch.cuda import StreamContext as CudaStreamContext

logger = logging.getLogger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()

_is_npu = is_npu()


@dataclass
class EmbeddingBatchResult:
    embeddings: torch.Tensor
    copy_done: Optional[torch.cuda.Event] = None

    def copy_to_cpu(self):
        """Copy embeddings tensor to CPU in overlap scheduling."""

        if isinstance(self.embeddings, torch.Tensor):
            self.copy_done = torch.get_device_module(self.embeddings.device).Event()
            self.embeddings = self.embeddings.to("cpu", non_blocking=True)
        else:
            assert isinstance(self.embeddings, list)
            if len(self.embeddings) == 0:
                return

            self.copy_done = torch.get_device_module(self.embeddings[0].device).Event()
            self.embeddings = [
                emb.to("cpu", non_blocking=True) for emb in self.embeddings
            ]

        self.copy_done.record()


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerRuntimeCheckerMixin,
    SchedulerPPMixin,
    SchedulerDPAttnMixin,
    SchedulerDllmMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ):
        self.is_initializing = True
        self.init_soft_watchdog(server_args)

        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.dp_rank = dp_rank
        self.tp_size = server_args.tp_size
        self.moe_ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.dp_size = server_args.dp_size
        self.nccl_port = port_args.nccl_port
        self.schedule_policy = server_args.schedule_policy
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.abort_on_priority_when_disabled = (
            server_args.abort_on_priority_when_disabled
        )
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold
        )
        self.enable_lora = server_args.enable_lora
        self.enable_lora_overlap_loading = server_args.enable_lora_overlap_loading
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.afd_overlap_enabled = getattr(
            server_args, "afd_enable_overlap_schedule", False
        )
        if self.afd_overlap_enabled:
            self.enable_overlap = True
        self.enable_pdmux = server_args.enable_pdmux
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.page_size = server_args.page_size
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()

        # Distributed rank info
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
                self.attn_cp_size,
            )
        )

        self.enable_kv_cache_events = bool(
            server_args.kv_events_config and self.attn_tp_rank == 0
        )

        # Init model configs
        self.init_model_config()

        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)

        # Init inter-process communication
        self.init_ipc_channels(port_args)

        # Init PD-multiplexing context
        if self.enable_pdmux:
            self.init_pdmux()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config and GEMM config (FP8 GEMM, etc.)
        self.init_moe_gemm_config()

        # Init mamba backend
        self.init_mamba_backend()

        # Launch a model worker and draft model worker if using speculative decoding
        self.init_model_worker()

        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)

        # Init cache and memory pool
        self.init_cache_with_memory_pool()

        # Init running status
        self.init_running_status()

        # Init chunked prefill
        self.init_chunked_prefill()

        # Init diffusion LLM
        self.init_diffusion_llm()

        # Init schedule policy and new token estimation
        self.init_schedule_policy()

        # Init watchdog, memory saver, input blocker and recv skipper
        self.init_watch_dog_memory_saver_input_blocker()

        # Init profiler
        self.init_profiler()

        # Init prefill-decodedisaggregation
        self.init_disaggregation()

        # Init overlap schedule
        self.init_overlap()

        # Init Ngram Embedding
        self.maybe_init_ngram_embedding()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

        # Init request dispatcher
        self.init_request_dispatcher()

        # Init LoRA overlap loader
        if self.enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

        # Init the grammar backend for constrained generation
        self.grammar_manager = GrammarManager(self)

        self.is_initializing = False

    def init_model_config(self):
        self.model_config = ModelConfig.from_server_args(self.server_args)
        if _is_npu:
            # make sure the page size is not larger than block_size and chunked_prefill_size on NPU backend
            # the npu backend request the defined page size to be no larger than block_size and chunked_prefill_size
            from sglang.srt.dllm.config import DllmConfig

            self.dllm_config = (  # For diffusion LLM
                DllmConfig.from_server_args(self.server_args)
                if self.server_args.dllm_algorithm is not None
                else None
            )
            if self.dllm_config:
                if self.dllm_config.block_size < self.page_size:
                    logger.warning(
                        "WARNING: "
                        f"The page size {self.page_size} should not be larger than dllm block size {self.dllm_config.block_size}."
                        f"Page size now falls back to {self.dllm_config.block_size}"
                    )
                    self.page_size = self.dllm_config.block_size

    def init_ipc_channels(self, port_args: PortArgs):
        context = zmq.Context(2)
        self.idle_sleeper = None

        if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if self.server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            self.send_to_tokenizer = SenderWrapper(send_to_tokenizer)
            self.send_to_detokenizer = SenderWrapper(send_to_detokenizer)

            if self.server_args.sleep_on_idle:
                self.idle_sleeper = IdleSleeper(
                    [
                        self.recv_from_tokenizer,
                        self.recv_from_rpc,
                    ]
                )
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SenderWrapper(None)
            self.send_to_detokenizer = SenderWrapper(None)

        if self.current_scheduler_metrics_enabled:
            self.send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )

        # AFD inter-scheduler channels (C5: configurable ports)
        self.afd_send_to_ffn = None
        self.afd_recv_from_attn = None
        from sglang.srt.layers.afd_type import AFDPerspective

        afd_perspective = getattr(self.server_args, "afd_perspective", None)
        if self.pp_rank == 0 and self.attn_tp_rank == 0:
            host = os.getenv("AFD_SCHED_HOST", "127.0.0.1")
            port = int(os.getenv("AFD_SCHED_PORT", "65300"))
            afd_ipc = f"tcp://{host}:{port}"
            if afd_perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
                self.afd_send_to_ffn = get_zmq_socket(
                    context, zmq.PUSH, afd_ipc, False
                )
            elif afd_perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
                self.afd_recv_from_attn = get_zmq_socket(
                    context, zmq.PULL, afd_ipc, True
                )

        # AFD DVFS (Tier 2 energy-aware frequency scaling)
        self._af_dvfs_ctrl = None
        self._dvfs_hw = None
        self._cur_f_a = 1410
        self._cur_f_f = 1410
        if getattr(self.server_args, "afd_dvfs_enabled", False):
            self._init_afd_dvfs(self.server_args)

        # Tier 1: Joint ILP resource planning
        self._tier1_solver = None
        self._tier1_solution = None
        self._tier1_monitor = None
        self._tier1_collector = None
        self._tier1_last_monitor_time = 0.0
        if getattr(self.server_args, "enable_tier1_pa", False):
            self._init_afd_tier1(self.server_args)

    def init_tokenizer(self):
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    use_fast=not server_args.disable_fast_image_processor,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

    def init_mamba_backend(self) -> None:
        initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        # For the MM models, check the text_config for MoE settings
        config_to_check = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config
        )

        if hasattr(config_to_check, "num_experts_per_tok"):
            initialize_moe_config(self.server_args)

        # Initialize GEMM-related configuration for FP8 and FP4 backends.
        initialize_fp8_gemm_config(self.server_args)
        initialize_fp4_gemm_config(self.server_args)

        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync(self.server_args)

    def init_tp_model_worker(self):
        from sglang.srt.managers.tp_worker import TpModelWorker

        self.tp_worker = TpModelWorker(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            pp_rank=self.pp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
            dp_rank=self.dp_rank,
            nccl_port=self.nccl_port,
        )

    def maybe_init_draft_worker(self):
        if self.spec_algorithm.is_none():
            self.draft_worker = None
            return

        # Launch a draft worker for speculative decoding
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=self.dp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
        )

        if self.server_args.speculative_draft_load_format is not None:
            self.server_args.load_format = (
                self.server_args.speculative_draft_load_format
            )
            logger.info(
                f"Using draft model load_format: '{self.server_args.speculative_draft_load_format}'"
            )

        DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)
        self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)

    def init_model_worker(self):
        self.init_tp_model_worker()
        self.maybe_init_draft_worker()

        # Dispatch the model worker
        if self.spec_algorithm.is_none():
            self.model_worker = self.tp_worker
        else:
            self.model_worker = self.draft_worker

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.forward_stream,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if get_global_server_args().pp_max_micro_batch_size is None:
            get_global_server_args().pp_max_micro_batch_size = max(
                self.max_running_requests // self.pp_size, 1
            )

        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_attention_cp_group()
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # NOTE: dp_tp_* are request/data-plane coordination groups (not tensor collectives).
        # When DP attention is enabled, scope to the attention-TP group; otherwise use
        # the base TP group. Entry rank is the local rank 0 in that group.
        # Use the CPU (gloo) group to broadcast VLM Python objects and avoid CUDA
        # stream/device coupling (#11910).
        self.dp_tp_group = (
            self.attn_tp_group
            if self.server_args.enable_dp_attention
            else self.tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Print debug info
        if self.tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={self.server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        if self.enable_metrics and hasattr(self, "metrics_collector"):
            self.metrics_collector.emit_cache_config_info(
                self.page_size, self.max_total_num_tokens // self.page_size
            )

    def init_cache_with_memory_pool(self):
        server_args = self.server_args

        # Hybrid memory pool
        self.is_hybrid_swa = self.tp_worker.is_hybrid_swa
        self.is_hybrid_ssm = (
            self.tp_worker.model_runner.hybrid_gdn_config is not None
            or self.tp_worker.model_runner.mamba2_config is not None
        )

        self.sliding_window_size = None
        if self.is_hybrid_swa:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        # Create cache
        params = CacheInitParams(
            disable=server_args.disable_radix_cache,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            page_size=self.page_size,
            is_eagle=self.spec_algorithm.is_eagle(),
            tp_cache_group=(
                self.attn_tp_cpu_group
                if self.server_args.enable_dp_attention
                else self.tp_cpu_group
            ),
            eviction_policy=server_args.radix_eviction_policy,
            enable_metrics=self.enable_metrics,
            enable_kv_cache_events=self.enable_kv_cache_events,
            enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            chunked_prefill_size=server_args.chunked_prefill_size,
            sliding_window_size=self.sliding_window_size,
        )

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            if not self.is_hybrid_swa:
                from sglang.srt.mem_cache.chunk_cache import ChunkCache

                self.tree_cache = ChunkCache(params)
            else:
                from sglang.srt.mem_cache.chunk_cache import SWAChunkCache

                self.tree_cache = SWAChunkCache(params)
        else:

            if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
                # lazy import to avoid JIT overhead
                from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

                logger.info("Using experimental C++ radix tree implementation.")
                self.tree_cache = RadixCacheCpp(params=params, server_args=server_args)
            elif self.enable_hierarchical_cache:
                if self.is_hybrid_ssm:
                    from sglang.srt.mem_cache.hi_mamba_radix_cache import (
                        HiMambaRadixCache,
                    )

                    self.tree_cache = HiMambaRadixCache(
                        params=params, server_args=server_args
                    )
                else:
                    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

                    self.tree_cache = HiRadixCache(
                        params=params, server_args=server_args
                    )
                self.tp_worker.register_hicache_layer_transfer_counter(
                    self.tree_cache.cache_controller.layer_done_counter
                )
            elif self.is_hybrid_swa:
                from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

                self.tree_cache = SWARadixCache(params=params)
            elif self.is_hybrid_ssm:
                from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

                self.tree_cache = MambaRadixCache(params)
            elif server_args.enable_lmcache:
                from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
                    LMCRadixCache,
                )

                self.tree_cache = LMCRadixCache(
                    params=params,
                    model_config=self.model_config,
                    tp_size=self.tp_size,
                    rank=self.tp_rank,
                    tp_group=self.tp_group,
                )
            else:
                self.tree_cache = RadixCache(params)

        if server_args.enable_streaming_session:

            self.tree_cache = SessionAwareCache(self.tree_cache)

        if (
            server_args.disaggregation_mode == "decode"
            and server_args.disaggregation_decode_enable_offload_kvcache
        ):
            self.decode_offload_manager = DecodeKVCacheOffloadManager(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=params.tp_cache_group,
                tree_cache=self.tree_cache,
                server_args=self.server_args,
            )
        else:
            self.decode_offload_manager = None

        embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
        init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

    def init_running_status(self):
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.return_health_check_ipcs: Deque[Optional[str]] = deque()
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.session_controller = SessionController(self.tree_cache)
        self.forward_sleep_time = None
        self._engine_paused = False

    def init_chunked_prefill(self):
        # Init chunked prefill
        self.chunked_prefill_size = self.server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None
            and self.server_args.enable_mixed_chunk
        )

        # Init the dynamic chunking predictor for PP
        self.enable_dynamic_chunking = (
            self.server_args.enable_dynamic_chunking and self.pp_size > 1
        )
        if self.enable_dynamic_chunking:
            try:
                self.profile_and_init_predictor()
            except Exception as e:
                logger.warning(
                    f"[PP Dynamic Chunk] Failed to profile prefill latency: {e}. "
                    "Dynamic chunking will be disabled."
                )
                self.enable_dynamic_chunking = False

    def init_schedule_policy(self):
        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.max_prefill_bs: int = 0
        if self.server_args.enable_prefill_delayer:
            self.prefill_delayer = PrefillDelayer(
                dp_size=self.dp_size,
                attn_tp_size=self.attn_tp_size,
                cpu_group=self.tp_cpu_group,
                server_args=self.server_args,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
                max_delay_passes=self.server_args.prefill_delayer_max_delay_passes,
                token_usage_low_watermark=self.server_args.prefill_delayer_token_usage_low_watermark,
                device=(
                    self.tp_group.device
                    if self.server_args.disable_overlap_schedule
                    else "cpu"
                ),
            )

        # NOTE: preemption is enabled by default for priority scheduling.
        self.enable_priority_preemption = (
            self.enable_priority_scheduling
            and not self.server_args.disable_priority_preemption
        )

        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * self.server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        self.new_token_ratio = self.init_new_token_ratio

    def init_soft_watchdog(self, server_args: ServerArgs):
        if (x := server_args.soft_watchdog_timeout) is not None:
            self.soft_watchdog = create_scheduler_watchdog(
                self, watchdog_timeout=x, soft=True
            )

    def init_watch_dog_memory_saver_input_blocker(self):
        # Start watchdog thread
        self.watchdog = create_scheduler_watchdog(
            self, watchdog_timeout=self.server_args.watchdog_timeout
        )

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )
        self.offload_tags = set()

        # Init recv skipper and input blocker
        self.recv_skipper = SchedulerRecvSkipper.maybe_create(self.server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Configure GC logger
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

    def init_disaggregation(self):
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        # FFN side in AFD mode does not participate in KV cache transfer,
        # but needs disagg queues for batch scheduling (prebuilt path).
        # Skip KV manager creation (no bootstrap/ZMQ) but keep everything else.
        from sglang.srt.layers.afd import afd_is_ffn as _afd_is_ffn_disagg
        _is_ffn = _afd_is_ffn_disagg()

        if _is_ffn:
            self._init_disagg_ffn_stubs()
            return

        if self.draft_worker is None or self.spec_algorithm.is_ngram():
            draft_token_to_kv_pool = None
        elif self.spec_algorithm.supports_spec_v2() and self.enable_overlap:
            if self.server_args.enable_multi_layer_eagle:
                draft_runner = self.draft_worker.draft_worker.draft_runner_list[0]
            else:
                draft_runner = self.draft_worker.draft_worker.draft_runner
            draft_token_to_kv_pool = draft_runner.token_to_kv_pool
            model_config = draft_runner.model_config
        else:
            # todo: should we fix this when enabling mtp or it doesn't matter since we only enable mtp in decode node thus we don't transfer draft kvs between P and D?
            draft_token_to_kv_pool = self.draft_worker.model_runner.token_to_kv_pool
            model_config = self.draft_worker.model_config

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *2 for the headroom.
            buffer_size = (self.req_to_token_pool.size) * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=(
                    model_config.hidden_size
                    if self.spec_algorithm.is_eagle()
                    else 16  # minimal padding size for RDMA
                ),
                hidden_states_dtype=(
                    model_config.dtype
                    if self.spec_algorithm.is_eagle()
                    else torch.float32
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                pp_rank=self.pp_rank,
                num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=(
                    model_config.hidden_size
                    if self.spec_algorithm.is_eagle()
                    or self.spec_algorithm.is_standalone()
                    else 16  # minimal padding size for RDMA
                ),
                hidden_states_dtype=(
                    model_config.dtype
                    if self.spec_algorithm.is_eagle()
                    or self.spec_algorithm.is_standalone()
                    else torch.float32
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                scheduler=self,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []

        # Init mm receiver for EPD disaggregation mode
        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                hf_config=self.model_config.hf_config,
                pp_rank=self.pp_rank,
                tp_rank=self.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )

    def _init_disagg_ffn_stubs(self):
        """Set stub attributes for FFN side so metrics/stats code doesn't crash.

        FFN does not participate in KV cache transfer, but shared code paths
        (report_prefill_stats, self_check_during_idle, abort handling, etc.)
        may reference these attributes.
        """

        class _StubQueue:
            """Minimal stub that mimics queue-like attributes."""
            queue = []
            retracted_queue = []
            num_tokens_pre_allocated = 0

            @staticmethod
            def pop_bootstrapped():
                return []

            @staticmethod
            def add(*args, **kwargs):
                pass

            @staticmethod
            def resume_retracted_reqs():
                return []

        self.disagg_prefill_bootstrap_queue = _StubQueue()
        self.disagg_prefill_inflight_queue = []
        self.disagg_decode_prealloc_queue = _StubQueue()
        self.disagg_decode_transfer_queue = _StubQueue()

    def init_overlap(self):
        self.device_module = torch.get_device_module(self.device)

        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        if not self.enable_overlap:
            self.future_map = None
            return

        self.future_map = FutureMap(
            self.max_running_requests,
            self.chunked_prefill_size,
            self.model_config.context_len,
            self.device,
            self.spec_algorithm,
        )
        self.batch_record_buf = [None] * 2
        self.batch_record_ct = 0

    def maybe_init_ngram_embedding(self):
        self.use_ngram_embedding = self.tp_worker.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            self.token_table = self.tp_worker.model_runner.token_table
            hf_config = self.tp_worker.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k

    def _maybe_prepare_ngram_embedding(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[ScheduleBatch]:
        """Fill the token table for ngram embedding before a forward pass."""
        if batch is None or not self.use_ngram_embedding:
            return batch
        batch.ne_token_table = self.token_table
        if batch.forward_mode == ForwardMode.EXTEND:
            all_tokens = []
            column_starts = []
            request_lengths = []
            for req in batch.reqs:
                start = len(req.prefix_indices)
                end = start + req.extend_input_len
                fill_ids = req.origin_input_ids + req.output_ids
                if start == 0:
                    tokens = fill_ids[start:end]
                    column_starts.append(0)
                elif start < self.ngram_embedding_n:
                    tokens = fill_ids[0:end]
                    column_starts.append(0)
                else:
                    # Prepend n-1 tokens before prefix_len for n-gram context
                    tokens = fill_ids[start - self.ngram_embedding_n + 1 : end]
                    column_starts.append(start - self.ngram_embedding_n + 1)
                all_tokens.extend(tokens)
                request_lengths.append(len(tokens))
            dtype = self.token_table.dtype
            device = self.token_table.device
            update_token_table(
                ne_token_table=self.token_table,
                tokens=torch.tensor(all_tokens, dtype=dtype, device=device),
                row_indices=batch.req_pool_indices,
                column_starts=torch.tensor(
                    column_starts, dtype=torch.int32, device=device
                ),
                req_lens=torch.tensor(
                    request_lengths, dtype=torch.int32, device=device
                ),
                ignore_tokens=None,
            )
        return batch

    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        if not self.server_args.enable_deterministic_inference:
            self.truncation_align_size = None
            return

        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
            "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
        }
        env_var, default_size = backend_sizes.get(
            self.server_args.attention_backend, (None, None)
        )
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )

    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (PinPrefixReqInput, self.pin_prefix_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (DestroyWeightsUpdateGroupReqInput, self.destroy_weights_update_group),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (UpdateWeightsFromIPCReqInput, self.update_weights_from_ipc),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (CheckWeightsReqInput, self.check_weights),
                (SlowDownReqInput, self.slow_down),
                (ProfileReq, self.profile),
                (FreezeGCReq, self.handle_freeze_gc),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (
                    LoadLoRAAdapterFromTensorsReqInput,
                    self.load_lora_adapter_from_tensors,
                ),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (GetLoadReqInput, self.get_load),
                (GetLoadsReqInput, self.get_loads),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (DumperControlReqInput, self.handle_dumper_control),
            ]
        )

    def _abort_on_running_timeout(self):
        # NOTE: this should be called before a batch is launched,
        # as current spec-v1 still filters batch inside verify stage.
        timeout_s = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()
        if timeout_s <= 0:
            return
        if self.running_batch.is_empty():
            return

        deadline = time.perf_counter() - timeout_s
        for req in self.running_batch.reqs:
            if not req.finished() and 0 < req.time_stats.forward_entry_time < deadline:
                req.to_finish = FINISH_ABORT(
                    "Request running timeout reached.", HTTPStatus.SERVICE_UNAVAILABLE
                )

    def get_init_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake.

        This method provides the initialization info needed by the tokenizer manager
        and other components to verify the scheduler is ready.
        """
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_req_input_len": self.max_req_input_len,
        }

        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            (
                remote_instance_transfer_engine_session_id,
                remote_instance_transfer_engine_weights_info_dict,
            ) = self.get_remote_instance_transfer_engine_info()
            result_dict.update(
                {
                    "tp_rank": self.tp_rank,
                    "remote_instance_transfer_engine_session_id": remote_instance_transfer_engine_session_id,
                    "remote_instance_transfer_engine_weights_info_dict": remote_instance_transfer_engine_weights_info_dict,
                }
            )

        return result_dict

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop.

        Sets up the schedule stream and dispatches to the appropriate event loop.
        The event loop blocks until shutdown.
        """
        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # No-op for CPU
        with self.device_module.StreamContext(self.schedule_stream):
            dispatch_event_loop(self)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self.cancel_bubble_timer()
                continue

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                self.self_check_during_idle()

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    @DynamicGradMode()
    def event_loop_afd(self):
        """AFD scheduler loop with zmq.Poller (optimization S2)."""
        from sglang.srt.layers.afd import (
            afd_is_attn,
            afd_is_ffn,
            get_afd_micro_batch,
            get_afd_perspective,
        )
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.disaggregation.utils import DisaggregationMode

        disagg_mode = self.disaggregation_mode

        logger.info(
            "event_loop_afd: role=%s m=%s disagg=%s",
            get_afd_perspective(),
            get_afd_micro_batch(),
            disagg_mode,
        )

        self._afd_batchsize_attn = None
        self._afd_forward_mode = None
        self._afd_req_ids = None

        # Eager-init AF communicator (UCX/StepMesh) so FFN listener is
        # ready before Attn attempts to connect.
        # When --afd-async-schedule is on, eager-init the per-mb channels
        # instead of the legacy single comm, for the same reason: the FFN
        # side must be listening on every per-mb endpoint before the Attn
        # side connects on its first forward.
        if getattr(self.server_args, "afd_async_schedule", False):
            # Interleaved schedule uses the same single shared channel as
            # the batch schedule — no per-mb channels needed.  Just init
            # the regular communicator.
            from sglang.srt.layers.afd import get_async_communicator
            try:
                get_async_communicator()
                logger.info(
                    "event_loop_afd: AF communicator ready (interleaved, %s, M=%d)",
                    get_afd_perspective(),
                    int(self.server_args.afd_micro_batch),
                )
            except Exception as e:
                logger.error(
                    "event_loop_afd: AF communicator init failed: %s", e
                )
                raise RuntimeError(
                    f"AF communicator init failed in event_loop_afd: {e}"
                ) from e
        else:
            from sglang.srt.layers.afd import get_async_communicator
            try:
                get_async_communicator()
                logger.info("event_loop_afd: AF communicator ready (%s)",
                            get_afd_perspective())
            except Exception as e:
                logger.error("event_loop_afd: AF communicator init failed: %s", e)
                raise RuntimeError(
                    f"AF communicator init failed in event_loop_afd: {e}"
                ) from e

        # S2: use Poller instead of busy-wait
        afd_poller = None
        if afd_is_ffn() and self.afd_recv_from_attn is not None:
            afd_poller = zmq.Poller()
            afd_poller.register(self.afd_recv_from_attn, zmq.POLLIN)

        _afd_loop_iter = 0

        def _recv_afd_messages():
            """Poll for messages from Attn scheduler."""
            if self.afd_recv_from_attn is None:
                return []
            extra_reqs = []
            while True:
                try:
                    msg = self.afd_recv_from_attn.recv_pyobj(zmq.NOBLOCK)
                    extra_reqs.append(msg)
                except zmq.ZMQError:
                    break
            return extra_reqs

        def _prepare_afd_overlap(batch):
            """Compute microbatch split points for AFD."""
            m = get_afd_micro_batch()
            if batch.batch_size() < m:
                batch.afd_split_seq_index = None
                return

            from sglang.srt.batch_overlap.afd_overlap import (
                _split_seq_indices_m_way,
            )

            # In PD+AF prefill mode, disable M>1 for EXTEND batches
            is_pd_prefill = disagg_mode == DisaggregationMode.PREFILL

            forward_mode = batch.forward_mode
            if forward_mode == ForwardMode.EXTEND:
                if is_pd_prefill:
                    batch.afd_split_seq_index = None
                    return
                extend_lens = batch.extend_lens
                split_indices = _split_seq_indices_m_way(
                    len(extend_lens), m, extend_lens
                )
            elif forward_mode.is_decode() or forward_mode.is_target_verify():
                split_indices = _split_seq_indices_m_way(
                    batch.batch_size(), m, None
                )
            else:
                batch.afd_split_seq_index = None
                return

            batch.afd_split_seq_index = split_indices

        # F3: CPU/GPU overlap scheduling support
        afd_overlap = self.afd_overlap_enabled
        if afd_overlap:
            self.result_queue: Deque = deque()

        def _pop_and_process():
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            _afd_loop_iter += 1
            recv_reqs = self.recv_requests()

            # Step 3.6: FFN also receives AFD messages
            if afd_is_ffn():
                extra_reqs = _recv_afd_messages()
                if extra_reqs:
                    from sglang.srt.layers.afd_mixin import _afd_sched_ts
                    _afd_sched_ts["zmq_recv"] = time.time()
                    recv_reqs = recv_reqs + extra_reqs
                # Re-broadcast so all TP ranks see the merged requests.
                # Must be outside `if extra_reqs` — all ranks must participate.
                if self.tp_size > 1 and not self.server_args.enable_dp_attention:
                    from sglang.srt.utils.common import broadcast_pyobj

                    recv_reqs = broadcast_pyobj(
                        recv_reqs,
                        self.tp_group.rank,
                        self.tp_cpu_group,
                        src=self.tp_group.ranks[0],
                    )

            self._afd_process_input_requests(recv_reqs)

            # FFN waits until Attn sends batch info
            if afd_is_ffn() and self._afd_batchsize_attn is None:
                if afd_poller is not None:
                    afd_poller.poll(timeout=10)

                # If running_batch is non-empty but Attn stopped sending
                # batch info, the requests are stale — clean them up.
                if (
                    self.running_batch is not None
                    and not self.running_batch.is_empty()
                ):
                    self._afd_ffn_cleanup_all()
                continue

            # FFN side: clear stale chunked_req — we force extend_lens=[1] after
            # batch creation so the chunking mechanism (which operates on original
            # extend_lens) would leave a dangling chunked_req that doesn't match
            # the FFN-side KV pool (smaller than Attn's).
            if afd_is_ffn():
                self.chunked_req = None

            batch = self._afd_get_next_batch(disagg_mode)
            self.cur_batch = batch
            disable_overlap_for_batch = False

            # ── FFN side: force-sync batch to match Attn's AFDReqInput ──
            if afd_is_ffn() and batch is not None:
                # UCX always brings [1, hidden_size] per-layer regardless of
                # the Attn-side batch size.  Force extend_lens=1 per req so
                # embedding + logits-extraction stay in sync.
                bs = batch.batch_size()
                if bs > 0:
                    batch.extend_lens = [1] * bs
                    for r in batch.reqs:
                        r.extend_input_len = 1

                # Deduplicate reqs by rid (merge in get_next_batch_to_run
                # can produce duplicates when running_batch is empty).
                seen = set()
                deduped = []
                for r in batch.reqs:
                    if r.rid not in seen:
                        seen.add(r.rid)
                        deduped.append(r)
                if len(deduped) < bs:
                    batch.reqs = deduped

            # F3: overlap — process last batch immediately if overlap disabled for this batch
            if afd_overlap:
                disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)
                if disable_overlap_for_batch:
                    _pop_and_process()

            if batch:
                # Attn side: notify FFN about current batch
                SchedulerAFDMixin.afd_send_batch_info(self, batch)
                # Record ZMQ-send wall-clock for cross-GPU latency breakdown
                if afd_is_attn():
                    from sglang.srt.layers.afd_mixin import _afd_sched_ts
                    _afd_sched_ts["zmq_sent"] = time.time()

                is_decode = batch.forward_mode.is_decode()
                bsz = batch.batch_size()
                perspective = "FFN" if afd_is_ffn() else "ATTN"
                print(f"[AFD_DBG] === ITER {_afd_loop_iter} {perspective} batch={bsz} max_running={self.max_running_requests} ===", flush=True)

                self._tier1_record_batch_start(is_prefill=not is_decode)
                self._afd_dvfs_before_batch(batch)
                _prepare_afd_overlap(batch)
                batch_result = self.run_batch(batch)

                if afd_overlap:
                    self.result_queue.append((batch.copy(), batch_result))
                else:
                    self.process_batch_result(batch, batch_result)

                t_iter = (time.perf_counter() - self._last_decode_batch_time) * 1e6 \
                    if is_decode and hasattr(self, "_last_decode_batch_time") and self._last_decode_batch_time is not None \
                    else 0.0
                self._tier1_record_batch(batch, is_decode, t_iter)

                # Reset AFD state for next iteration
                self._afd_batchsize_attn = None
                self._afd_forward_mode = None
                self._afd_req_ids = None
            else:
                batch_result = None
                self._afd_batchsize_attn = None
                self._afd_forward_mode = None
                self._afd_req_ids = None
                if afd_overlap:
                    self.cancel_bubble_timer()
                else:
                    # FFN side: force-clean any remaining stale requests before
                    # the idle memory check, since no more AFDReqInput will arrive.
                    if afd_is_ffn():
                        self._afd_ffn_cleanup_all()
                    self.self_check_during_idle()

            # F3: overlap — process last batch (while GPU runs current batch)
            if afd_overlap:
                if self.last_batch:
                    if not disable_overlap_for_batch:
                        _pop_and_process()
                elif batch is None:
                    self.self_check_during_idle()
                if self.is_generation:
                    self.launch_batch_sample_if_needed(batch_result)

            self.last_batch = batch

            # ── Tier 1 workload monitor (periodic re-planning check) ──
            self._tier1_monitor_check()

    def _tier1_monitor_check(self):
        """Periodically check WorkloadMonitor and trigger re-plan if needed."""
        if self._tier1_monitor is None or self._tier1_collector is None:
            return
        from sglang.srt.layers.afd import afd_is_attn

        if not afd_is_attn():
            return

        now = time.time()
        if now - self._tier1_last_monitor_time < self.server_args.tier1_monitor_window_s:
            return
        self._tier1_last_monitor_time = now

        # Build MonitoringWindow from collector and feed to monitor
        is_saving = (
            hasattr(self, "_af_dvfs_ctrl")
            and self._af_dvfs_ctrl is not None
            and self._af_dvfs_ctrl.is_energy_saving
        )
        window = self._tier1_collector.build_window(
            is_tier2_energy_saving=is_saving,
        )
        if window is None:
            return  # not enough data yet

        self._tier1_monitor.record_window(window)
        self._tier1_collector.reset_window()

        should_replan, reasons = self._tier1_monitor.should_replan()

        # ── Per-window detection summary ──
        logger.info(
            "[Tier1] window=%.0fs | SLO_violation=%.2f%% "
            "(TTFT_SLO=%.0fms TPOT_SLO=%.0fus) | "
            "a_util=%.2f f_util=%.2f p_util=%.2f d_util=%.2f | "
            "TTFT_p99=%.0fms TPOT_p99=%.0fus | active=%d | "
            "replan=%s%s",
            self._tier1_collector.window_elapsed_s,
            window.slo_violation_rate * 100,
            self.server_args.afd_ttft_slo_ms,
            self.server_args.afd_tpot_slo_us,
            window.a_util, window.f_util, window.p_util, window.d_util,
            window.ttft_p99_ms, window.tpot_p99_us,
            window.active_requests,
            "YES" if should_replan else "no",
            f" reasons={reasons}" if should_replan else "",
        )
        if window.load_distribution:
            logger.info("[Tier1] load_dist=%s", window.load_distribution)

        if should_replan:
            logger.info("[Tier1] Triggering re-plan: %s", reasons)
            self._replan_tier1(reasons)

    def _tier1_record_batch(self, batch, is_decode: bool, t_iter_us: float = 0.0):
        """Record per-batch metrics into the collector (no-op if collector is None)."""
        if self._tier1_collector is None:
            return
        # End-of-batch: extract per-request TTFT/TPOT from this batch
        self._tier1_collector.record_batch_end(batch, is_decode, t_iter_us)

    def _tier1_record_batch_start(self, is_prefill: bool):
        """Record batch start time (no-op if collector is None)."""
        if self._tier1_collector is None:
            return
        self._tier1_collector.record_batch_start(is_prefill)

    def _replan_tier1(self, reasons: list[str]):
        """Re-run the Tier 1 solver and switch configuration.

        Called when WorkloadMonitor detects a significant workload shift.
        Lazy-initialises the solver if it was deferred (pre-computed solution
        mode).  Currently logs the new solution; the drain-then-switch
        transition will be implemented in a follow-up.
        """
        try:
            self._ensure_tier1_solver()
            if self._tier1_solver is None:
                logger.warning("Tier 1 re-plan: solver init failed")
                return

            server_args = self.server_args
            wl, slo = self._make_tier1_workload_slo(server_args)

            new_solution = self._tier1_solver.solve(
                G=server_args.tier1_gpu_count,
                workload=wl,
                slo=slo,
            )

            if not new_solution.feasible:
                logger.warning(
                    "[Tier1] Re-plan: INFEASIBLE — keeping current config "
                    "(reasons: %s)", reasons)
                return

            old = self._tier1_solution
            logger.info(
                "[Tier1] Re-plan OLD: PA(tp=%d,f=%d) PF(tp=%d,f=%d) "
                "DA(tp=%d,f=%d) DF(tp=%d,f=%d) k_P=%d k_D=%d E/layer=%.2fmJ",
                old.tp_pa, old.f_pa, old.tp_pf, old.f_pf,
                old.tp_da, old.f_da, old.tp_df, old.f_df,
                old.k_p, old.k_d, old.total_energy_mj_per_layer,
            )
            logger.info(
                "[Tier1] Re-plan NEW: PA(tp=%d,f=%d) PF(tp=%d,f=%d) "
                "DA(tp=%d,f=%d) DF(tp=%d,f=%d) k_P=%d k_D=%d E/layer=%.2fmJ | "
                "reasons=%s",
                new_solution.tp_pa, new_solution.f_pa,
                new_solution.tp_pf, new_solution.f_pf,
                new_solution.tp_da, new_solution.f_da,
                new_solution.tp_df, new_solution.f_df,
                new_solution.k_p, new_solution.k_d,
                new_solution.total_energy_mj_per_layer,
                reasons,
            )
            self._tier1_solution = new_solution

            # Reset monitor reference distribution
            if self._tier1_monitor is not None:
                self._tier1_monitor.reset_reference({})

            # ── Transition: apply new configuration ──
            self._apply_tier1_transition(old, new_solution)

        except Exception as e:
            logger.error("Tier 1 re-plan failed: %s", e)

    def _apply_tier1_transition(self, old: "Tier1Solution", new: "Tier1Solution"):
        """Apply Tier 1 configuration transition based on what changed.

        Three scenarios per design doc Section 1.5:
          1. Frequency-only change: apply immediately (~6ms)
          2. TP change: requires drain-then-switch (stop new requests,
             wait for active to finish, restart with new TP)
          3. k_P/k_D change: P/D rebalance (shrink then expand)

        Currently implements scenario 1 (freq change) and updates Tier 2
        baseline. Scenarios 2/3 log a warning — full drain-then-switch
        requires orchestrator-level coordination beyond the scheduler.
        """
        freq_only = (
            old.tp_pa == new.tp_pa and old.tp_pf == new.tp_pf
            and old.tp_da == new.tp_da and old.tp_df == new.tp_df
            and old.k_p == new.k_p and old.k_d == new.k_d
        )

        if freq_only:
            logger.info("[Tier1] Transition: frequency-only change, applying immediately")
            self._apply_freq(new.f_da, new.f_df)
            if hasattr(self, "_af_dvfs_ctrl") and self._af_dvfs_ctrl is not None:
                self._af_dvfs_ctrl.update_baseline(new.f_da, new.f_df)
        else:
            logger.warning(
                "[Tier1] Transition: TP or k change detected "
                "(tp_pa %d→%d, tp_da %d→%d, k_p %d→%d, k_d %d→%d). "
                "Drain-then-switch required — upgrading to max freq as "
                "interim measure. Full transition requires orchestrator.",
                old.tp_pa, new.tp_pa, old.tp_da, new.tp_da,
                old.k_p, new.k_p, old.k_d, new.k_d,
            )
            self._apply_freq(1410, 1410)
            if hasattr(self, "_af_dvfs_ctrl") and self._af_dvfs_ctrl is not None:
                self._af_dvfs_ctrl.update_baseline(1410, 1410)

    def _afd_get_next_batch(self, disagg_mode):
        """Select the correct batch scheduling function based on disagg mode.

        Both Attn and FFN sides must use the same disagg-specific batch
        scheduling to keep forward_mode perfectly in sync:
        - DECODE: get_next_disagg_decode_batch_to_run (prebuilt → decode).
        - PREFILL: get_next_disagg_prefill_batch_to_run.
        - NULL: standard get_next_batch_to_run.
        """
        from sglang.srt.disaggregation.utils import DisaggregationMode

        if disagg_mode == DisaggregationMode.PREFILL:
            return self.get_next_disagg_prefill_batch_to_run()
        elif disagg_mode == DisaggregationMode.DECODE:
            return self.get_next_disagg_decode_batch_to_run()
        else:
            return self.get_next_batch_to_run()

    def _init_afd_dvfs(self, server_args):
        """Initialize Tier 2 DVFS components (predictor + controller + HW)."""
        try:
            from sglang.srt.energy.af_profile_predictor import AFProfilePredictor
            from sglang.srt.energy.af_dvfs_controller import AFDVFSController

            predictor = AFProfilePredictor(server_args.afd_energy_model_dir)
            tp_a = getattr(server_args, "afd_attn_tp", None) or server_args.tp_size
            tp_f = getattr(server_args, "afd_ffn_tp", None) or server_args.tp_size
            self._af_dvfs_ctrl = AFDVFSController(
                predictor=predictor,
                num_layers=self.model_config.num_hidden_layers,
                tp_a=tp_a,
                tp_f=tp_f,
            )
            logger.info("AFD DVFS controller initialized")
        except Exception as e:
            logger.warning("Failed to init AFD DVFS controller: %s", e)
            self._af_dvfs_ctrl = None
            return

        try:
            import torch
            from sglang.srt.layers.dvfs import DVFSController

            # Use physical NVML GPU index set by launcher (not CUDA index,
            # which may differ when CUDA_VISIBLE_DEVICES remaps devices).
            nvml_device_index = int(os.environ.get("AFD_NVML_DEVICE_INDEX",
                torch.cuda.current_device() if torch.cuda.is_available() else 0))

            self._dvfs_hw = DVFSController(device_index=nvml_device_index)
            logger.info("DVFS HW controller initialized on NVML GPU %d", nvml_device_index)
        except Exception as e:
            logger.warning("DVFS HW unavailable (libdvfs_ctrl.so missing?): %s. "
                           "Frequency decisions will be logged but not applied.", e)
            self._dvfs_hw = None

    def _init_afd_tier1(self, server_args):
        """Initialize Tier 1 dynamic monitoring on the PA scheduler.

        Always starts WorkloadMonitor + WorkloadMetricsCollector for real-time
        per-request TTFT/TPOT tracking and GPU utilisation polling.

        Solver initialisation depends on the startup mode:
        - If ``--tier1-initial-solution`` is set (written by af_launcher.py
          --start-with-workload), the solution is loaded from JSON and the
          solver is NOT initialised.  It is lazy-created only when a re-plan
          is triggered.
        - Otherwise the solver is initialised immediately and run once at
          startup.
        """
        from sglang.srt.layers.afd_type import AFDPerspective

        if server_args.afd_perspective != AFDPerspective.AFD_PERSPECTIVE_ATTN:
            return

        try:
            # ── Monitor + collector (always) ───────────────────────
            from sglang.srt.energy.workload_monitor import WorkloadMonitor
            from sglang.srt.energy.workload_collector import WorkloadMetricsCollector

            # Read GPU→pool mapping from launcher-set env vars
            attn_gpu_indices = _parse_gpu_indices_env("AFD_ATTN_GPU_INDICES")
            ffn_gpu_indices = _parse_gpu_indices_env("AFD_FFN_GPU_INDICES")

            self._tier1_monitor = WorkloadMonitor(
                window_s=server_args.tier1_monitor_window_s,
            )
            self._tier1_collector = WorkloadMetricsCollector(
                ttft_slo_ms=server_args.afd_ttft_slo_ms,
                tpot_slo_us=server_args.afd_tpot_slo_us,
                window_s=server_args.tier1_monitor_window_s,
                attn_gpu_indices=attn_gpu_indices,
                ffn_gpu_indices=ffn_gpu_indices,
                enable_nvml_polling=True,
                stats_path=server_args.tier1_stats_path,
            )
            self._tier1_last_monitor_time = time.time()
            logger.info(
                "Tier 1 monitor+collector started (window=%.1fs, "
                "attn_gpus=%s, ffn_gpus=%s)",
                server_args.tier1_monitor_window_s,
                attn_gpu_indices, ffn_gpu_indices,
            )

            # ── Solution ──────────────────────────────────────────
            init_sol_path = server_args.tier1_initial_solution
            if init_sol_path:
                # Pre-computed by af_launcher.py --start-with-workload
                self._load_tier1_solution_from_json(init_sol_path)
                self._tier1_solver = None  # lazy-init on re-plan
                sol = self._tier1_solution
                logger.info(
                    "[Tier1] Loaded pre-computed solution from %s: "
                    "PA(tp=%d,f=%d) PF(tp=%d,f=%d) "
                    "DA(tp=%d,f=%d) DF(tp=%d,f=%d) "
                    "k_P=%d k_D=%d E/layer=%.2fmJ GPU=%d",
                    init_sol_path,
                    sol.tp_pa, sol.f_pa, sol.tp_pf, sol.f_pf,
                    sol.tp_da, sol.f_da, sol.tp_df, sol.f_df,
                    sol.k_p, sol.k_d, sol.total_energy_mj_per_layer,
                    sol.gpu_used,
                )
            else:
                # No pre-computed solution — run solver now
                self._tier1_solver = self._make_tier1_solver(server_args)
                wl, slo = self._make_tier1_workload_slo(server_args)
                self._tier1_solution = self._tier1_solver.solve(
                    G=server_args.tier1_gpu_count,
                    workload=wl,
                    slo=slo,
                )
                if self._tier1_solution.feasible:
                    sol = self._tier1_solution
                    logger.info(
                        "[Tier1] Init solution: "
                        "PA(tp=%d,f=%d) PF(tp=%d,f=%d) "
                        "DA(tp=%d,f=%d) DF(tp=%d,f=%d) "
                        "k_P=%d k_D=%d E/layer=%.2fmJ GPU=%d",
                        sol.tp_pa, sol.f_pa, sol.tp_pf, sol.f_pf,
                        sol.tp_da, sol.f_da, sol.tp_df, sol.f_df,
                        sol.k_p, sol.k_d, sol.total_energy_mj_per_layer,
                        sol.gpu_used,
                    )
                else:
                    logger.warning(
                        "[Tier1] Init: INFEASIBLE — using warm_start fallback"
                    )
                    self._tier1_solution = self._tier1_solver.warm_start(
                        G=server_args.tier1_gpu_count,
                        workload=wl,
                        slo=slo,
                    )

        except Exception as e:
            logger.warning("Failed to init Tier 1: %s", e)
            self._tier1_solver = None
            self._tier1_solution = None
            self._tier1_monitor = None
            self._tier1_collector = None

    # ── Lazy solver init helpers ─────────────────────────────────────

    def _load_tier1_solution_from_json(self, path: str):
        """Create a Tier1Solution from a JSON file written by af_launcher.py."""
        import json
        from sglang.srt.energy.tier1_solver import Tier1Solution

        with open(path, "r") as f:
            data = json.load(f)
        self._tier1_solution = Tier1Solution(
            k_p=data.get("k_p", 1),
            k_d=data.get("k_d", 1),
            tp_pa=data.get("tp_pa", 1),
            tp_pf=data.get("tp_pf", 1),
            tp_da=data.get("tp_da", 1),
            tp_df=data.get("tp_df", 1),
            f_pa=data.get("f_pa", 1410),
            f_pf=data.get("f_pf", 1410),
            f_da=data.get("f_da", 1410),
            f_df=data.get("f_df", 1410),
            total_energy_mj_per_layer=data.get("total_energy_mj_per_layer", 0.0),
            gpu_used=data.get("gpu_used", 0),
            feasible=data.get("feasible", True),
        )

    @staticmethod
    def _make_tier1_workload_slo(server_args):
        from sglang.srt.energy.tier1_solver import SLOConfig, WorkloadProfile

        wl = WorkloadProfile(
            lambda_prefill=server_args.tier1_lambda_prefill,
            n_active_decode=server_args.tier1_n_active_decode,
            il_rep_p=server_args.tier1_il_rep_p,
            bs_avg_p=server_args.tier1_bs_avg_p,
            il_rep_d=server_args.tier1_il_rep_d,
            ol_rep_d=server_args.tier1_ol_rep_d,
            bs_avg_d=server_args.tier1_bs_avg_d,
        )
        slo = SLOConfig(
            ttft_ms=server_args.afd_ttft_slo_ms,
            tpot_ms=server_args.afd_tpot_slo_us / 1000.0,
        )
        return wl, slo

    def _ensure_tier1_solver(self):
        """Lazy-init the Tier1Solver if it was deferred (pre-computed solution mode)."""
        if self._tier1_solver is not None:
            return
        server_args = self.server_args
        logger.info("Tier 1: lazy-initialising solver for re-plan …")
        self._tier1_solver = self._make_tier1_solver(server_args)

    def _make_tier1_solver(self, server_args):
        """Create a fully initialised Tier1Solver with ProfileTable."""
        from sglang.srt.energy.profile_table import ProfileTable
        from sglang.srt.energy.tier1_solver import Tier1Solver

        pt = ProfileTable(
            prefill_path=server_args.tier1_prefill_data_path,
            decode_path=server_args.tier1_decode_data_path,
            energy_model_dir=server_args.afd_energy_model_dir,
        )
        return Tier1Solver(
            profile_table=pt,
            num_layers=self.model_config.num_hidden_layers,
            num_kv_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            hidden_size=self.model_config.hidden_size,
            gpu_mem_gb=80.0,
        )

    def _compute_prefill_slack(self, batch) -> float:
        """Compute tightest TTFT slack (us) across all requests in batch."""
        slo_us = self.server_args.afd_ttft_slo_ms * 1000
        min_slack = slo_us
        now = time.perf_counter()
        for req in batch.reqs:
            ts = getattr(req, "time_stats", None)
            dispatch_t = getattr(ts, "api_server_dispatch_time", 0.0) if ts else 0.0
            if dispatch_t > 0:
                elapsed = (now - dispatch_t) * 1e6
                min_slack = min(min_slack, slo_us - elapsed)
        return max(min_slack, 0)

    def _apply_freq(self, f_a: int, f_f: int):
        """Apply frequency via NVML. Each process only controls its own GPU."""
        from sglang.srt.layers.afd import afd_is_attn
        target_f = f_a if afd_is_attn() else f_f
        if self._dvfs_hw is not None:
            self._dvfs_hw.lock_sm_clock(target_f)
        self._cur_f_a = f_a
        self._cur_f_f = f_f

    def _afd_dvfs_before_batch(self, batch):
        """Select and apply frequency before running a batch."""
        if self._af_dvfs_ctrl is None:
            return
        from sglang.srt.layers.afd import get_afd_micro_batch
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        M = get_afd_micro_batch()

        if batch.forward_mode == ForwardMode.EXTEND:
            slack_us = self._compute_prefill_slack(batch)
            max_il = max(
                (r.extend_input_len for r in batch.reqs), default=1024
            )
            decision = self._af_dvfs_ctrl.select_freq_prefill(
                bs=batch.batch_size(), il=max_il,
                slack_us=slack_us, M=M,
            )
            if decision.f_a != self._cur_f_a or decision.f_f != self._cur_f_f:
                self._apply_freq(decision.f_a, decision.f_f)

        elif batch.forward_mode.is_decode():
            self._af_dvfs_ctrl.tick_decode_iteration()
            t_iter_us = 0.0
            if hasattr(self, "_last_decode_batch_time"):
                t_iter_us = (time.perf_counter() - self._last_decode_batch_time) * 1e6
                self._af_dvfs_ctrl.compute_window_size(t_iter_us)
            self._last_decode_batch_time = time.perf_counter()

            # Share decode iteration time with PA's Tier1 monitor via stats file
            self._write_afd_decode_stats(t_iter_us, batch.batch_size())

            reeval_reason = self._af_dvfs_ctrl.should_reevaluate_decode(
                batch.batch_size(),
                current_tpot_us=t_iter_us,
                slo_tpot_us=self.server_args.afd_tpot_slo_us,
            )
            if reeval_reason:
                repr_il = int(sum(len(r.origin_input_ids) for r in batch.reqs) / max(len(batch.reqs), 1))
                repr_ol = int(sum(
                    (r.seqlen - len(r.origin_input_ids)) for r in batch.reqs
                ) / max(len(batch.reqs), 1))
                repr_ol = max(repr_ol, 1)
                decision = self._af_dvfs_ctrl.select_freq_decode(
                    bs=batch.batch_size(), il=repr_il, ol=repr_ol,
                    slo_tpot_us=self.server_args.afd_tpot_slo_us, M=M,
                    reeval_reason=reeval_reason,
                )
                if decision.switched:
                    self._apply_freq(decision.f_a, decision.f_f)

    def _write_afd_decode_stats(self, t_iter_us: float, bs: int):
        """Write decode iteration timing to shared stats file for PA's Tier1 monitor."""
        import json
        stats_path = getattr(self.server_args, "tier1_stats_path", None)
        if not stats_path:
            return
        try:
            with open(stats_path, "w") as f:
                json.dump({
                    "t_iter_us": t_iter_us,
                    "bs": bs,
                    "timestamp": time.time(),
                }, f)
        except Exception as e:
            logger.error("Failed to write decode stats to %s: %s", stats_path, e)

    def _afd_process_input_requests(self, recv_reqs):
        """Process input requests with AFD awareness (S1, S4).

        AFDReqInput messages are queued and consumed one per iteration
        to avoid last-one-wins overwrites when multiple arrive in a
        single drain cycle (fixes L6).
        """
        from sglang.srt.layers.afd import afd_is_attn
        from sglang.srt.managers.io_struct import AFDReqInput
        from sglang.srt.managers.io_struct import (
            TokenizedEmbeddingReqInput,
            TokenizedGenerateReqInput,
        )

        filtered_reqs = []
        for recv_req in recv_reqs:
            if isinstance(recv_req, AFDReqInput):
                if not hasattr(self, "_afd_pending_batch_infos"):
                    from collections import deque
                    self._afd_pending_batch_infos = deque()
                self._afd_pending_batch_infos.append(recv_req)
                continue

            if afd_is_attn() and self.afd_send_to_ffn is not None:
                if isinstance(
                    recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                ):
                    self.afd_send_to_ffn.send_pyobj(recv_req)

            # FFN side: skip forwarded TokenizedGenerateReqInput — Reqs are
            # created exclusively from AFDReqInput via _afd_ensure_reqs_from_afdreq
            # so the FFN batch *always* matches the Attn's current batch.
            if not afd_is_attn() and isinstance(
                recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
            ):
                continue

            filtered_reqs.append(recv_req)

        pending = getattr(self, "_afd_pending_batch_infos", None)
        if pending is None:
            from collections import deque
            self._afd_pending_batch_infos = deque()
            pending = self._afd_pending_batch_infos

        # Track whether we just consumed an AFDReqInput so we can create Reqs
        # AFTER process_input_requests (to avoid duplicates with TokenizedGenerateReqInput).
        consumed_afd_req = None

        if self._afd_batchsize_attn is None and pending:
            afd_req = self._afd_pending_batch_infos.popleft()
            consumed_afd_req = afd_req
            old_req_ids = self._afd_req_ids
            self._afd_batchsize_attn = afd_req.batch_size
            self._afd_forward_mode = afd_req.forward_mode
            self._afd_req_ids = afd_req.req_ids

            if afd_req.output_ids_per_req and afd_req.req_ids:
                self._afd_sync_output_ids(afd_req)

            # FFN decode side: clean up requests no longer tracked by Attn
            if not afd_is_attn() and old_req_ids is not None:
                self._afd_ffn_cleanup_stale(
                    set(afd_req.req_ids), set(old_req_ids)
                )

        # process_input_requests handles TokenizedGenerateReqInput → creates Req objects
        self.process_input_requests(filtered_reqs)

        # Now create Reqs from AFDReqInput for any rids that still don't exist
        if (
            consumed_afd_req is not None
            and not afd_is_attn()
            and consumed_afd_req.input_ids_per_req
        ):
            self._afd_ensure_reqs_from_afdreq(consumed_afd_req)

    def _afd_ensure_reqs_from_afdreq(self, afd_req):
        """FFN decode: create Req objects from AFDReqInput data when missing.

        In PD+AF decode, DA sends AFDReqInput with metadata but the FFN side
        may not have corresponding Req objects (since they arrive via KV-cache
        transfer, not TokenizedGenerateReqInput). This creates minimal Reqs so
        get_next_disagg_decode_batch_to_run can build a batch.
        """
        from sglang.srt.layers.afd import afd_is_attn
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        if afd_is_attn():
            return

        existing_rids = set(req.rid for req in self.waiting_queue)
        if self.running_batch is not None:
            existing_rids.update(req.rid for req in self.running_batch.reqs)
        # AFD event_loop never populates running_batch on FFN side — reqs
        # live in last_batch instead.  Without this check every AFDReqInput
        # creates a duplicate Req, tripling the batch.
        if self.last_batch is not None:
            existing_rids.update(req.rid for req in self.last_batch.reqs)

        for i, rid in enumerate(afd_req.req_ids):
            if rid in existing_rids:
                continue

            origin_input_ids = (
                afd_req.input_ids_per_req[i]
                if afd_req.input_ids_per_req and i < len(afd_req.input_ids_per_req)
                else []
            )
            max_new_tokens = (
                afd_req.max_new_tokens_per_req[i]
                if afd_req.max_new_tokens_per_req and i < len(afd_req.max_new_tokens_per_req)
                else 128
            )

            req = Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=origin_input_ids,
                sampling_params=SamplingParams(
                    max_new_tokens=max_new_tokens,
                    stop=[],
                    stop_regex=[],
                ),
                vocab_size=self.model_config.vocab_size,
            )
            # Pre-populate output_ids so fill_ids is correct for the forward pass
            if afd_req.output_ids_per_req and i < len(afd_req.output_ids_per_req):
                req.output_ids = list(afd_req.output_ids_per_req[i])
            req.fill_ids = list(origin_input_ids) + list(req.output_ids)
            req.set_extend_input_len(len(req.fill_ids))
            self.waiting_queue.append(req)

    def _afd_sync_output_ids(self, afd_req):
        """Sync output_ids from Attn→FFN so fill_ids match for AF communication.

        In PD+AF decode, Attn side has output_ids (from prefill's first token)
        but FFN side doesn't. Without syncing, FFN's fill_ids is shorter by 1,
        causing tensor shape mismatch in AF communication.
        """
        rid_to_oids = dict(zip(afd_req.req_ids, afd_req.output_ids_per_req))

        # Sync to waiting_queue requests
        for req in self.waiting_queue:
            oids = rid_to_oids.get(req.rid)
            if oids is not None and len(req.output_ids) < len(oids):
                req.output_ids = list(oids)
                req.fill_ids = req.origin_input_ids + req.output_ids
                req.set_extend_input_len(len(req.fill_ids))

        # Sync to running_batch requests
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                oids = rid_to_oids.get(req.rid)
                if oids is not None and len(req.output_ids) < len(oids):
                    req.output_ids = list(oids)
                    req.fill_ids = req.origin_input_ids + req.output_ids

    def _afd_ffn_cleanup_stale(self, active_req_ids: set, old_req_ids: set):
        """Release KV cache for FFN-side requests that Attn has stopped tracking.

        On the decode-FFN side, the Attn side drives the request lifecycle.
        When a request ID disappears from AFDReqInput, it means Attn has
        finished the request. The FFN side must release its KV cache pages
        to avoid a false memory-leak detection during idle checks.
        """
        from sglang.srt.layers.afd import afd_is_attn as _is_attn
        from sglang.srt.mem_cache.common import release_kv_cache as _release_kv

        if _is_attn():
            return

        # Req IDs that left the Attn-managed set since last batch info
        finished_ids = old_req_ids - active_req_ids
        if not finished_ids:
            return

        # Clean waiting_queue
        stale_waiting = [r for r in self.waiting_queue if r.rid in finished_ids]
        if stale_waiting:
            for req in stale_waiting:
                _release_kv(req, self.tree_cache)
            self.waiting_queue = [r for r in self.waiting_queue if r.rid not in finished_ids]

        # Clean running_batch — force-finish so filter_batch removes them
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.rid in finished_ids and not req.finished():
                    from sglang.srt.managers.schedule_batch import FINISH_LENGTH
                    req.finished_reason = FINISH_LENGTH(length=0)
                    _release_kv(req, self.tree_cache)

    def _afd_ffn_cleanup_all(self):
        """Force-clean ALL remaining requests on the FFN decode side at idle.

        Called when no more AFDReqInput will arrive (DA side is idle too).
        Without this, stale requests in running_batch hold KV cache pages that
        trigger a false memory-leak detection.
        """
        from sglang.srt.layers.afd import afd_is_attn as _is_attn
        from sglang.srt.mem_cache.common import release_kv_cache as _release_kv

        if _is_attn():
            return

        # Clean waiting_queue
        for req in self.waiting_queue:
            if req.req_pool_idx is not None:
                _release_kv(req, self.tree_cache)
        self.waiting_queue.clear()

        # Clean running_batch
        if self.running_batch is not None:
            from sglang.srt.managers.schedule_batch import FINISH_LENGTH
            for req in self.running_batch.reqs:
                if not req.finished():
                    req.finished_reason = FINISH_LENGTH(length=0)
                if req.req_pool_idx is not None:
                    _release_kv(req, self.tree_cache)
            # Clear the batch to prevent double-free on the next call
            # (the wait loop calls _afd_ffn_cleanup_all every 10ms, and
            #  release_kv_cache sets req_pool_idx=None on the first call).
            self.running_batch.reqs.clear()
            self.running_batch.batch_is_full = False

        # Clean last_batch — may hold reqs from the final EXTEND batch that
        # were never merged into running_batch (e.g. DECODE-mode last_batch).
        if self.last_batch is not None:
            for req in self.last_batch.reqs:
                if req.finished():
                    continue
                from sglang.srt.managers.schedule_batch import FINISH_LENGTH
                req.finished_reason = FINISH_LENGTH(length=0)
                if req.req_pool_idx is not None:
                    _release_kv(req, self.tree_cache)
            self.last_batch.reqs.clear()
            self.last_batch = None

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            # If we do not need to overlap the current batch with the last batch,
            # we can process the last batch immediately.
            if disable_overlap_for_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None
                self.cancel_bubble_timer()

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    def is_disable_overlap_for_batch(self, batch: ScheduleBatch) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        # In DP attention mode, use the globally synchronized is_extend_in_batch
        # so all DP ranks make the same overlap decision (avoiding deadlock).
        # In non-DP mode, use the local forward_mode directly.
        if self.require_mlp_sync:
            is_extend = lambda b: b and b.is_extend_in_batch
        else:
            is_extend = lambda b: b and b.forward_mode.is_extend()

        batch_is_extend = is_extend(batch)
        last_batch_is_extend = is_extend(self.last_batch)

        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # We do not support overlap + spec + grammar yet,
        # so we need to turn off overlap for this batch.
        # TODO(lsyin): support overlap + spec + grammar
        need_grammar_sync = (
            batch
            and batch.is_spec_v2
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.recv_skipper is not None:
            last_forward_mode = (
                self.last_batch.forward_mode if self.last_batch is not None else None
            )
            if not self.recv_skipper.handle(last_forward_mode):
                return []

        if self.pp_rank == 0:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                        recv_req = unwrap_shm_features(recv_req)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                dp_offset = self.attn_dp_rank * self.attn_tp_size
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.pp_rank * self.tp_size + dp_offset,
                    self.world_group.cpu_group,
                    (self.pp_rank - 1) * self.tp_size + dp_offset,
                    self.pp_rank * self.tp_size + dp_offset,
                )
            else:
                recv_reqs = None

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        if self.server_args.enable_dp_attention:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            if self.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            if self.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            if self.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

        # Process MM requests under EPD-disaggregation mode
        if (
            self.pp_rank == 0
            and self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:

                status_code = (
                    HTTPStatus.BAD_REQUEST
                    if error_code == 400
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)

        return recv_reqs

    def _split_work_and_control_reqs(self, recv_reqs: List):
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs

    def process_input_requests(self, recv_reqs: List):
        now = time.monotonic()
        self.session_controller.maybe_reap(now)
        for recv_req in recv_reqs:
            # Skip health check when server is busy — ongoing requests already carry health info.
            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
                self.return_health_check_ipcs.append(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if not isinstance(output, RpcReqOutput):
                    self.send_to_tokenizer.send_output(output, recv_req)
                else:
                    if self.recv_from_rpc is not None:
                        self.recv_from_rpc.send_pyobj(output)

    def init_req_max_new_tokens(self, req):
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs: Optional[dict],
    ):
        """Materialize MultimodalInputs once on the entry rank and broadcast to others.

        Entry rank:
        - constructs MultimodalInputs.from_dict(raw_mm_inputs) once
        - broadcasts to other ranks in self.cpu_group (if world_size > 1)

        Non-entry ranks:
        - receive the object via broadcast (if world_size > 1)
        - otherwise (single-rank / no group) fall back to local from_dict

        Returns:
            MultimodalInputs | None
        """
        if raw_mm_inputs is None:
            return None

        group_world_size = 1
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.dp_tp_cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.dp_tp_cpu_group
                )
        except Exception as e:
            logger.warning(
                f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
            )

        # In case tp size > 1, all the Scheduler TP ranks runs the duplicated computing
        # process in CPU which occupies the main thread CPU cycle. This computing logic
        # merely needs to be run on TP0 and be broadcast to other TP ranks.
        # Since the Scheduler is single-threaded, any large CPU cost will impact
        # handling of other messages. For example, CPU hits 99.9% can significantly
        # increase the CUDA kernel launch time.
        if self.dp_tp_group.rank_in_group == 0:
            # Only the entry rank materializes once from dict.
            image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
            # Broadcast to other TP ranks (use src=0 within the group).
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
            else:
                image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)

        return image_inputs

    def _get_multimodal_inputs(self, mm_inputs_dict: dict):
        if self.server_args.enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs_dict)
        else:
            return MultimodalInputs.from_dict(mm_inputs_dict)

    def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue
            # For session requests, keep mm_inputs for the next request
            if req.session:
                continue
            # For non-session requests, clear features and mm_inputs
            for item in mm_inputs.mm_items:
                item.feature = None
            req.multimodal_inputs = None

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Route: normal request / session request / session-not-found
        session_id = (
            recv_req.session_params.id if recv_req.session_params is not None else None
        )

        if session_id is None:
            # Normal non-session request
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [1] * seq_length
                recv_req.input_ids = fake_input_ids

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                input_embeds=recv_req.input_embeds,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                disagg_mode=self.disaggregation_mode,
                routed_dp_rank=recv_req.routed_dp_rank,
                disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
                routing_key=recv_req.routing_key,
                http_worker_ipc=recv_req.http_worker_ipc,
                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
            )
            req.tokenizer = self.tokenizer

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if (
                    recv_req.bootstrap_room is None
                    and self.transfer_backend != TransferBackend.FAKE
                ):
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    recv_req.time_stats.trace_ctx.abort(
                        abort_info={"reason": error_msg}
                    )
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.stream_output([req], req.return_logprob)
                    return

        elif session_id in self.session_controller:
            # Session exists: create request from session
            session = self.session_controller.get(session_id)
            req = session.create_req(
                recv_req,
                self.tokenizer,
                self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
            )
            # TODO: set trace context
            if self.enable_metrics:
                req.time_stats.set_metrics_collector(self.metrics_collector)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        else:
            # Session ID provided but session not found
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                vocab_size=self.model_config.vocab_size,
            )
            req.tokenizer = self.tokenizer
            req.set_finish_with_abort(
                f"Invalid request: session id {session_id} does not exist"
            )
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)

            SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # initialize before returning
        self.init_req_max_new_tokens(req)

        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
            # When return_logprob is False, logprob_start_len should be ignored
            recv_req.logprob_start_len = -1

        if recv_req.logprob_start_len == -1:
            if recv_req.return_logprob and recv_req.token_ids_logprob is None:
                # If logprob is required but neither token_ids_logprob nor logprob_start_len is
                # set, return the logprobs for output tokens by default
                req.logprob_start_len = len(req.origin_input_ids)
            elif req.is_prefill_only:
                # For prefill-only requests with logprob_start_len == -1, set logprob_start_len
                # beyond input sequence to skip input logprob computation entirely
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # If return_logprob is False, only the last token requires logprob computation
                req.logprob_start_len = -1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len > len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self._add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _prefetch_kvcache(self, req: Req):
        if self.enable_hicache_storage:
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            last_host_node = (
                req.last_host_backup_node
                if req.last_host_backup_node is not None
                else req.last_host_node
            )
            if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
                last_hash = last_host_node.get_last_hash_value()
                matched_len = len(req.prefix_indices) + req.host_hit_length
                new_input_tokens = req.fill_ids[matched_len:]

                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                self.tree_cache.prefetch_from_storage(
                    req.rid,
                    last_host_node,
                    new_input_tokens,
                    last_hash,
                    prefix_keys,
                )

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        if self.disaggregation_mode == DisaggregationMode.NULL:
            if not self._set_or_validate_priority(req):
                return
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            req.time_stats.set_wait_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # In AFD mode, FFN side doesn't do KV transfer — skip bootstrap
            # queue and go straight to waiting_queue.
            from sglang.srt.layers.afd import afd_is_ffn
            if afd_is_ffn():
                self._prefetch_kvcache(req)
                req.sampling_params.max_new_tokens = 1
                self.waiting_queue.append(req)
                req.time_stats.set_wait_queue_entry_time()
            else:
                self._prefetch_kvcache(req)
                self.disagg_prefill_bootstrap_queue.add(
                    req, self.model_config.num_key_value_heads
                )
                req.time_stats.set_prefill_bootstrap_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # In AFD mode, FFN side runs event_loop_afd (non-disagg) and needs
            # requests in waiting_queue directly.
            from sglang.srt.layers.afd import afd_is_ffn as _afd_is_ffn_decode
            if _afd_is_ffn_decode():
                self._prefetch_kvcache(req)
                # Set extend_input_len so get_new_batch_prefill builds a
                # valid EXTEND batch (without this it stays 0 → empty logits
                # → CUDA assert in sampling).
                req.set_extend_input_len(len(req.fill_ids))
                self.waiting_queue.append(req)
                req.time_stats.set_wait_queue_entry_time()
            else:
                self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
                if not is_retracted:
                    req.time_stats.set_decode_prealloc_queue_entry_time()
                else:
                    req.time_stats.set_retract_time()
        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")

    def _set_or_validate_priority(self, req: Req) -> bool:
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if self.enable_priority_scheduling and req.priority is None:
            if self.schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif (
            not self.enable_priority_scheduling
            and req.priority is not None
            and self.abort_on_priority_when_disabled
        ):
            abort_req = AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
                rid=req.rid,
            )
            req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
            self.send_to_tokenizer.send_output(abort_req, req)
            return False
        return True

    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted."""
        if (
            self.max_queued_requests is None
            or len(self.waiting_queue) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            direction = 1 if self.schedule_low_priority_values_first else -1
            key_fn = lambda item: (
                direction * item[1].priority,
                item[1].time_stats.wait_queue_entry_time,
            )
            idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(candidate_req.rid)
                elif self.enable_hierarchical_cache:
                    self.tree_cache.terminate_prefetch(candidate_req.rid)
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.send_to_tokenizer.send_output(
            AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
                rid=req_to_abort.rid,
            ),
            req_to_abort,
        )
        req_to_abort.time_stats.trace_ctx.abort(abort_info={"reason": message})
        return req_to_abort.rid == recv_req.rid

    def _abort_on_waiting_timeout(self):
        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) <= 0:
            return

        deleted_reqs = set()
        deadline = time.perf_counter() - timeout_s
        for req in self.waiting_queue:
            entry_time = req.time_stats.wait_queue_entry_time
            if 0 < entry_time < deadline:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(req.rid)
                self.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason={
                            "type": "abort",
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                            "message": "Request waiting timeout reached.",
                        },
                        rid=req.rid,
                    ),
                    req,
                )
                deleted_reqs.add(req)

        if deleted_reqs:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in deleted_reqs
            ]

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            token_type_ids=recv_req.token_type_ids,
            routed_dp_rank=recv_req.routed_dp_rank,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            lora_id=recv_req.lora_id,
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
        )
        req.tokenizer = self.tokenizer

        # Handle multimodal inputs
        if recv_req.image_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if self.pad_input_ids_func:
                req.origin_input_ids = self.pad_input_ids_func(
                    req.origin_input_ids, image_inputs
                )

            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self._add_request_to_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        req.logprob_start_len = -1
        self._add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    def stash_chunked_request(self, req: Req):
        self.tree_cache.cache_unfinished_req(req, chunked=True)

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        self._abort_on_waiting_timeout()
        self._abort_on_running_timeout()
        if self.dllm_config is not None:
            self.dllm_manager.filter_finished_reqs()

        # Merge the prefill batch into the running batch
        chunked_req_to_exclude = set()

        if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
            chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
            for req in self.dllm_manager.staging_queue:
                self.stash_chunked_request(req)

        if self.chunked_req is not None:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)
            self.stash_chunked_request(self.chunked_req)

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            if self.dllm_config is not None and self.last_batch.reqs:
                chunked_req_to_exclude.update(self.last_batch.reqs)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            if not self.last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        # For prefill-only batch, filter out finished requests since they
        # won't go through the decode step. This keeps running_batch accurate
        # for load reporting (num_running_reqs via /get_load).
        # Runs outside the last_batch block so stale requests are cleaned
        # even when no new batches arrive (e.g. traffic stops).
        if self.running_batch.is_prefill_only:
            self.running_batch.filter_batch()

        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm()
        else:
            new_batch = self.get_new_batch_prefill()

        need_mlp_sync = self.require_mlp_sync
        if need_mlp_sync and not self.spec_algorithm.is_none():
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before merging the new batch into running batch:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.maybe_prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode (skip for prefill-only batches)
            if (
                not self.running_batch.is_empty()
                and not self.running_batch.is_prefill_only
            ):
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None

        # Handle DP attention and log stats
        ret = self.maybe_prepare_mlp_sync_batch(ret, need_sync=need_mlp_sync)

        # Handle ngram embedding
        ret = self._maybe_prepare_ngram_embedding(ret)

        if ret:
            set_schedule_time_batch(ret)

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        res = get_global_server_args().pp_max_micro_batch_size - running_bs
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            _, token_usage, _, _ = self._get_token_info()
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=token_usage
            )

        ret = self._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=prefill_delayer_single_pass
        )

        if self.prefill_delayer:
            prefill_delayer_single_pass.finalize(actual_prefill=ret is not None)

        return ret

    def _get_new_batch_prefill_raw(
        self, prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor]
    ) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        if self.enable_priority_preemption:
            # Reset batch_is_full to try preemption with a prefill adder.
            self.running_batch.batch_is_full = False

        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)

        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.chunked_req is not None
            and not self.enable_priority_preemption
        ):
            self.running_batch.batch_is_full = True
            return None

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue, self.running_batch)

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the running batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            return None

        # Determine chunked_prefill_size for this batch
        chunked_prefill_size = self.chunked_prefill_size
        if self.chunked_req is not None and self.enable_dynamic_chunking:
            history_len = len(self.chunked_req.prefix_indices)
            dynamic_size = self.predict_next_chunk_size(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=self.max_prefill_bs,
            max_running_requests=self.max_running_requests,
            prefill_max_requests=self.server_args.prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            running_loras = {req.lora_id for req in self.running_batch.reqs}

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if self.enable_lora and req.lora_id not in running_loras:
                if self.enable_lora_overlap_loading:
                    # For overlapping loading of LoRA weights with computation, we will load each adapter one at a time,
                    # as opposed to loading them in one batch
                    res = self.lora_overlap_loader.try_overlap_load_lora(
                        req.lora_id, running_loras
                    )
                    if not res:
                        continue
                else:
                    new_lora_set = {req.lora_id} | running_loras
                    if not self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                        new_lora_set
                    ):
                        continue

            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue
                # Pop the number of tokens loaded from storage (L3 hits)
                req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(
                    req.rid
                )

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if self.enable_lora:
                running_loras.add(req.lora_id)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req is not None:
            self.chunked_req.is_chunked += 1

        # Record for logging prefill stats after forward
        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

        set_time_batch(can_run_list, "set_forward_entry_time")

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )
        self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        # Record prefill stats for logging after forward
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, self.running_batch.reqs, self.enable_priority_scheduling
        )

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
            # mix_with_running cats input_ids but not input_embeds — shapes would mismatch
            and new_batch.input_embeds is None
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch(v1_spec_info_filtered=True)
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch(v1_spec_info_filtered=True)
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Eagerly release lock_ref on completed write-through nodes so they
        # become evictable, improving batch scheduling headroom.
        if self.enable_hierarchical_cache:
            self.tree_cache.flush_write_through_acks()

        # Check if decode out of memory
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens

            self.num_retracted_reqs = len(retracted_reqs)
            if self.enable_metrics and len(retracted_reqs) > 0:
                self.metrics_collector.increment_retracted_reqs(
                    num_retracted_reqs=len(retracted_reqs),
                    num_retracted_input_tokens=sum(
                        len(r.origin_input_ids) for r in retracted_reqs
                    ),
                    num_retracted_output_tokens=sum(
                        len(r.output_ids) for r in retracted_reqs
                    ),
                )
            self.new_token_ratio = new_token_ratio
            for req in reqs_to_abort:
                abort_reason: FINISH_ABORT = req.to_finish
                self.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason=abort_reason.to_json(),
                        rid=req.rid,
                    ),
                    req,
                )

            msg_prefix = (
                "KV cache pool is full. Retract requests. "
                if kv_full_retract_flag
                else "Testing retraction. "
            )
            msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
            if kv_full_retract_flag:
                msg_details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
                )
            logger.warning(msg_prefix + msg_details)

            for req in retracted_reqs:
                self._add_request_to_queue(req, is_retracted=True)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        if batch.is_empty():
            return batch

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def record_batch_in_overlap(self, model_worker_batch: ModelWorkerBatch):
        # FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC
        # NOTE: More Reliable: record all tensors into the forward stream
        # NOTE: - for all future tensors, we shall always read from future map
        #       - for all non-future tensors (produced only by schedule stream),
        #       we shall keep its reference not being release during all the forwarding pass
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        self.batch_record_buf[self.batch_record_ct] = model_worker_batch

    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1

        # Whether to run the profiler
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Capture prefill start time for EXTEND mode
        if batch.forward_mode == ForwardMode.EXTEND:
            set_time_batch(batch.reqs, "set_prefill_run_batch_start_time")

        # Place holder handling for pd-disagg decode event loop
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)

        # Run forward
        if self.is_generation:
            if self.spec_algorithm.is_none() or self.enable_overlap:
                # In most cases, we use the model worker batch to run the forward.
                worker_batch_or_batch = batch.get_model_worker_batch()
            else:
                # In speculative decoding v1 (non-overlap) case, we use the batch directly.
                # TODO(lsyin): delete this branch after unifying the abstraction.
                worker_batch_or_batch = batch

            if self.enable_overlap:
                model_worker_batch = worker_batch_or_batch
                self.record_batch_in_overlap(model_worker_batch)

                # Sampling info will be modified during forward, so we store a copy.
                model_worker_batch.sampling_info = (
                    model_worker_batch.sampling_info.copy_for_forward()
                )

                bs = len(model_worker_batch.seq_lens)
                future_indices = self.future_map.alloc_future_indices(bs)

                with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    t_fwd_start = time.perf_counter()
                    with self.record_forward_metrics(batch):
                        batch_result = self.model_worker.forward_batch_generation(
                            model_worker_batch
                            # here pp is not compatible with overlap
                        )
                    t_fwd_ms = (time.perf_counter() - t_fwd_start) * 1000
                    count_added = 0
                    for req in batch.reqs:
                        if not req.finished() and not req.is_retracted:
                            req.time_stats.add_model_forward_time(t_fwd_ms / 1000.0)
                            count_added += 1
                    logger.debug(f"[MODEL_FWD_TIMING] overlap path: t_fwd_ms={t_fwd_ms:.1f}, forward_mode={batch.forward_mode}, batch_size={len(batch.reqs)}, count_added={count_added}")
                    # FIXME(lsyin): maybe move this to forward_batch_generation
                    batch_result.copy_done = self.device_module.Event()
                    if batch_result.delay_sample_func is None:
                        self.future_map.store_to_map(future_indices, batch_result)
                        batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
                    else:
                        batch_result.future_indices = future_indices

                # FIXME(lsyin): move this assignment elsewhere
                future_indices_or_next_token_ids = -future_indices.indices

                if batch.is_spec_v2:
                    # FIXME(lsyin): tmp code for spec v2
                    # We only keep future indices for next draft input

                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_indices = future_indices

                    # batch.spec_info = EagleDraftInput(
                    #     future_indices=future_indices,
                    #     verify_done=batch_result.next_draft_input.verify_done,
                    # )

                    # The future value, usually for next batch preparation
                    # Current implementation strictly synchronizes the seq_lens
                    batch.seq_lens = batch_result.next_draft_input.new_seq_lens
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                future_indices_or_next_token_ids = batch_result.next_token_ids
            else:
                kwargs = (
                    {"pp_proxy_tensors": pp_proxy_tensors}
                    if self.spec_algorithm.is_none()
                    else {}
                )
                t_fwd_start = time.perf_counter()
                with self.record_forward_metrics(batch):
                    batch_result = self.model_worker.forward_batch_generation(
                        worker_batch_or_batch, **kwargs
                    )
                t_fwd_ms = (time.perf_counter() - t_fwd_start) * 1000
                count_added = 0
                for req in batch.reqs:
                    if not req.finished() and not req.is_retracted:
                        req.time_stats.add_model_forward_time(t_fwd_ms / 1000.0)
                        count_added += 1
                logger.debug(f"[MODEL_FWD_TIMING] non-overlap path: t_fwd_ms={t_fwd_ms:.1f}, forward_mode={batch.forward_mode}, batch_size={len(batch.reqs)}, count_added={count_added}")
                future_indices_or_next_token_ids = batch_result.next_token_ids
                self.update_cache_from_scheduler(batch, batch_result)

            # NOTE: future_indices_or_next_token_ids is used in ScheduleBatch,
            #       which can probably be replaced by future_indices later [TODO(lsyin)].
            #       we shall still keep the original outputs, e.g. next_token_ids
            #       in the GenerationBatchOutput for processing after copy_done.
            batch.output_ids = future_indices_or_next_token_ids

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob:
                batch_result.extend_input_len_per_req = [
                    req.extend_input_len for req in batch.reqs
                ]
                batch_result.extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                batch_result.extend_input_len_per_req = None
                batch_result.extend_logprob_start_len_per_req = None

            ret = batch_result
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()

            if self.enable_overlap:
                self.record_batch_in_overlap(model_worker_batch)
                with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    embeddings = self.tp_worker.forward_batch_embedding(
                        model_worker_batch
                    )
                    ret = EmbeddingBatchResult(embeddings=embeddings)
                    ret.copy_to_cpu()
            else:
                embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
                ret = EmbeddingBatchResult(embeddings=embeddings)

        # Capture prefill end time for EXTEND mode
        if batch.forward_mode == ForwardMode.EXTEND:
            set_time_batch(batch.reqs, "set_prefill_run_batch_end_time")

        if (
            self.server_args.enable_dp_attention
            and self.server_args.elastic_ep_backend is not None
        ):
            # Get the tensors indicating rank activeness
            tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
            tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
            tp_active_ranks &= tp_active_ranks_cpu
            dp_active_ranks = tp_active_ranks.reshape(self.dp_size, -1).prod(axis=1)
            self.send_to_tokenizer.send_output(
                ActiveRanksOutput(status=dp_active_ranks.tolist())
            )

        return ret

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult
    ) -> Union[GenerationBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            self.future_map.store_to_map(batch_result.future_indices, batch_result)
            batch_result.copy_to_cpu(return_logprob=self.cur_batch.return_logprob)

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)
        elif batch.forward_mode.is_extend():
            if batch.is_dllm():
                self.process_batch_result_dllm(batch, result)
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.process_batch_result_disagg_prefill(batch, result)
            else:
                self.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_prebuilt():
            self.process_batch_result_prebuilt(batch)
        elif batch.forward_mode.is_idle():
            self.process_batch_result_idle(batch, result)

        self.log_batch_result_stats(batch, result)
        self._maybe_clear_mm_inputs(batch)
        self.maybe_send_health_check_signal()

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ipcs:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.send_to_tokenizer.send_output(
                HealthCheckOutput(
                    http_worker_ipc=self.return_health_check_ipcs.popleft()
                )
            )

    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        success = self.flush_cache()
        return FlushCacheReqOutput(success=success)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def is_fully_idle(self, for_health_check=False) -> bool:
        # Health check piggybacks on running requests in process_output.
        # Only running_batch + waiting_queue guarantee active GPU processing;
        # disagg queues (bootstrap/prealloc/transfer) may have items without
        # any request actually running on GPU — e.g. stuck handshake, full
        # KV cache, or stalled transfer — so they can't carry health info.
        # Batch running status
        idle = (
            self.running_batch.is_empty()
            and self.chunked_req is None
            and not self.dllm_manager.any_staging_reqs()
            and (self.last_batch is None or self.last_batch.is_empty())
            and (self.cur_batch is None or self.cur_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and (self.pp_size == 1 or all(x.is_empty() for x in self.running_mbs))
        )

        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)
        idle &= len(self.waiting_queue) == 0

        if not for_health_check:
            # Grammar queue and prefill inflight queue may not produce batch
            # results instantly, but they still indicate the server is not idle.
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0

            # HiCache: in-flight async ops (GPU↔Host↔L3) must drain before
            # destructive operations like attach/detach/flush_cache.
            if self.enable_hierarchical_cache:
                tc = self.tree_cache
                idle &= len(tc.ongoing_write_through) == 0
                idle &= len(tc.ongoing_load_back) == 0
                if tc.enable_storage:
                    idle &= len(tc.ongoing_prefetch) == 0
                    idle &= len(tc.ongoing_backup) == 0

        return idle

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return AttachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return AttachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject attach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "attach_storage_backend"):
            return AttachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic attach.",
            )

        try:
            ok, msg = self.tree_cache.attach_storage_backend(
                storage_backend=recv_req.hicache_storage_backend,
                storage_backend_extra_config_json=recv_req.hicache_storage_backend_extra_config_json,
                served_model_name=self.server_args.served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            self.enable_hicache_storage = True
            self.server_args.hicache_storage_backend = recv_req.hicache_storage_backend
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                self.server_args.hicache_storage_backend_extra_config = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                self.server_args.hicache_storage_prefetch_policy = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                self.server_args.hicache_write_policy = recv_req.hicache_write_policy
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return DetachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return DetachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject detach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "detach_storage_backend"):
            return DetachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic detach.",
            )

        # Idempotent detach: even if scheduler thinks storage is disabled, we still
        # attempt best-effort cleanup in tree_cache (it may have leftover state).
        try:
            ok, msg = self.tree_cache.detach_storage_backend()
        except Exception as e:
            logger.exception("Detach HiCache storage backend failed with exception.")
            return DetachHiCacheStorageReqOutput(success=False, message=str(e))

        if ok or (not self.enable_hicache_storage):
            # Treat "already disabled / nothing to do" as success for idempotence.
            self.enable_hicache_storage = False
            self.server_args.hicache_storage_backend = None
            self.server_args.hicache_storage_backend_extra_config = None
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    def pin_prefix_wrapped(self, recv_req: PinPrefixReqInput):
        if not hasattr(self.tree_cache, "pin_prefix"):
            return PinPrefixReqOutput(
                success=False,
                nodes_pinned=0,
                message="PIN requires --enable-hierarchical-cache",
            )
        if getattr(self.tree_cache, "_max_pinned_tokens", 0) <= 0:
            return PinPrefixReqOutput(
                success=False,
                nodes_pinned=0,
                message="Pinning is disabled (SGLANG_HICACHE_MAX_PINNED_RATIO is 0)",
            )
        nodes_pinned, reject_reason = self.tree_cache.pin_prefix(
            recv_req.token_ids, recv_req.ttl_seconds
        )
        if nodes_pinned == 0:
            return PinPrefixReqOutput(
                success=False,
                nodes_pinned=0,
                message=reject_reason or "No matching prefix found in cache to pin",
            )
        msg = f"Pinned {nodes_pinned} nodes (ttl={recv_req.ttl_seconds}s)"
        if reject_reason:
            msg += f"; {reject_reason}"
        return PinPrefixReqOutput(
            success=True,
            nodes_pinned=nodes_pinned,
            message=msg,
        )

    def flush_cache(self):
        """Flush the memory pool and cache."""
        if self.is_fully_idle():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.grammar_manager.clear()
            self.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            # TODO: allow optional empty cache
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            success = False
        return success

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = vars(get_global_server_args())
        ret["last_gen_throughput"] = self.last_gen_throughput
        ret["memory_usage"] = {
            "weight": round(self.tp_worker.model_runner.weight_load_mem_usage, 2),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
            "graph": round(self.tp_worker.model_runner.graph_mem_usage, 2),
        }
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests

        if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
            ret["avg_spec_accept_length"] = (
                self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict

        # This field is not serializable.
        ret.pop("model_config", None)

        return GetInternalStateReqOutput(internal_state=ret)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "pp_max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
            ]
        )

        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "pp_max_micro_batch_size" and (
                v > self.max_running_requests // self.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.pp_size}]."
                )
                if_success = False
                break

        if if_success:
            if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
                avg_spec_accept_length = (
                    self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.spec_total_num_accepted_tokens = self.spec_total_num_forward_ct = 0
            for k, v in server_args_dict.items():
                setattr(get_global_server_args(), k, v)
            logger.info(f"Global server args updated! {get_global_server_args()=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=vars(get_global_server_args()),
        )

    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)
            if recv_req.parameters is not None:
                func(**recv_req.parameters)
            else:
                func()
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier()
        return RpcReqOutput(success, "" if not exec else str(exec))

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            self.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                release_kv_cache(req, self.tree_cache)
            # For disaggregation prefill mode, free the metadata buffer index
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                release_req_to_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )

            # For mamba radix cache
            if (
                req.mamba_pool_idx is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        # Abort method 2: call `set_finish_with_abort`
        # The request will still run one prefill forward pass.
        # In this case, we change the input_ids to be only one token to make this prefill cheap.
        self.grammar_manager.abort_requests(recv_req)

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # Abort in-flight requests
            for req in self.disagg_prefill_inflight_queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort inflight queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests waiting for kvcache to release tree cache
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests already retracted to CPU cache
            if self.disagg_decode_prealloc_queue.retracted_queue:
                remaining_retracted = []
                for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
                    if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                        assert hasattr(decode_req, "kv_cache_cpu")
                        del decode_req.kv_cache_cpu
                        self.send_to_tokenizer.send_output(
                            AbortReq(rid=decode_req.rid), decode_req
                        )
                    else:
                        remaining_retracted.append(decode_req)
                self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted

        # Delete requests in the running batch
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_finish = FINISH_ABORT()

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        self._engine_paused = True

        if self.enable_overlap and self.last_batch:
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            chunked_req_to_exclude = set()
            if recv_req.mode == "in_place":
                if self.chunked_req is not None:
                    chunked_req_to_exclude.add(self.chunked_req)
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            self.running_batch.merge_batch(self.last_batch)

        self.last_batch = None
        self.cur_batch = None

        if recv_req.mode == "retract":
            self.running_batch.filter_batch(v1_spec_info_filtered=True)
            if len(self.running_batch.reqs) != 0:
                retracted_reqs = self.running_batch.retract_all(self.server_args)
                for req in retracted_reqs:
                    self._add_request_to_queue(req)

            self.running_batch.batch_is_full = False
            self.chunked_req = None

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        self._engine_paused = False

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """In-place loading a new lora adapter from serialized tensors."""

        result = self.tp_worker.load_lora_adapter_from_tensors(recv_req)
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""

        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(success, message)

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success, message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.forward_sleep_time = t
        return SlowDownReqOutput()

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        action = recv_req.action
        if action == ExpertDistributionReqType.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif action == ExpertDistributionReqType.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif action == ExpertDistributionReqType.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    def open_session(self, recv_req: OpenSessionReqInput):
        return self.session_controller.open(recv_req)

    def close_session(self, recv_req: CloseSessionReqInput):
        self.session_controller.close(recv_req)

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.send_to_detokenizer.send_output(recv_req, recv_req)
        return None

    def handle_dumper_control(self, recv_req: DumperControlReqInput):
        from sglang.srt.debug_utils.dumper import dumper

        try:
            response: list = []
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                response = dumper._http_manager.handle_request(
                    method=recv_req.method, body=recv_req.body
                )
            self.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass

    def get_remote_instance_transfer_engine_info(self):
        return self.tp_worker.get_remote_instance_transfer_engine_info()


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """

    def __init__(self, sockets):
        self.poller = zmq.Poller()
        self.last_empty_time = real_time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        self.poller.poll(1000)
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            torch.cuda.empty_cache()


def is_health_check_generate_req(recv_req):
    rid = getattr(recv_req, "rid", None)
    return rid is not None and rid.startswith("HEALTH_CHECK")


def is_work_request(recv_req):
    return isinstance(
        recv_req,
        (
            TokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            BatchTokenizedEmbeddingReqInput,
        ),
    )


def _parse_gpu_indices_env(var_name: str) -> List[int]:
    """Parse a comma-separated list of GPU indices from an environment variable."""
    val = os.environ.get(var_name, "")
    if not val:
        return []
    try:
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    except ValueError:
        logger.warning("Invalid %s value: %r, ignoring", var_name, val)
        return []


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
    ):
        if self.socket is None:
            return

        if (
            isinstance(recv_obj, BaseReq)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # handle communicator reqs for multi-http worker case
            output.http_worker_ipc = recv_obj.http_worker_ipc

        self.socket.send_pyobj(output)


def dispatch_event_loop(scheduler: Scheduler):
    # Dispatch to the appropriate event loop based on the disaggregation mode
    from sglang.srt.layers.afd import afd_is_attn, afd_is_ffn

    server_args = scheduler.server_args
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if afd_is_attn() or afd_is_ffn():
            scheduler.event_loop_afd()
        elif scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if afd_is_attn():
            scheduler.event_loop_afd_disagg_prefill()
        elif afd_is_ffn():
            scheduler.event_loop_afd()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if afd_is_attn():
            scheduler.event_loop_afd_disagg_decode()
        elif afd_is_ffn():
            scheduler.event_loop_afd()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()


def configure_scheduler(
    server_args: ServerArgs,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
) -> Optional[int]:
    """Configure scheduler worker: logging, process title, etc.

    Returns:
        dp_rank
    """
    # Generate the logger prefix
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.attn_cp_size > 1:
        prefix += f" ATTN_CP{attn_cp_rank}"
    if server_args.moe_dp_size > 1:
        prefix += f" MOE_DP{moe_dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    return dp_rank


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    dp_rank = configure_scheduler(
        server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank
    )

    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    numa_node = None
    if (numa_nodes := server_args.numa_node) is not None:
        numa_node = numa_nodes[gpu_id]
    elif envs.SGLANG_AUTO_NUMA_BIND.get():
        numa_node = get_numa_node(gpu_id)
        logger.info(f"auto get NUMA node {numa_node} for GPU {gpu_id}")
    if numa_node is not None and not envs.SGLANG_NUMA_BIND_V2.get():
        numa_bind_to_node(numa_node)

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # Send initialization info back to the parent process
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until shutdown)
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
