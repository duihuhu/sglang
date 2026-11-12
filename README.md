
## Heterogeneous AF results (2026-10-03)

The heterogeneous sweep uses `ipc_cpp` rank-0 CUDA IPC and active-GPU-only energy accounting. Three 12-GPU layouts were measured over Code/Conversation and QPS 2/4/8/16 (24/24 PASS):

- `af_a1_f2_x4`: four replicas, A=TP1/F=TP2.
- `af_a2_f4_x2`: two replicas, A=TP2/F=TP4.
- `af_a4_f2_x2`: two replicas, A=TP4/F=TP2.

For a fair comparison, all three homogeneous layouts were remeasured with the same `ipc_cpp` backend (24/24 PASS). At QPS16, only `4×AF-A1/F2` satisfies the 120 ms TPOT P90 SLO on both traces: 100.9 ms (Code) and 90.6 ms (Conversation). Its active-only energy is 107.88 and 145.64 mJ/token, respectively.

Artifacts:

- `data/macro_e2e_megascale_hetero_3layouts_24points_20261002.json`
- `data/macro_e2e_megascale_ipc_cpp_homogeneous_24points_20261002.json`
- `data/macro_e2e_megascale_homo_hetero_6layouts_48points_20261003.json`
- `charts/end_to_end_code_trace_with_megascale_hetero.pdf`
- `charts/end_to_end_conversation_trace_with_megascale_hetero.pdf`
- `charts/megascale_homo_hetero_tpot_energy.pdf`
