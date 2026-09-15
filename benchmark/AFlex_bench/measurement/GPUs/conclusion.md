# 常用 NVIDIA GPU 硬件能力调研

> 调研日期：2026-08-20。本文只整理 GPU 硬件规格，不讨论推理框架、分离架构、部署策略或软件优化。范围包括 NVIDIA 数据中心 B/H/A/L/T/V 系列、专业工作站 RTX 系列，以及消费级 GeForce RTX 50/40/30 系列中的常见型号。

## 1. 指标口径

- `TFLOPS` 表示每秒万亿次浮点运算，`TOPS` 表示每秒万亿次整数/低精度运算。
- 表中形如 `312 / 624*` 的数值依次表示 **稠密 / 2:4 结构化稀疏**理论峰值；`*` 表示稀疏峰值。只有模型和内核实际使用结构化稀疏时，后一个数值才有意义。
- H100/H200 产品页主要公布带稀疏的峰值，表中据此列为 `稠密 / 稀疏`。
- GeForce 与 RTX PRO 产品页常使用综合 `AI TOPS`，其精度、稀疏和累加格式口径与数据中心 GPU 的某一项 Tensor TFLOPS 不完全相同，不能直接横向相除。
- PCIe 带宽采用厂商常见的**双向合计**口径：PCIe 3.0 x16 为 32 GB/s、PCIe 4.0 x16 为 64 GB/s、PCIe 5.0 x16 为 128 GB/s；单方向约为其一半。
- NVLink 数值为每 GPU 的双向聚合峰值。PCIe 卡是否带 NVLink、可连接几张卡，与 SXM/HGX/NVL 形态不同，必须按完整 SKU 判断。
- 以下均为峰值规格，不等同于任何具体应用的实测吞吐。

## 2. 系列定位总览

| 系列 | 代表型号 | 主要硬件定位 | 典型显存 | 节点内互连 |
|---|---|---|---|---|
| B 系列 | B300、B200 | Blackwell / Blackwell Ultra 旗舰数据中心 AI/HPC | 180–288 GB HBM3e | 第五代 NVLink/NVSwitch，最高 1.8 TB/s/GPU |
| H 系列 | H200、H100、H100 NVL | Hopper 数据中心 AI/HPC，原生 FP8 | 80–141 GB HBM3/HBM3e | 第四代 NVLink，600–900 GB/s |
| A100/A800 | A100、A800 | Ampere 通用 AI/HPC，FP64/TF32/FP16 | 40/80 GB HBM2(e) | NVLink，A100 600 GB/s；A800 400 GB/s |
| A 系列中端 | A40、A30、A10 | 视觉计算、主流训练/推理、企业服务器 | 24–48 GB HBM2/GDDR6 | A40/A30 支持双卡 NVLink；A10 无 NVLink |
| L 系列 | L40S、L40、L4 | Ada 推理、图形、视频和边缘，原生 FP8 | 24–48 GB GDDR6 | 无 NVLink，PCIe |
| T/V 系列 | T4、V100 | 常见存量推理卡与上一代 HPC/训练卡 | 16–32 GB GDDR6/HBM2 | T4 无 NVLink；V100 SXM2 有 NVLink |
| RTX PRO/RTX 专业卡 | RTX PRO 6000、RTX 6000 Ada、RTX A6000 | 工作站、专业可视化、本地 AI | 48–96 GB ECC GDDR6/7 | 新型号多为 PCIe-only；RTX A6000 支持双卡 NVLink |
| GeForce RTX | RTX 50/40/30 | 消费级游戏、创作和本地 AI | 16–32 GB GDDR6X/7 | RTX 4090/5090 等无 NVLink，PCIe |

## 3. 旗舰数据中心 GPU：B 与 H 系列

| GPU / 形态 | 架构 | FP32 | 主要 Tensor 峰值 | 显存 | 显存带宽 | GPU 互连与主机接口 | MIG | 最大功耗 |
|---|---|---:|---|---|---:|---|---|---:|
| B300 SXM | Blackwell Ultra | 官方单卡资料未统一列出 | DGX B300 系统折算：FP8 约 4.5 / 9 PFLOPS*；FP4 约 9 / 18 PFLOPS* | 288 GB HBM3e | 8 TB/s | NVLink 1.8 TB/s；HGX/NVSwitch | 支持 | 最高约 1,400 W |
| B200 SXM（HGX/DGX） | Blackwell | 官方单卡资料未统一列出 | DGX B200 系统折算：FP8 约 4.5 / 9 PFLOPS*；FP4 约 9 / 18 PFLOPS* | 180 GB HBM3e | 8 TB/s | NVLink 1.8 TB/s；HGX/NVSwitch | 支持 | 最高约 1,200 W |
| H200 SXM | Hopper | 34 TFLOPS | TF32 494.5 / 989*；FP16/BF16 989.5 / 1,979*；FP8 1,979 / 3,958* TFLOPS | 141 GB HBM3e | 4.8 TB/s | NVLink 900 GB/s；PCIe 5.0 | 最多 7 个 | 最高 700 W |
| H200 NVL | Hopper | 30 TFLOPS | TF32 417.5 / 835*；FP16/BF16 835.5 / 1,671*；FP8 1,670.5 / 3,341* TFLOPS | 141 GB HBM3e | 4.8 TB/s | 2/4-way NVLink Bridge，900 GB/s/GPU；PCIe 5.0 | 最多 7 个 | 最高 600 W |
| H100 SXM | Hopper | 67 TFLOPS | TF32 494.5 / 989*；FP16/BF16 989.5 / 1,979*；FP8 1,979 / 3,958* TFLOPS | 80 GB HBM3 | 3.35 TB/s | NVLink 900 GB/s；PCIe 5.0 | 最多 7 个 | 最高 700 W |
| H100 NVL | Hopper | 60 TFLOPS | TF32 417.5 / 835*；FP16/BF16 835.5 / 1,671*；FP8 1,670.5 / 3,341* TFLOPS | 94 GB HBM3 | 3.9 TB/s | 双卡 NVLink Bridge 600 GB/s；PCIe 5.0 | 最多 7 个 | 350–400 W |
| H100 PCIe 80GB | Hopper | 48 TFLOPS | TF32 400 / 800*；FP16/BF16 800 / 1,600*；FP8 1,600 / 3,200* TFLOPS | 80 GB HBM2e | 2.0 TB/s | 双卡 NVLink Bridge 600 GB/s；PCIe 5.0 | 最多 7 个 | 350 W |

说明：这里按 NVIDIA 当前 HGX/DGX B200 系统资料采用每卡 180 GB；个别 NVIDIA 云服务参考页把通用 “B200” 写为 192 GB，不能据此覆盖具体 HGX B200 BOM。B300 是 288 GB 的 Blackwell Ultra。B200/B300 的表列计算值由 DGX 8 GPU 系统的 72 PFLOPS FP8、144 PFLOPS FP4 聚合峰值折算，属于系统规格换算，不是独立板卡数据表原值。功耗也会随 HGX/DGX/OEM 配置变化。

## 4. Ampere 数据中心 GPU：A 系列

| GPU / 形态 | FP32 | TF32 Tensor | FP16/BF16 Tensor | 显存 | 显存带宽 | NVLink / PCIe | MIG | 最大功耗 |
|---|---:|---:|---:|---|---:|---|---|---:|
| A100 80GB SXM4 | 19.5 | 156 / 312* | 312 / 624* TFLOPS | 80 GB HBM2e ECC | 2,039 GB/s | NVLink 600 GB/s；PCIe 4.0 | 7×10 GB | 400 W（部分 CTS 500 W） |
| A100 80GB PCIe | 19.5 | 156 / 312* | 312 / 624* TFLOPS | 80 GB HBM2e ECC | 1,935 GB/s | 双卡 NVLink Bridge 600 GB/s；PCIe 4.0 | 7×10 GB | 300 W |
| A100 40GB SXM4 | 19.5 | 156 / 312* | 312 / 624* TFLOPS | 40 GB HBM2 ECC | 1,555 GB/s | NVLink 600 GB/s；PCIe 4.0 | 7×5 GB | 400 W |
| A800 80GB SXM4 | 19.5 | 156 / 312* | 312 / 624* TFLOPS | 80 GB HBM2e ECC | 2,039 GB/s | NVLink 400 GB/s；PCIe 4.0 | 7×10 GB | 400 W（部分 SKU 更高） |
| A800 80GB PCIe | 19.5 | 156 / 312* | 312 / 624* TFLOPS | 80 GB HBM2e ECC | 1,935 GB/s | 双卡 NVLink Bridge 400 GB/s；PCIe 4.0 | 7×10 GB | 300 W |
| A800 40GB PCIe/Active | 19.5 | 156 / 312* | 312 / 624* TFLOPS | 40 GB HBM2 ECC | 1,555 GB/s | 双卡 NVLink Bridge 400 GB/s；PCIe 4.0 | 7×5 GB | 240–250 W |
| A40 | 37.4 | 74.8 / 149.6* | 149.7 / 299.4* TFLOPS | 48 GB GDDR6 ECC | 696 GB/s | 双卡 NVLink 112.5 GB/s；PCIe 4.0 | 不支持 | 300 W |
| A30 | 10.3 | 82 / 165* | 165 / 330* TFLOPS | 24 GB HBM2 ECC | 933 GB/s | 双卡 NVLink 200 GB/s；PCIe 4.0 | 最多 4 个 | 165 W |
| A10 | 31.2 | 62.5 / 125* | 125 / 250* TFLOPS | 24 GB GDDR6 | 600 GB/s | 无 NVLink；PCIe 4.0 | 不支持 | 150 W |

A100 与 A800 的计算和显存规格基本相同，关键硬件差异是 A800 将 NVLink 聚合带宽从 A100 的 600 GB/s 降至 400 GB/s。A40 偏图形/视觉计算，A30 偏主流 AI/HPC，A10 是低功耗单槽企业卡；它们虽然同属 A 系列，但不是同一性能等级。

## 5. Ada 数据中心 GPU：L 系列

| GPU | FP32 | TF32 Tensor | FP16/BF16 Tensor | FP8 Tensor | 显存 | 显存带宽 | 互连 | MIG | 最大功耗/形态 |
|---|---:|---:|---:|---:|---|---:|---|---|---|
| L40S | 91.6 | 183 / 366* | 362 / 733* | 733 / 1,466* TFLOPS | 48 GB GDDR6 ECC | 864 GB/s | 无 NVLink；PCIe 4.0 x16 | 不支持 | 350 W，双槽被动 |
| L40 | 90.5 | 90.5 / 181* | 181 / 362.1* | 362 / 724* TFLOPS | 48 GB GDDR6 ECC | 864 GB/s | 无 NVLink；PCIe 4.0 x16 | 不支持 | 300 W，双槽被动 |
| L4 | 30.3 | 60 / 120* | 121 / 242* | 242.5 / 485* TFLOPS | 24 GB GDDR6 | 300 GB/s | 无 NVLink；PCIe 4.0 x16 | 不支持 | 72 W，半高半长单槽被动 |

L40S 与 L40 的显存完全相同，但 L40S 提高功耗和 Tensor 吞吐，定位更偏生成式 AI；L40 更偏视觉计算；L4 强调低功耗、小尺寸和视频/推理密度。三者都没有 NVLink。

## 6. 常见存量数据中心 GPU：T4 与 V100

| GPU / 形态 | 架构 | FP32 | Tensor 峰值 | 显存 | 显存带宽 | 互连 | 最大功耗 |
|---|---|---:|---:|---|---:|---|---:|
| T4 | Turing | 8.1 TFLOPS | FP16/FP32 Mixed 65 TFLOPS；INT8 130 TOPS；INT4 260 TOPS | 16 GB GDDR6 ECC | 300 GB/s | 无 NVLink；PCIe 3.0 x16 | 70 W，低矮单槽 |
| V100 SXM2 32GB | Volta | 15.7 TFLOPS | FP16 Tensor 125 TFLOPS | 32 GB HBM2 ECC | 1,134 GB/s | NVLink 300 GB/s | 300 W |
| V100 PCIe 16/32GB | Volta | 14 TFLOPS | FP16 Tensor 112 TFLOPS | 16/32 GB HBM2 ECC | 900 GB/s | PCIe 3.0 x16 | 250 W |

T4 仍常见于存量云端推理和视频服务器；V100 是 Volta 时代训练/HPC 主力。它们不支持 Ampere 的 TF32、Ada/Hopper 的原生 FP8，也不应与新卡按不同精度峰值直接比较。

## 7. 专业工作站与服务器 RTX GPU

| GPU | 架构 | FP32 / 官方 AI 算力 | 显存 | 显存带宽 | ECC | PCIe / NVLink | MIG | 最大功耗 |
|---|---|---|---|---:|---|---|---|---:|
| RTX PRO 6000 Blackwell Workstation | RTX Blackwell | FP32 125 TFLOPS；4,000 AI TOPS | 96 GB GDDR7 | 1,792 GB/s | 是 | PCIe 5.0 x16；无 NVLink | 最多 4 个分区 | 600 W |
| RTX PRO 6000 Blackwell Max-Q | RTX Blackwell | 同架构低功耗配置，峰值低于 600 W 版 | 96 GB GDDR7 | 1,792 GB/s | 是 | PCIe 5.0 x16；无 NVLink | 支持 | 300 W |
| RTX PRO 6000 Blackwell Server Edition | RTX Blackwell | 官方约 4,000 AI TOPS（依 SKU） | 96 GB GDDR7 | 1,792 GB/s | 是 | PCIe 5.0 x16；无 NVLink | 支持 | 最高约 600 W |
| RTX 6000 Ada | Ada Lovelace | FP32 91.1；FP8 Tensor 728.5 / 1,457* TFLOPS | 48 GB GDDR6 | 960 GB/s | 是 | PCIe 4.0 x16；无 NVLink | 不支持 | 300 W |
| RTX A6000 | Ampere | FP32 38.7；Tensor 309.7* TFLOPS（官方稀疏口径） | 48 GB GDDR6 | 768 GB/s | 是 | PCIe 4.0 x16；双卡 NVLink 112.5 GB/s | 不支持 | 300 W |
| RTX A5000 | Ampere | FP32 27.8；Tensor 222.2* TFLOPS（官方稀疏口径） | 24 GB GDDR6 | 768 GB/s | 是 | PCIe 4.0 x16；双卡 NVLink 112.5 GB/s | 不支持 | 230 W |

专业 RTX 与 GeForce 使用相近的图形架构，但通常提供更大 ECC 显存、专业驱动/认证和不同形态。RTX PRO 6000 Blackwell 新增 MIG；RTX 6000 Ada 取消了 RTX A6000 上的 NVLink。

## 8. 消费级 GeForce RTX GPU

### 8.1 RTX 50 系列（Blackwell）

| GPU | CUDA Core | 官方 AI TOPS | 显存 | 显存带宽 | PCIe | NVLink | 公版 TGP |
|---|---:|---:|---|---:|---|---|---:|
| RTX 5090 | 21,760 | 3,352 | 32 GB GDDR7，512-bit | 1,792 GB/s | PCIe 5.0 | 不支持 | 575 W |
| RTX 5080 | 10,752 | 1,801 | 16 GB GDDR7，256-bit | 960 GB/s | PCIe 5.0 | 不支持 | 360 W |
| RTX 5070 Ti | 8,960 | 1,406 | 16 GB GDDR7，256-bit | 896 GB/s | PCIe 5.0 | 不支持 | 300 W |
| RTX 5070 | 6,144 | 988 | 12 GB GDDR7，192-bit | 672 GB/s | PCIe 5.0 | 不支持 | 250 W |

### 8.2 RTX 40 系列（Ada Lovelace）

| GPU | CUDA Core | FP32 / 官方 AI TOPS | 显存 | 显存带宽 | PCIe | NVLink | 公版 TGP |
|---|---:|---|---|---:|---|---|---:|
| RTX 4090 | 16,384 | 82.6 TFLOPS；1,321 AI TOPS | 24 GB GDDR6X，384-bit | 1,008 GB/s | PCIe 4.0 | 不支持 | 450 W |
| RTX 4080 SUPER | 10,240 | 52 TFLOPS；836 AI TOPS | 16 GB GDDR6X，256-bit | 736 GB/s | PCIe 4.0 | 不支持 | 320 W |
| RTX 4080 | 9,728 | 49 TFLOPS；780 AI TOPS | 16 GB GDDR6X，256-bit | 716.8 GB/s | PCIe 4.0 | 不支持 | 320 W |
| RTX 4070 Ti SUPER | 8,448 | 44 TFLOPS；706 AI TOPS | 16 GB GDDR6X，256-bit | 672 GB/s | PCIe 4.0 | 不支持 | 285 W |

### 8.3 RTX 30 系列（Ampere）

| GPU | CUDA Core | FP32 | 显存 | 显存带宽 | PCIe | NVLink | 公版 TGP |
|---|---:|---:|---|---:|---|---|---:|
| RTX 3090 Ti | 10,752 | 40.0 TFLOPS | 24 GB GDDR6X，384-bit | 1,008 GB/s | PCIe 4.0 | 双卡 NVLink | 450 W |
| RTX 3090 | 10,496 | 35.6 TFLOPS | 24 GB GDDR6X，384-bit | 936 GB/s | PCIe 4.0 | 双卡 NVLink | 350 W |
| RTX 3080 Ti | 10,240 | 34.1 TFLOPS | 12 GB GDDR6X，384-bit | 912 GB/s | PCIe 4.0 | 不支持 | 350 W |
| RTX 3080 | 8,704/8,960 | 29.8 TFLOPS 级 | 10/12 GB GDDR6X | 760/912 GB/s | PCIe 4.0 | 不支持 | 320/350 W |

消费级卡一般没有 MIG，也不提供与数据中心卡等价的 ECC/RAS、被动服务器散热和企业生命周期保证。RTX 40/50 系列旗舰卡已取消 NVLink；RTX 3090/3090 Ti 是消费级产品中仍带双卡 NVLink 的常见存量型号。

## 9. 显存与互连能力横向视图

### 9.1 显存容量层级

| 容量层级 | 代表 GPU |
|---|---|
| 180 GB HBM3e | B200 |
| 288 GB HBM3e | B300 |
| 141 GB HBM3e | H200 SXM/NVL |
| 94–96 GB | H100 NVL、RTX PRO 6000 Blackwell |
| 80 GB HBM | H100 SXM/PCIe、A100 80GB、A800 80GB |
| 48 GB GDDR | L40S、L40、A40、RTX 6000 Ada、RTX A6000 |
| 32 GB | RTX 5090、V100S/部分 V100 |
| 24 GB | A30、A10、L4、RTX A5000、RTX 4090、RTX 3090/Ti |
| 16 GB | T4、RTX 5080、RTX 4080/SUPER、RTX 5070 Ti |

### 9.2 显存带宽层级

| 带宽层级 | 代表 GPU |
|---|---|
| 8 TB/s | B200 |
| 4.8 TB/s | H200 |
| 3.35–3.9 TB/s | H100 SXM/NVL |
| 1.8–2.1 TB/s | A100/A800 80GB、RTX 5090、RTX PRO 6000 Blackwell |
| 0.9–1.2 TB/s | RTX 4090/3090 Ti、RTX 6000 Ada、A30、V100 SXM2 |
| 0.6–0.9 TB/s | L40/L40S、RTX A6000/A5000、A40、A10、RTX 4080 系列 |
| 300 GB/s | L4、T4 |

### 9.3 GPU 间互连层级

| 互连 | 代表 GPU | 每 GPU 双向聚合峰值 |
|---|---|---:|
| 第五代 NVLink/NVSwitch | B200 | 1.8 TB/s |
| 第四代 NVLink/NVSwitch | H100/H200 SXM | 900 GB/s |
| NVLink Bridge / NVL | H200 NVL、H100 NVL/PCIe | 600–900 GB/s |
| 第三代 NVLink/NVSwitch | A100 | 600 GB/s |
| 限速第三代 NVLink | A800 | 400 GB/s |
| 第三代 NVLink Bridge | A30 | 200 GB/s |
| NVLink Bridge | A40、RTX A6000/A5000 | 112.5 GB/s |
| 第二代 NVLink | V100 SXM2 | 300 GB/s |
| PCIe-only | L40S/L40/L4、A10、RTX 40/50、RTX 6000 Ada | PCIe 4.0/5.0，远低于 NVLink |

## 10. 网络支持说明

GPU 本身通常不直接集成集群网口，“网络支持”由 GPU、PCIe/NVLink 拓扑、NIC/DPU 和服务器平台共同决定：

- **B/H/A100/A800 等数据中心平台**：通常与 NVIDIA ConnectX/BlueField、InfiniBand 或 RoCE 组合，支持 GPUDirect RDMA，使 NIC 可以直接访问 GPU 显存。实际能力取决于服务器认证、PCIe Root Complex、IOMMU、驱动和 NIC。
- **L40S/L40/L4、A40/A30/A10/T4**：作为数据中心卡可进入支持 GPUDirect RDMA 的服务器生态，但是否可用仍取决于具体系统拓扑和认证；“GPU 是数据中心型号”不等于任意主板都能直通 RDMA。
- **RTX PRO/专业卡**：部分专业产品和驱动组合可进入 GPUDirect/企业生态，应按具体 SKU、驱动和平台支持矩阵确认。
- **GeForce RTX 30/40/50**：NVIDIA GPUDirect RDMA 官方文档不将 GeForce 列为支持产品；有 RDMA 网卡不等于 GeForce 可直接使用 GPUDirect RDMA。
- **PCIe P2P 与 GPUDirect RDMA 是两回事**：前者是 GPU 与 GPU 的 PCIe Peer-to-Peer，后者是 NIC/第三方设备与 GPU 显存直接交换数据。

## 11. 官方资料来源

### 当前数据中心系列

1. [NVIDIA Data Center GPU Line Card](https://docs.nvidia.com/data-center-gpu/line-card.pdf)
2. [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)
3. [NVIDIA HGX B200 技术说明](https://developer.nvidia.com/blog/nvidia-hgx-b200-reduces-embodied-carbon-emissions-intensity/)
4. [NVIDIA DGX B300 User Guide](https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html)
5. [NVIDIA HGX H200/B200/B300 组件规格](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
6. [NVIDIA H200 官方产品页](https://www.nvidia.com/en-us/data-center/h200/)
7. [NVIDIA H100 官方产品页](https://www.nvidia.com/en-us/data-center/h100/)
8. [NVIDIA H100 Datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet/)
9. [NVIDIA H100 PCIe Product Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf)
10. [NVIDIA H100 NVL Product Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf)

### A、L、T、V 系列

11. [NVIDIA A100 官方产品页](https://www.nvidia.com/en-us/data-center/a100/)
12. [NVIDIA A100 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)
13. [NVIDIA A800 Tensor Core GPU Datasheet（镜像）](https://chaoqing-i.com/upload/20231128/NVIDIA%20A800%20GPU%20Datasheet.pdf)
14. [NVIDIA A800 40GB Active](https://www.nvidia.com/en-us/products/workstations/a800/)
15. [NVIDIA A40 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a40/proviz-print-nvidia-a40-datasheet-us-nvidia-1469711-r8-web.pdf)
16. [NVIDIA A30 官方产品页](https://www.nvidia.com/en-us/data-center/products/a30-gpu/)
17. [NVIDIA A10 官方产品页](https://www.nvidia.com/en-us/data-center/products/a10-gpu/)
18. [NVIDIA L40S 官方产品页](https://www.nvidia.com/en-us/data-center/l40s/)
19. [NVIDIA L40 Datasheet](https://images.nvidia.com/content/Solutions/data-center/vgpu-L40-datasheet.pdf)
20. [NVIDIA L4 官方产品页](https://www.nvidia.com/en-us/data-center/l4/)
21. [NVIDIA T4 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)
22. [NVIDIA V100 Datasheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet.pdf)

### 专业与消费级 RTX

23. [RTX PRO 6000 Blackwell Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf)
24. [NVIDIA RTX PRO Blackwell Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf)
25. [RTX 6000 Ada 官方产品页](https://www.nvidia.com/en-us/products/workstations/rtx-6000/)
26. [RTX A6000 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a6000-datasheet.pdf)
27. [GeForce RTX 50/40/30 系列官方对比页](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/)
28. [NVIDIA RTX Blackwell Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
29. [NVIDIA Ada GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
30. [NVIDIA GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/archive/13.0.3/pdf/GPUDirect_RDMA.pdf)
