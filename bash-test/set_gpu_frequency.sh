#!/usr/bin/env bash
# 批量设置 / 重置 NVIDIA GPU 图形核心频率（通过 nvidia-smi 锁定 GPU SM 时钟）。
# 通常需要 root（或与 nvidia-smi 等效的权限）；部分机型/策略可能禁止锁频。
#
# 示例：
#   sudo ./set_gpu_frequency.sh --gpus 0,1 --gpu-clock 1410
#   sudo ./set_gpu_frequency.sh --gpus 0-3 --gpu-clock 1200,1500
#   sudo ./set_gpu_frequency.sh --gpus all --reset-gpu-clocks
#   ./set_gpu_frequency.sh --list
#   ./set_gpu_frequency.sh --interactive   # 按提示输入卡号与频率
#  ./set_gpu_frequency.sh --gpus all --gpu-clock 210

set -euo pipefail

usage() {
  sed -n '1,30p' "$0" | tail -n +2
  cat <<'EOF'

用法:
  set_gpu_frequency.sh [选项]

选项:
  -g, --gpus IDS        GPU 索引，逗号分隔；支持区间如 0-3,7；或 all 表示全部卡
  -c, --gpu-clock SPEC  锁定 GPU SM 时钟(MHz)：单值如 1410，或区间 min,max
  -M, --mem-clock SPEC  锁定显存时钟(MHz)：单值或 min,max（可选）
  -r, --reset-gpu       重置 GPU 核心时钟到默认
  -R, --reset-mem       重置显存时钟到默认
  -n, --dry-run         只打印将执行的命令，不实际执行
  -l, --list            列出可见 GPU 及每张卡的部分可用图形时钟（可能较慢）
  -I, --interactive     交互式：输入卡号与频率
  -h, --help            显示本帮助

环境变量:
  NVIDIA_SMI            nvidia-smi 路径，默认 nvidia-smi
EOF
}

NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
GPUS_SPEC=""
GPU_CLOCK=""
MEM_CLOCK=""
RESET_GPU=0
RESET_MEM=0
DRY_RUN=0
LIST_ONLY=0
INTERACTIVE=0

die() { echo "错误: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "未找到命令: $1"
}

gpu_count() {
  # 避免 nvidia-smi | head 在 pipefail 下触发 SIGPIPE(141)
  local raw
  raw="$("$NVIDIA_SMI" --query-gpu=count --format=csv,noheader,nounits 2>/dev/null)"
  echo "$raw" | head -1 | tr -d ' '
}

list_gpus() {
  echo "=== GPU 列表 ==="
  "$NVIDIA_SMI" --query-gpu=index,name,uuid --format=csv
}

list_supported_gpu_clocks() {
  local n
  n="$(gpu_count)"
  [[ -n "$n" && "$n" =~ ^[0-9]+$ ]] || die "无法获取 GPU 数量，请确认 nvidia-smi 可用"
  local i
  for ((i = 0; i < n; i++)); do
    echo ""
    echo "=== GPU $i 支持的 graphics 时钟 (节选，MHz) ==="
    local clocks_raw
    clocks_raw="$("$NVIDIA_SMI" --query-supported-clocks=gpu --format=csv,noheader -i "$i" 2>/dev/null || true)"
    echo "$clocks_raw" | head -20
    local total
    total="$(echo "$clocks_raw" | wc -l | tr -d ' ')"
    if [[ "${total:-0}" -gt 20 ]]; then
      echo "... (共 ${total} 档，完整列表请运行: $NVIDIA_SMI --query-supported-clocks=gpu -i $i)"
    fi
  done
}

# 将 "0-2,5,7-8" 展开为排序去重后的索引列表（空格分隔）
expand_gpu_ids() {
  local spec="$1"
  local out="" token a b i
  if [[ "$spec" == "all" ]]; then
    local n
    n="$(gpu_count)"
    [[ "$n" =~ ^[0-9]+$ ]] || die "无法解析 GPU 数量"
    for ((i = 0; i < n; i++)); do
      out+=" $i"
    done
    echo "$out"
    return
  fi
  IFS=',' read -r -a parts <<<"$spec"
  for token in "${parts[@]}"; do
    token="$(echo "$token" | tr -d '[:space:]')"
    [[ -z "$token" ]] && continue
    if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      a="${BASH_REMATCH[1]}"
      b="${BASH_REMATCH[2]}"
      ((a <= b)) || die "区间无效: $token"
      for ((i = a; i <= b; i++)); do
        out+=" $i"
      done
    elif [[ "$token" =~ ^[0-9]+$ ]]; then
      out+=" $token"
    else
      die "无法解析 GPU 索引片段: $token"
    fi
  done
  # shellcheck disable=SC2086
  echo "$out" | tr ' ' '\n' | awk 'NF' | sort -n | uniq | tr '\n' ' '
}

validate_indices() {
  local max_idx
  max_idx=$(( $(gpu_count) - 1 ))
  local id
  for id in "$@"; do
    [[ "$id" =~ ^[0-9]+$ ]] || die "非法 GPU 索引: $id"
    (( id >= 0 && id <= max_idx )) || die "GPU 索引越界: $id (当前有效: 0-$max_idx)"
  done
}

join_csv() {
  local IFS=,
  echo "$*"
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    echo "+ $*"
    "$@"
  fi
}

interactive_mode() {
  list_gpus
  local n spec ids_a clock
  n="$(gpu_count)"
  read -r -p "请输入要设置的 GPU 索引（逗号或区间，如 0,2 或 0-3，或 all）: " spec
  [[ -n "$spec" ]] || die "未输入 GPU 索引"
  read -r -p "请输入 GPU 时钟 MHz（单值或 min,max；留空则仅询问是否重置）: " clock

  mapfile -t ids_a < <(expand_gpu_ids "$spec" | tr -s ' ' '\n' | awk 'NF')
  ((${#ids_a[@]} > 0)) || die "没有解析到任何 GPU 索引"
  validate_indices "${ids_a[@]}"

  local gpu_csv
  gpu_csv="$(join_csv "${ids_a[@]}")"

  if [[ -n "${clock:-}" ]]; then
    run "$NVIDIA_SMI" -i "$gpu_csv" --lock-gpu-clocks="$clock"
  else
    read -r -p "是否重置 GPU 时钟到默认? [y/N] " ans
    if [[ "${ans:-}" =~ ^[yY]$ ]]; then
      local id
      for id in "${ids_a[@]}"; do
        run "$NVIDIA_SMI" -i "$id" --reset-gpu-clocks
      done
    else
      die "未指定频率且未选择重置，已取消"
    fi
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpus) GPUS_SPEC="${2:-}"; shift 2 ;;
    -c|--gpu-clock) GPU_CLOCK="${2:-}"; shift 2 ;;
    -M|--mem-clock) MEM_CLOCK="${2:-}"; shift 2 ;;
    -r|--reset-gpu) RESET_GPU=1; shift ;;
    -R|--reset-mem) RESET_MEM=1; shift ;;
    -n|--dry-run) DRY_RUN=1; shift ;;
    -l|--list) LIST_ONLY=1; shift ;;
    -I|--interactive) INTERACTIVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数: $1（使用 -h 查看帮助）" ;;
  esac
done

require_cmd "$NVIDIA_SMI"

if [[ "$LIST_ONLY" -eq 1 ]]; then
  list_gpus
  list_supported_gpu_clocks
  exit 0
fi

if [[ "$INTERACTIVE" -eq 1 ]]; then
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "提示: 锁频/重置通常需要 root；当前非 root，若失败请用 sudo 重试。" >&2
  fi
  interactive_mode
  exit 0
fi

[[ -n "$GPUS_SPEC" ]] || die "请指定 --gpus（或使用 --interactive / --list）"

mapfile -t IDS < <(expand_gpu_ids "$GPUS_SPEC" | tr -s ' ' '\n' | awk 'NF')
((${#IDS[@]} > 0)) || die "没有解析到任何 GPU 索引"
validate_indices "${IDS[@]}"

GPU_CSV="$(join_csv "${IDS[@]}")"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "提示: 锁频/重置通常需要 root；当前非 root，若失败请用 sudo 重试。" >&2
fi

if [[ "$RESET_GPU" -eq 1 ]]; then
  run "$NVIDIA_SMI" -i "$GPU_CSV" --reset-gpu-clocks
fi
if [[ "$RESET_MEM" -eq 1 ]]; then
  run "$NVIDIA_SMI" -i "$GPU_CSV" --reset-memory-clocks
fi

if [[ -n "$GPU_CLOCK" ]]; then
  run "$NVIDIA_SMI" -i "$GPU_CSV" --lock-gpu-clocks="$GPU_CLOCK"
fi
if [[ -n "$MEM_CLOCK" ]]; then
  run "$NVIDIA_SMI" -i "$GPU_CSV" --lock-memory-clocks="$MEM_CLOCK"
fi

if [[ "$RESET_GPU" -eq 0 && "$RESET_MEM" -eq 0 && -z "$GPU_CLOCK" && -z "$MEM_CLOCK" ]]; then
  die "未指定任何操作：请提供 --gpu-clock / --mem-clock 或 --reset-gpu / --reset-mem"
fi

echo "完成。"
