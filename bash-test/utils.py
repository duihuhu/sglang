import os
import subprocess

def set_all_visible_gpus_clock(mem_clock: int, graphics_clock: int):
    """
    根据 CUDA_VISIBLE_DEVICES 获取所有可见 GPU，并设置相同的显存和核心频率。
    
    参数:
        mem_clock: 显存频率 (MHz)
        graphics_clock: 核心频率 (MHz)
    """
    # 1️⃣ 获取可见 GPU 列表
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        print("⚠️ 没有检测到 CUDA_VISIBLE_DEVICES，默认修改所有 GPU")
        # 获取所有 GPU 数量
        try:
            result = subprocess.run("nvidia-smi -L", shell=True, check=True, capture_output=True, text=True)
            cuda_visible = ",".join([str(i) for i, line in enumerate(result.stdout.splitlines())])
        except subprocess.CalledProcessError as e:
            print("获取 GPU 列表失败:\n", e.stderr)
            return False

    gpu_ids = [int(x) for x in cuda_visible.split(",")]

    # 2️⃣ 遍历每个 GPU 设置频率
    for gpu_id in gpu_ids:
        print(f"设置 GPU {gpu_id} ...")

        # # 获取支持频率
        # try:
        #     check_cmd = f"nvidia-smi -i {gpu_id} -q -d SUPPORTED_CLOCKS"
        #     result = subprocess.run(check_cmd, shell=True, check=True, capture_output=True, text=True)
        #     output = result.stdout
        # except subprocess.CalledProcessError as e:
        #     print(f"GPU {gpu_id} 获取支持频率失败:\n", e.stderr)
        #     continue

        # # 简单判断是否在支持范围
        # if str(mem_clock) not in output or str(graphics_clock) not in output:
        #     print(f"⚠️ GPU {gpu_id} 频率可能不支持: 核心 {graphics_clock} MHz, 显存 {mem_clock} MHz")

        # 设置频率
        cmd = f"nvidia-smi -i {gpu_id} -ac {mem_clock},{graphics_clock}"
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"GPU {gpu_id} 设置成功:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"GPU {gpu_id} 设置失败:\n", e.stderr)

# 使用示例
# set_all_visible_gpus_clock(mem_clock=210, graphics_clock=1593)
# 注册退出时自动恢复频率
def reset_clocks():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        print("⚠️ 没有检测到 CUDA_VISIBLE_DEVICES，默认修改所有 GPU")
        # 获取所有 GPU 数量
        try:
            result = subprocess.run("nvidia-smi -L", shell=True, check=True, capture_output=True, text=True)
            cuda_visible = ",".join([str(i) for i, line in enumerate(result.stdout.splitlines())])
        except subprocess.CalledProcessError as e:
            print("获取 GPU 列表失败:\n", e.stderr)
            return False

    gpu_ids = [int(x) for x in cuda_visible.split(",")]
    for gpu_id in gpu_ids:
        print(f"恢复 GPU {gpu_id} 默认频率 ...")
        cmd = f"nvidia-smi -i {gpu_id} -rac"
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"GPU {gpu_id} 已恢复默认频率")
        except subprocess.CalledProcessError as e:
            print(f"GPU {gpu_id} 恢复失败:\n{e.stderr}")