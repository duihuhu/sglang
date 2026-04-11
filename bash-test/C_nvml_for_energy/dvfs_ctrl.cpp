#include "nvml_energy_api.h"

#include <cstdio>

#include <nvml.h>

static nvmlReturn_t handle_by_index(unsigned int index, nvmlDevice_t *out_dev) {
    if (!out_dev)
        return NVML_ERROR_INVALID_ARGUMENT;
    return nvmlDeviceGetHandleByIndex(index, out_dev);
}

extern "C" int nvml_energy_init(void) {
    return static_cast<int>(nvmlInit());
}

extern "C" void nvml_energy_shutdown(void) {
    nvmlShutdown();
}

extern "C" int nvml_energy_device_count(unsigned int *out_count) {
    if (!out_count)
        return static_cast<int>(NVML_ERROR_INVALID_ARGUMENT);
    return static_cast<int>(nvmlDeviceGetCount(out_count));
}

extern "C" int nvml_energy_get_total_mj(unsigned int index,
                                        unsigned long long *out_mj) {
    if (!out_mj)
        return static_cast<int>(NVML_ERROR_INVALID_ARGUMENT);
    nvmlDevice_t dev{};
    nvmlReturn_t r = handle_by_index(index, &dev);
    if (r != NVML_SUCCESS)
        return static_cast<int>(r);
    r = nvmlDeviceGetTotalEnergyConsumption(dev, out_mj);
    return static_cast<int>(r);
}

extern "C" int nvml_clk_set_applications_mhz(unsigned int index,
                                             unsigned int mem_mhz,
                                             unsigned int graphics_mhz) {
    nvmlDevice_t dev{};
    nvmlReturn_t r = handle_by_index(index, &dev);
    if (r != NVML_SUCCESS)
        return static_cast<int>(r);
    r = nvmlDeviceSetApplicationsClocks(dev, mem_mhz, graphics_mhz);
    return static_cast<int>(r);
}

extern "C" int nvml_clk_reset_applications(unsigned int index) {
    nvmlDevice_t dev{};
    nvmlReturn_t r = handle_by_index(index, &dev);
    if (r != NVML_SUCCESS)
        return static_cast<int>(r);
    r = nvmlDeviceResetApplicationsClocks(dev);
    return static_cast<int>(r);
}

extern "C" int nvml_clk_set_gpu_locked_mhz(unsigned int index,
                                           unsigned int min_mhz,
                                           unsigned int max_mhz) {
    nvmlDevice_t dev{};
    nvmlReturn_t r = handle_by_index(index, &dev);
    if (r != NVML_SUCCESS)
        return static_cast<int>(r);
    r = nvmlDeviceSetGpuLockedClocks(dev, min_mhz, max_mhz);
    return static_cast<int>(r);
}

extern "C" int nvml_clk_reset_gpu_locked(unsigned int index) {
    nvmlDevice_t dev{};
    nvmlReturn_t r = handle_by_index(index, &dev);
    if (r != NVML_SUCCESS)
        return static_cast<int>(r);
    r = nvmlDeviceResetGpuLockedClocks(dev);
    return static_cast<int>(r);
}

extern "C" int dvfs_lock_sm_clock(int gpu_idx, unsigned int sm_clock_mhz) {
    if (gpu_idx < 0)
        return -1;
    nvmlDevice_t dev{};
    nvmlReturn_t r =
        handle_by_index(static_cast<unsigned int>(gpu_idx), &dev);
    if (r != NVML_SUCCESS) {
        std::fprintf(stderr, "[dvfs_ctrl] GetHandle(%d) failed: %s\n", gpu_idx,
                     nvmlErrorString(r));
        return static_cast<int>(r);
    }
    r = nvmlDeviceSetGpuLockedClocks(dev, sm_clock_mhz, sm_clock_mhz);
    if (r != NVML_SUCCESS) {
        std::fprintf(
            stderr,
            "[dvfs_ctrl] SetGpuLockedClocks(%d, %u, %u) failed: %s\n", gpu_idx,
            sm_clock_mhz, sm_clock_mhz, nvmlErrorString(r));
    }
    return static_cast<int>(r);
}

extern "C" int dvfs_unlock_sm_clock(int gpu_idx) {
    if (gpu_idx < 0)
        return -1;
    nvmlDevice_t dev{};
    nvmlReturn_t r =
        handle_by_index(static_cast<unsigned int>(gpu_idx), &dev);
    if (r != NVML_SUCCESS) {
        std::fprintf(stderr, "[dvfs_ctrl] GetHandle(%d) failed: %s\n", gpu_idx,
                     nvmlErrorString(r));
        return static_cast<int>(r);
    }
    r = nvmlDeviceResetGpuLockedClocks(dev);
    if (r != NVML_SUCCESS) {
        std::fprintf(stderr, "[dvfs_ctrl] ResetGpuLockedClocks(%d) failed: %s\n",
                     gpu_idx, nvmlErrorString(r));
    }
    return static_cast<int>(r);
}

extern "C" const char *nvml_energy_strerror(int nvml_ret) {
    return nvmlErrorString(static_cast<nvmlReturn_t>(nvml_ret));
}
