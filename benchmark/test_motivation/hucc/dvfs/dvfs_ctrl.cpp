/*
 * dvfs_ctrl.cpp — Minimal C++ wrapper around NVML frequency control APIs.
 *
 * Compiled into libdvfs_ctrl.so, loaded by Python via ctypes.
 * Supports two frequency control modes:
 *   - SetApplicationsClocks (soft hint, only up-scaling effective)
 *   - SetGpuLockedClocks    (hard lock, both up/down effective)
 *
 * Build:
 *   g++ -shared -fPIC -O2 -o libdvfs_ctrl.so dvfs_ctrl.cpp -lnvidia-ml
 *
 * Or with Makefile:
 *   make
 */

#include <nvml.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// All exported functions use C linkage for ctypes compatibility
extern "C" {

// ── Device handle cache ────────────────────────────────────────────────

#define MAX_GPUS 32

static nvmlDevice_t g_devices[MAX_GPUS] = {};
static int g_devices_init[MAX_GPUS] = {};
static unsigned int g_max_mem_clock[MAX_GPUS] = {};
static int g_max_mem_init[MAX_GPUS] = {};

// ── Lifecycle ──────────────────────────────────────────────────────────

int dvfs_init(void) {
    nvmlReturn_t ret = nvmlInit_v2();
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[dvfs_ctrl] nvmlInit failed: %s\n",
                nvmlErrorString(ret));
        return (int)ret;
    }
    return 0;
}

int dvfs_shutdown(void) {
    nvmlReturn_t ret = nvmlShutdown();
    // clear cached handles — they become invalid after shutdown
    memset(g_devices, 0, sizeof(g_devices));
    memset(g_devices_init, 0, sizeof(g_devices_init));
    memset(g_max_mem_clock, 0, sizeof(g_max_mem_clock));
    memset(g_max_mem_init, 0, sizeof(g_max_mem_init));
    return (int)ret;
}

// ── Device handle helpers ──────────────────────────────────────────────

static nvmlDevice_t get_device(int gpu_idx) {
    if (gpu_idx < 0 || gpu_idx >= MAX_GPUS) return nullptr;
    if (!g_devices_init[gpu_idx]) {
        nvmlReturn_t ret = nvmlDeviceGetHandleByIndex_v2(gpu_idx, &g_devices[gpu_idx]);
        if (ret != NVML_SUCCESS) {
            fprintf(stderr, "[dvfs_ctrl] Cannot get handle for GPU %d: %s\n",
                    gpu_idx, nvmlErrorString(ret));
            return nullptr;
        }
        g_devices_init[gpu_idx] = 1;
    }
    return g_devices[gpu_idx];
}

// Cache max memory clock per GPU (queried once, reused by set_sm_clock)
static unsigned int get_max_mem_clock(int gpu_idx) {
    if (gpu_idx < 0 || gpu_idx >= MAX_GPUS) return 0;
    if (!g_max_mem_init[gpu_idx]) {
        nvmlDevice_t dev = get_device(gpu_idx);
        if (!dev) return 0;
        nvmlReturn_t ret = nvmlDeviceGetMaxClockInfo(dev, NVML_CLOCK_MEM,
                                                      &g_max_mem_clock[gpu_idx]);
        if (ret != NVML_SUCCESS) {
            fprintf(stderr, "[dvfs_ctrl] GetMaxClockInfo(MEM) for GPU %d failed: %s\n",
                    gpu_idx, nvmlErrorString(ret));
            return 0;
        }
        g_max_mem_init[gpu_idx] = 1;
    }
    return g_max_mem_clock[gpu_idx];
}

// ── Query APIs ─────────────────────────────────────────────────────────

int dvfs_get_device_count(void) {
    unsigned int count = 0;
    nvmlReturn_t ret = nvmlDeviceGetCount_v2(&count);
    if (ret != NVML_SUCCESS) return -1;
    return (int)count;
}

// Get device name, returns 0 on success
int dvfs_get_device_name(int gpu_idx, char* buf, int buf_len) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    nvmlReturn_t ret = nvmlDeviceGetName(dev, buf, buf_len);
    return (int)ret;
}

// Get current SM (graphics) clock in MHz
int dvfs_get_sm_clock(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned int clock = 0;
    nvmlReturn_t ret = nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &clock);
    if (ret != NVML_SUCCESS) return -1;
    return (int)clock;
}

// Get current memory clock in MHz
int dvfs_get_mem_clock(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned int clock = 0;
    nvmlReturn_t ret = nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &clock);
    if (ret != NVML_SUCCESS) return -1;
    return (int)clock;
}

// Get max SM clock in MHz
int dvfs_get_max_sm_clock(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned int clock = 0;
    nvmlReturn_t ret = nvmlDeviceGetMaxClockInfo(dev, NVML_CLOCK_GRAPHICS, &clock);
    if (ret != NVML_SUCCESS) return -1;
    return (int)clock;
}

// Get max memory clock in MHz
int dvfs_get_max_mem_clock(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned int clock = 0;
    nvmlReturn_t ret = nvmlDeviceGetMaxClockInfo(dev, NVML_CLOCK_MEM, &clock);
    if (ret != NVML_SUCCESS) return -1;
    return (int)clock;
}

// Get current power usage in milliwatts
int dvfs_get_power_mw(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned int power = 0;
    nvmlReturn_t ret = nvmlDeviceGetPowerUsage(dev, &power);
    if (ret != NVML_SUCCESS) return -1;
    return (int)power;
}

// Get power limit in milliwatts
int dvfs_get_power_limit_mw(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned int limit = 0;
    nvmlReturn_t ret = nvmlDeviceGetEnforcedPowerLimit(dev, &limit);
    if (ret != NVML_SUCCESS) return -1;
    return (int)limit;
}

// Get GPU temperature in Celsius
int dvfs_get_temperature(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned int temp = 0;
    nvmlReturn_t ret = nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp);
    if (ret != NVML_SUCCESS) return -1;
    return (int)temp;
}

// Get supported memory clocks. Returns count, fills buf (up to buf_len entries).
int dvfs_get_supported_mem_clocks(int gpu_idx, unsigned int* buf, int buf_len) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned int count = (unsigned int)buf_len;
    nvmlReturn_t ret = nvmlDeviceGetSupportedMemoryClocks(dev, &count, buf);
    if (ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE) return -1;
    return (int)count;
}

// Get supported graphics clocks for a given memory clock.
// Returns count, fills buf (up to buf_len entries).
int dvfs_get_supported_sm_clocks(int gpu_idx, unsigned int mem_clock_mhz,
                                  unsigned int* buf, int buf_len) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned int count = (unsigned int)buf_len;
    nvmlReturn_t ret = nvmlDeviceGetSupportedGraphicsClocks(dev, mem_clock_mhz,
                                                             &count, buf);
    if (ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE) return -1;
    return (int)count;
}

// ── Control APIs ───────────────────────────────────────────────────────

/**
 * Set application clocks (SM + memory frequency).
 *
 * Equivalent to: nvidia-smi -ac <mem_clock_mhz>,<sm_clock_mhz>
 *
 * @param gpu_idx      GPU index
 * @param mem_clock_mhz  Memory clock in MHz (use dvfs_get_max_mem_clock for max)
 * @param sm_clock_mhz   SM/Graphics clock in MHz
 * @return 0 on success, NVML error code on failure
 */
int dvfs_set_app_clocks(int gpu_idx, unsigned int mem_clock_mhz,
                         unsigned int sm_clock_mhz) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    nvmlReturn_t ret = nvmlDeviceSetApplicationsClocks(dev, mem_clock_mhz,
                                                        sm_clock_mhz);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[dvfs_ctrl] SetApplicationsClocks(%d, mem=%u, sm=%u) "
                "failed: %s\n", gpu_idx, mem_clock_mhz, sm_clock_mhz,
                nvmlErrorString(ret));
    }
    return (int)ret;
}

/**
 * Set SM frequency only, using max memory clock.
 * Convenience wrapper: queries max mem clock, then calls SetApplicationsClocks.
 *
 * @param gpu_idx      GPU index
 * @param sm_clock_mhz SM/Graphics clock in MHz
 * @return 0 on success, NVML error code on failure
 */
int dvfs_set_sm_clock(int gpu_idx, unsigned int sm_clock_mhz) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;

    unsigned int max_mem = get_max_mem_clock(gpu_idx);
    if (max_mem == 0) return -1;

    nvmlReturn_t ret = nvmlDeviceSetApplicationsClocks(dev, max_mem, sm_clock_mhz);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[dvfs_ctrl] SetApplicationsClocks(%d, mem=%u, sm=%u) "
                "failed: %s\n", gpu_idx, max_mem, sm_clock_mhz,
                nvmlErrorString(ret));
    }
    return (int)ret;
}

/**
 * Reset application clocks to default.
 * Equivalent to: nvidia-smi -rac
 */
int dvfs_reset_clocks(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    nvmlReturn_t ret = nvmlDeviceResetApplicationsClocks(dev);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[dvfs_ctrl] ResetApplicationsClocks(%d) failed: %s\n",
                gpu_idx, nvmlErrorString(ret));
    }
    return (int)ret;
}

// ── Locked Clocks APIs (force GPU to run at exact frequency) ───────────

/**
 * Lock GPU SM clock to an exact frequency.
 * Sets both min and max to the same value → GPU forced to run at this freq.
 *
 * Equivalent to: nvidia-smi -lgc <sm_clock_mhz>,<sm_clock_mhz>
 *
 * Unlike SetApplicationsClocks (which is a "suggestion" that only boosts up),
 * SetGpuLockedClocks is a hard constraint — both up-scaling and down-scaling
 * take effect immediately on running CUDA kernels.
 *
 * @param gpu_idx      GPU index
 * @param sm_clock_mhz SM clock in MHz
 * @return 0 on success, NVML error code on failure
 */
int dvfs_lock_sm_clock(int gpu_idx, unsigned int sm_clock_mhz) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    nvmlReturn_t ret = nvmlDeviceSetGpuLockedClocks(dev, sm_clock_mhz, sm_clock_mhz);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[dvfs_ctrl] SetGpuLockedClocks(%d, %u, %u) failed: %s\n",
                gpu_idx, sm_clock_mhz, sm_clock_mhz, nvmlErrorString(ret));
    }
    return (int)ret;
}

/**
 * Lock GPU SM clock to a range [min_mhz, max_mhz].
 *
 * Equivalent to: nvidia-smi -lgc <min_mhz>,<max_mhz>
 */
int dvfs_lock_sm_clock_range(int gpu_idx, unsigned int min_mhz,
                              unsigned int max_mhz) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    nvmlReturn_t ret = nvmlDeviceSetGpuLockedClocks(dev, min_mhz, max_mhz);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[dvfs_ctrl] SetGpuLockedClocks(%d, %u, %u) failed: %s\n",
                gpu_idx, min_mhz, max_mhz, nvmlErrorString(ret));
    }
    return (int)ret;
}

/**
 * Unlock GPU clocks (remove locked clock constraint).
 *
 * Equivalent to: nvidia-smi -rgc
 */
int dvfs_unlock_sm_clock(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    nvmlReturn_t ret = nvmlDeviceResetGpuLockedClocks(dev);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[dvfs_ctrl] ResetGpuLockedClocks(%d) failed: %s\n",
                gpu_idx, nvmlErrorString(ret));
    }
    return (int)ret;
}

/**
 * Lock SM clock and return C++-measured API latency in nanoseconds.
 * Timing covers ONLY the nvmlDeviceSetGpuLockedClocks() call.
 */
int dvfs_lock_sm_clock_timed(int gpu_idx, unsigned int sm_clock_mhz,
                              long long* elapsed_ns) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev || !elapsed_ns) return -1;

    auto t0 = std::chrono::high_resolution_clock::now();
    nvmlReturn_t ret = nvmlDeviceSetGpuLockedClocks(dev, sm_clock_mhz, sm_clock_mhz);
    auto t1 = std::chrono::high_resolution_clock::now();

    *elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return (int)ret;
}

// ── Timed set (for overhead measurement) ───────────────────────────────

/**
 * Set SM clock and return the API call duration in nanoseconds.
 * Uses std::chrono::high_resolution_clock for precise timing.
 *
 * @param gpu_idx      GPU index
 * @param sm_clock_mhz SM clock in MHz
 * @param elapsed_ns   [out] API call duration in nanoseconds
 * @return 0 on success
 */
int dvfs_set_sm_clock_timed(int gpu_idx, unsigned int sm_clock_mhz,
                             long long* elapsed_ns) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev || !elapsed_ns) return -1;

    unsigned int max_mem = get_max_mem_clock(gpu_idx);
    if (max_mem == 0) return -1;

    // timing covers ONLY the SetApplicationsClocks call
    auto t0 = std::chrono::high_resolution_clock::now();
    nvmlReturn_t ret = nvmlDeviceSetApplicationsClocks(dev, max_mem, sm_clock_mhz);
    auto t1 = std::chrono::high_resolution_clock::now();

    *elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return (int)ret;
}

/**
 * Set application clocks (mem + SM) and return API call duration in nanoseconds.
 */
int dvfs_set_app_clocks_timed(int gpu_idx, unsigned int mem_clock_mhz,
                               unsigned int sm_clock_mhz, long long* elapsed_ns) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev || !elapsed_ns) return -1;

    auto t0 = std::chrono::high_resolution_clock::now();
    nvmlReturn_t ret = nvmlDeviceSetApplicationsClocks(dev, mem_clock_mhz,
                                                        sm_clock_mhz);
    auto t1 = std::chrono::high_resolution_clock::now();

    *elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return (int)ret;
}

// ── Energy consumption query ────────────────────────────────────────────

/**
 * Get total energy consumption since driver load in millijoules.
 *
 * Uses nvmlDeviceGetTotalEnergyConsumption() — a hardware counter that
 * accumulates continuously at high resolution, much more accurate than
 * sampling power and multiplying by time.
 *
 * Usage for measuring a kernel's energy:
 *   e0 = dvfs_get_energy_mj(gpu);
 *   // run kernel
 *   e1 = dvfs_get_energy_mj(gpu);
 *   energy = e1 - e0;  // millijoules
 *
 * @param gpu_idx  GPU index
 * @return cumulative energy in millijoules, or -1 on error
 */
long long dvfs_get_energy_mj(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned long long energy_mj = 0;
    nvmlReturn_t ret = nvmlDeviceGetTotalEnergyConsumption(dev, &energy_mj);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[dvfs_ctrl] GetTotalEnergyConsumption(%d) failed: %s\n",
                gpu_idx, nvmlErrorString(ret));
        return -1;
    }
    return (long long)energy_mj;
}

} // extern "C"
