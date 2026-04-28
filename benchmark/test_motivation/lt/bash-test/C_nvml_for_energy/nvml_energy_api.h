/**
 * C ABI: NVML init, application clocks (-ac/-rac), SM locked clocks (-lgc-like),
 * and total energy (mJ).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int nvml_energy_init(void);
void nvml_energy_shutdown(void);

int nvml_clk_set_applications_mhz(unsigned int index, unsigned int mem_mhz,
                                  unsigned int graphics_mhz);

int nvml_clk_reset_applications(unsigned int index);

/**
 * Lock SM clocks to [min_mhz, max_mhz] (nvidia-smi -lgc style, Volta+).
 * Often min==max. Can take effect under active CUDA; typically needs root.
 */
int nvml_clk_set_gpu_locked_mhz(unsigned int index, unsigned int min_mhz,
                                unsigned int max_mhz);

/** Clear GPU locked clocks (nvidia-smi reset locked clocks). Volta+. */
int nvml_clk_reset_gpu_locked(unsigned int index);

/**
 * Lock SM to a single MHz (min=max). @p gpu_idx < 0 returns -1.
 * Otherwise returns nvmlReturn_t as int; logs to stderr on failure.
 */
int dvfs_lock_sm_clock(int gpu_idx, unsigned int sm_clock_mhz);

/** Clear SM lock. @p gpu_idx < 0 returns -1. */
int dvfs_unlock_sm_clock(int gpu_idx);

/** GPU count (informational). */
int nvml_energy_device_count(unsigned int *out_count);

/**
 * Total energy since driver reload, millijoules (Volta+).
 * Not power integration — hardware counter delta for benchmarking.
 */
int nvml_energy_get_total_mj(unsigned int index, unsigned long long *out_mj);

const char *nvml_energy_strerror(int nvml_ret);

#ifdef __cplusplus
}
#endif
