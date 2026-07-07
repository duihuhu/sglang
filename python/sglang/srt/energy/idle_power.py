"""Idle / bubble power helpers for AF DVFS energy modeling.

Measured idle power (W) per GPU at each frequency on A800-SXM4-80GB
(memory resident, no active compute).  Source: layer-profile motivation
experiments (see fig5_AF_ratio).
"""

from __future__ import annotations

# MHz -> idle power (W)
IDLE_POWER_TABLE: dict[int, float] = {
    210: 65.81,
    450: 66.31,
    690: 67.28,
    930: 68.86,
    1170: 73.50,
    1410: 90.02,
}

_DEFAULT_IDLE_POWER_W = 80.0


def idle_power_w(freq_mhz: int) -> float:
    """Return measured idle power (W) for a locked GPU frequency."""
    return IDLE_POWER_TABLE.get(int(freq_mhz), _DEFAULT_IDLE_POWER_W)


def layer_bubble_energy_mj(
    lat_a_us: float,
    lat_f_us: float,
    f_a: int,
    f_f: int,
    tp_a: int,
    tp_f: int,
    M: int = 1,
) -> float:
    """Estimate per-layer pipeline bubble energy for AF-disaggregated execution.

    M == 1 (serial): the side not computing waits for the other side.
    M > 1 (pipelined): faster side idles for |t_A - t_F| scaled by (M-1)/M.
    """
    if lat_a_us <= 0 or lat_f_us <= 0:
        return 0.0

    if M <= 1:
        # A runs then F (or overlapped via comm): both sides idle part of the time.
        e_a_idle = idle_power_w(f_a) * lat_f_us * tp_a * 1e-3
        e_f_idle = idle_power_w(f_f) * lat_a_us * tp_f * 1e-3
        return e_a_idle + e_f_idle

    t_wait_us = abs(lat_a_us - lat_f_us)
    if t_wait_us <= 0:
        return 0.0
    scale = (M - 1) / M
    if lat_a_us >= lat_f_us:
        return idle_power_w(f_f) * t_wait_us * tp_f * scale * 1e-3
    return idle_power_w(f_a) * t_wait_us * tp_a * scale * 1e-3


def scheduler_idle_energy_mj(
    t_idle_us: float,
    freq_mhz: int,
    n_gpus: int,
) -> float:
    """Static power consumed while GPUs wait between scheduler batches."""
    if t_idle_us <= 0 or n_gpus <= 0:
        return 0.0
    return idle_power_w(freq_mhz) * t_idle_us * n_gpus * 1e-3
