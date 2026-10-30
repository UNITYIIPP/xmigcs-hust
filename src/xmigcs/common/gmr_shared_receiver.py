"""Optional subprocess GMR receiver using shared memory latest slots.

This mirrors the deployment pattern used by Humanoid-GPT: keep high-rate
reference receiving/parsing out of the 50 Hz control interpreter so Python GIL
contention and callback bursts cannot block policy inference.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GMRSharedSnapshot:
    gmr_data: Optional[np.ndarray]
    gmr_recv_t: float
    gmr_counter: int
    gmr_6kp_data: Optional[np.ndarray]
    gmr_6kp_recv_t: float
    gmr_6kp_counter: int


def _normalize_cpu_ids(cpu_ids):
    if cpu_ids is None:
        return None
    if isinstance(cpu_ids, int):
        return {int(cpu_ids)}
    if isinstance(cpu_ids, str):
        result = set()
        for part in cpu_ids.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                result.update(range(int(lo), int(hi) + 1))
            else:
                result.add(int(part))
        return result or None
    return {int(cpu_id) for cpu_id in cpu_ids}


def _try_set_rt(cpu_ids, fifo_prio: Optional[int]) -> None:
    cpu_set = _normalize_cpu_ids(cpu_ids)
    if cpu_set is None and fifo_prio is None:
        return
    try:
        if cpu_set is not None:
            os.sched_setaffinity(0, cpu_set)
        if fifo_prio is not None:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(fifo_prio)))
    except (AttributeError, OSError, PermissionError, ValueError) as exc:
        print(f"[gmr_shared_receiver] RT pin failed: {exc}", flush=True)


def _receiver_worker(
    gmr_topic: str,
    gmr_6kp_topic: str,
    gmr_expected_len: int,
    gmr_6kp_expected_len: int,
    arr_gmr,
    arr_gmr_6kp,
    val_gmr_t,
    val_gmr_cnt,
    val_gmr_6kp_t,
    val_gmr_6kp_cnt,
    shared_lock,
    ready_evt,
    stop_evt,
    cpu_ids,
    fifo_prio,
) -> None:
    _try_set_rt(cpu_ids, fifo_prio)
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Float32MultiArray
    except Exception as exc:
        print(f"[gmr_shared_receiver] ROS import failed: {exc}", flush=True)
        ready_evt.set()
        return

    gmr_view = np.frombuffer(arr_gmr, dtype=np.float32)
    gmr_6kp_view = np.frombuffer(arr_gmr_6kp, dtype=np.float32)

    def write_gmr(msg) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != gmr_expected_len:
            return
        with shared_lock:
            gmr_view[:] = data
            val_gmr_t.value = time.monotonic()
            val_gmr_cnt.value += 1

    def write_gmr_6kp(msg) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != gmr_6kp_expected_len:
            return
        with shared_lock:
            gmr_6kp_view[:] = data
            val_gmr_6kp_t.value = time.monotonic()
            val_gmr_6kp_cnt.value += 1

    rclpy.init(args=None)
    node = None
    executor = None
    try:
        node = rclpy.create_node("xmigcs_gmr_shared_receiver")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        node.create_subscription(Float32MultiArray, gmr_topic, write_gmr, qos)
        node.create_subscription(Float32MultiArray, gmr_6kp_topic, write_gmr_6kp, qos)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        ready_evt.set()
        while rclpy.ok() and not stop_evt.is_set():
            executor.spin_once(timeout_sec=0.05)
    finally:
        if executor is not None and node is not None:
            try:
                executor.remove_node(node)
            except Exception:
                pass
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


class GMRSharedReceiver:
    """Parent-side manager for a ROS GMR receiver subprocess."""

    def __init__(
        self,
        gmr_expected_len: int,
        gmr_6kp_expected_len: int,
        gmr_topic: str = "/gmr_info",
        gmr_6kp_topic: str = "/gmr_6kp",
        cpu_id: Optional[int] = None,
        cpu_ids=None,
        fifo_prio: Optional[int] = None,
    ) -> None:
        self.gmr_expected_len = int(gmr_expected_len)
        self.gmr_6kp_expected_len = int(gmr_6kp_expected_len)
        self.gmr_topic = str(gmr_topic)
        self.gmr_6kp_topic = str(gmr_6kp_topic)
        self.cpu_ids = _normalize_cpu_ids(cpu_ids if cpu_ids is not None else cpu_id)
        self.fifo_prio = fifo_prio
        self._ctx = mp.get_context("spawn")
        self._arr_gmr = self._ctx.Array("f", self.gmr_expected_len, lock=False)
        self._arr_gmr_6kp = self._ctx.Array("f", self.gmr_6kp_expected_len, lock=False)
        self._val_gmr_t = self._ctx.Value("d", 0.0, lock=False)
        self._val_gmr_cnt = self._ctx.Value("Q", 0, lock=False)
        self._val_gmr_6kp_t = self._ctx.Value("d", 0.0, lock=False)
        self._val_gmr_6kp_cnt = self._ctx.Value("Q", 0, lock=False)
        self._lock = self._ctx.Lock()
        self._ready_evt = self._ctx.Event()
        self._stop_evt = self._ctx.Event()
        self._proc: Optional[mp.Process] = None
        self._gmr_view = np.frombuffer(self._arr_gmr, dtype=np.float32)
        self._gmr_6kp_view = np.frombuffer(self._arr_gmr_6kp, dtype=np.float32)
        self._last_gmr_cnt = 0
        self._last_gmr_6kp_cnt = 0

    def start(self, timeout_s: float = 5.0) -> bool:
        if self._proc is not None and self._proc.is_alive():
            return True
        self._proc = self._ctx.Process(
            target=_receiver_worker,
            args=(
                self.gmr_topic,
                self.gmr_6kp_topic,
                self.gmr_expected_len,
                self.gmr_6kp_expected_len,
                self._arr_gmr,
                self._arr_gmr_6kp,
                self._val_gmr_t,
                self._val_gmr_cnt,
                self._val_gmr_6kp_t,
                self._val_gmr_6kp_cnt,
                self._lock,
                self._ready_evt,
                self._stop_evt,
                self.cpu_ids,
                self.fifo_prio,
            ),
            name="xmigcs-gmr-rx",
            daemon=True,
        )
        self._proc.start()
        ready = bool(self._ready_evt.wait(timeout_s))
        return ready and self._proc is not None and self._proc.is_alive()

    def stop(self) -> None:
        self._stop_evt.set()
        proc = self._proc
        if proc is not None:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        self._proc = None

    def read_latest(self) -> GMRSharedSnapshot:
        with self._lock:
            gmr_cnt = int(self._val_gmr_cnt.value)
            gmr_6kp_cnt = int(self._val_gmr_6kp_cnt.value)
            gmr_data = self._gmr_view.copy() if gmr_cnt != self._last_gmr_cnt else None
            gmr_6kp_data = (
                self._gmr_6kp_view.copy()
                if gmr_6kp_cnt != self._last_gmr_6kp_cnt
                else None
            )
            snap = GMRSharedSnapshot(
                gmr_data=gmr_data,
                gmr_recv_t=float(self._val_gmr_t.value),
                gmr_counter=gmr_cnt,
                gmr_6kp_data=gmr_6kp_data,
                gmr_6kp_recv_t=float(self._val_gmr_6kp_t.value),
                gmr_6kp_counter=gmr_6kp_cnt,
            )
        self._last_gmr_cnt = gmr_cnt
        self._last_gmr_6kp_cnt = gmr_6kp_cnt
        return snap
