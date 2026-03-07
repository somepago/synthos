#!/usr/bin/env python3
"""
Memory watchdog — runs as a separate process and kills GPU-heavy jobs if memory gets too high.

Usage:
    python watchdog.py                    # default: 85% threshold, check every 5s
    python watchdog.py --threshold 80     # custom threshold
    python watchdog.py --interval 2       # check every 2 seconds

Monitors system RAM (unified memory). When usage exceeds threshold:
  1. Logs warning
  2. Finds the largest Python/CUDA process (excluding vLLM)
  3. Sends SIGTERM, waits 5s, then SIGKILL if still alive
  4. Keeps running to protect against future spikes
"""

import argparse
import os
import signal
import subprocess
import time

import psutil


def get_gpu_processes():
    """Get GPU processes from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,name",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        )
        procs = []
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                pid = int(parts[0])
                mem_mb = int(parts[1])
                name = parts[2]
                procs.append({"pid": pid, "mem_mb": mem_mb, "name": name})
        return procs
    except Exception:
        return []


def find_kill_target(protect_pids):
    """Find the largest GPU process that's not in the protect list."""
    procs = get_gpu_processes()
    # Sort by memory usage descending
    procs.sort(key=lambda p: p["mem_mb"], reverse=True)
    for p in procs:
        if p["pid"] not in protect_pids and p["pid"] != os.getpid():
            return p
    return None


def kill_process(pid, name):
    """SIGTERM then SIGKILL after 5s."""
    print(f"  Sending SIGTERM to PID {pid} ({name})")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"  PID {pid} already gone")
        return True

    for _ in range(10):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)  # check if alive
        except ProcessLookupError:
            print(f"  PID {pid} terminated gracefully")
            return True

    print(f"  SIGTERM didn't work, sending SIGKILL to PID {pid}")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return True


def main():
    parser = argparse.ArgumentParser(description="Memory watchdog")
    parser.add_argument("--threshold", type=float, default=85,
                        help="Memory usage %% threshold to trigger kill (default: 85)")
    parser.add_argument("--interval", type=float, default=5,
                        help="Check interval in seconds (default: 5)")
    parser.add_argument("--protect", type=str, default="vllm,Xorg,gnome",
                        help="Comma-separated substrings — processes matching these are protected")
    parser.add_argument("--dry_run", action="store_true",
                        help="Log what would be killed but don't actually kill")
    args = parser.parse_args()

    protect_substrings = [s.strip().lower() for s in args.protect.split(",")]
    print(f"Watchdog started: threshold={args.threshold}%, interval={args.interval}s")
    print(f"Protected processes containing: {protect_substrings}")
    if args.dry_run:
        print("DRY RUN MODE — will not actually kill processes")
    print()

    while True:
        mem = psutil.virtual_memory()
        pct = mem.percent

        if pct >= args.threshold:
            print(f"[{time.strftime('%H:%M:%S')}] ALERT: Memory at {pct:.1f}% (threshold {args.threshold}%)")
            print(f"  Total: {mem.total/1e9:.1f}GB, Used: {mem.used/1e9:.1f}GB, Available: {mem.available/1e9:.1f}GB")

            # Find processes to potentially kill
            gpu_procs = get_gpu_processes()
            print(f"  GPU processes: {len(gpu_procs)}")
            for p in sorted(gpu_procs, key=lambda x: x["mem_mb"], reverse=True):
                protected = any(s in p["name"].lower() for s in protect_substrings)
                tag = " [PROTECTED]" if protected else ""
                print(f"    PID {p['pid']}: {p['mem_mb']}MB — {p['name']}{tag}")

            # Find kill target (largest non-protected process)
            protect_pids = set()
            for p in gpu_procs:
                if any(s in p["name"].lower() for s in protect_substrings):
                    protect_pids.add(p["pid"])

            target = find_kill_target(protect_pids)
            if target:
                if args.dry_run:
                    print(f"  DRY RUN: Would kill PID {target['pid']} ({target['name']}, {target['mem_mb']}MB)")
                else:
                    print(f"  KILLING: PID {target['pid']} ({target['name']}, {target['mem_mb']}MB)")
                    kill_process(target["pid"], target["name"])
                    time.sleep(2)
                    mem_after = psutil.virtual_memory()
                    print(f"  After kill: {mem_after.percent:.1f}% ({mem_after.available/1e9:.1f}GB available)")
            else:
                print("  No killable target found (all processes are protected)")
        else:
            # Quiet periodic status every 60s
            if int(time.time()) % 60 < args.interval:
                print(f"[{time.strftime('%H:%M:%S')}] OK: Memory at {pct:.1f}%")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
