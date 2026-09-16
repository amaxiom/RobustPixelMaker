"""Is this machine quiet enough for a timing measurement to mean anything?

Wall-clock timings taken while other jobs are running are not a property of the code
under test, they are a property of whatever else happened to be scheduled. Rather than
leaving that as a caveat in prose, the timing benchmark checks it, records the answer
next to every number, and refuses to run by default when the machine is busy.

Two measurements, because either alone misleads:

  * **System CPU utilisation**, sampled over a real interval. This is the test that
    matters, since contention is what perturbs a measurement.
  * **Actively working competing processes.** Existence is not contention: a dormant
    interpreter someone left open consumes no cores and cannot perturb anything, yet an
    existence-only check would block a timing run on it forever, which is how a careful
    precondition turns into a deadlock. Each competing process is therefore tagged with
    its CPU share, and only the working ones count against the gate.

A single utilisation sample is also not enough on a bursty workload: a test suite
between batches can read 30% and be back at 95% seconds later. Callers that care should
require several consecutive quiet samples (``await_idle_timing`` does).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def _powershell(script: str, timeout: float = 30.0) -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                             capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip()
    except Exception:
        return ""


def cpu_busy_percent(sample_seconds: float = 3.0) -> float | None:
    """System-wide CPU utilisation, sampled over an interval. None if unavailable."""
    try:
        import psutil                                    # optional dependency
        return float(psutil.cpu_percent(interval=sample_seconds))
    except Exception:
        pass
    if sys.platform == "win32":
        # counter sampled twice: the first read of a rate counter is meaningless
        raw = _powershell(
            "$c = New-Object System.Diagnostics.PerformanceCounter "
            "'Processor','% Processor Time','_Total'; $null = $c.NextValue(); "
            f"Start-Sleep -Seconds {int(max(1, sample_seconds))}; "
            "[math]::Round($c.NextValue(),1)",
            timeout=sample_seconds + 25)
        try:
            return float(raw)
        except ValueError:
            return None
    try:
        one, _, _ = os.getloadavg()
        return 100.0 * one / max(1, os.cpu_count() or 1)
    except Exception:
        return None


def _process_cpu() -> dict:
    """pid -> percent of ONE core currently used, for python processes."""
    if sys.platform != "win32":
        return {}
    raw = _powershell(
        "Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | "
        "Where-Object {$_.Name -like 'python*'} | "
        "ForEach-Object { \"$($_.IDProcess)`t$($_.PercentProcessorTime)\" }")
    out = {}
    for line in raw.splitlines():
        pid, _, pct = line.partition("\t")
        try:
            out[int(pid.strip())] = float(pct.strip())
        except ValueError:
            continue
    return out


def competing_processes(active_threshold: float = 5.0) -> list[dict]:
    """Other python processes, each tagged with whether it is actually working."""
    me = os.getpid()
    try:
        parent = os.getppid()
    except Exception:
        parent = -1
    cpu = _process_cpu()
    found = []

    def add(pid, cmd):
        pct = cpu.get(pid, float("nan"))
        found.append({"pid": pid, "cpu_percent": pct,
                      "active": bool(pct == pct and pct >= active_threshold),
                      "cmd": cmd[:120]})

    try:
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            if "python" not in (p.info["name"] or "").lower():
                continue
            if p.info["pid"] in (me, parent):
                continue
            add(p.info["pid"], " ".join(p.info["cmdline"] or []))
        return found
    except Exception:
        pass

    if sys.platform == "win32":
        raw = _powershell(
            "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
            "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
        for line in raw.splitlines():
            pid, _, cmd = line.partition("\t")
            try:
                pid = int(pid.strip())
            except ValueError:
                continue
            if pid in (me, parent):
                continue
            add(pid, cmd.strip())
    return found


def snapshot(sample_seconds: float = 3.0) -> dict:
    """Everything a timing number needs recorded beside it to be interpretable."""
    import numpy, scipy, sklearn

    busy = cpu_busy_percent(sample_seconds)
    procs = competing_processes()
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_count": os.cpu_count(),
        "cpu_busy_percent": busy,
        "n_competing_python": len(procs),
        "n_active_python": sum(1 for p in procs if p.get("active")),
        "competing": procs,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def is_idle(snap: dict, max_busy: float = 25.0, max_competing: int = 0) -> tuple[bool, str]:
    """Verdict plus the reason, so a refusal explains itself."""
    reasons = []
    busy = snap.get("cpu_busy_percent")
    if busy is not None and busy > max_busy:
        reasons.append(f"CPU {busy:.0f}% busy (limit {max_busy:.0f}%)")
    active = snap.get("n_active_python", snap.get("n_competing_python", 0))
    if active > max_competing:
        reasons.append(f"{active} actively working python process(es) "
                       f"(limit {max_competing})")
    if busy is None:
        reasons.append("CPU utilisation could not be sampled")
    return (not reasons), "; ".join(reasons) or "machine is quiet"


if __name__ == "__main__":
    snap = snapshot()
    ok, why = is_idle(snap)
    print(f"idle: {ok}  ({why})")
    print(f"cores {snap['cpu_count']}, busy {snap['cpu_busy_percent']}%, "
          f"competing python {snap['n_competing_python']} "
          f"({snap['n_active_python']} actively working)")
    for p in snap["competing"][:10]:
        tag = "ACTIVE" if p.get("active") else "idle  "
        print(f"   {tag} pid {p['pid']} cpu {p.get('cpu_percent')}%: {p['cmd'][:78]}")
