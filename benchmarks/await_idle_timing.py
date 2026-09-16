"""Wait for the machine to go quiet, then run the timing benchmark automatically.

Three preconditions, all required, because a timing run that starts too early produces
numbers that have to be thrown away:

  1. The main benchmark suite has finished (it is itself a heavy competing job).
  2. The machine is quiet by ``machine_state.is_idle``.
  3. It has been quiet for several **consecutive** polls.

The third condition earns its place. A test suite between batches genuinely reads 30%
CPU and is back at 95% seconds later; firing on a single quiet sample would start a
measurement directly into the next burst. Requiring a run of quiet polls costs a few
minutes of delay and removes that failure mode.

Polls on a fixed interval, appends one line per poll to a watch log so the wait is
auditable afterwards, and gives up at a deadline rather than waiting forever, saying so
plainly rather than recording contaminated numbers as if they were clean.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from machine_state import is_idle, snapshot          # noqa: E402

WATCH_LOG = HERE / "results" / "TIMING_WATCH.log"
SUITE_LOG = HERE / "results" / "RUN_ALL.log"


def suite_finished() -> bool:
    if not SUITE_LOG.exists():
        return True                                   # nothing to wait for
    text = SUITE_LOG.read_text(encoding="utf-8", errors="ignore")
    return "suite finished" in text


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-minutes", type=float, default=10.0)
    ap.add_argument("--max-hours", type=float, default=48.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-busy", type=float, default=25.0)
    ap.add_argument("--consecutive", type=int, default=3,
                    help="consecutive quiet polls required before measuring")
    args = ap.parse_args()

    deadline = time.time() + args.max_hours * 3600
    log(f"watch started; need the suite finished and {args.consecutive} consecutive "
        f"quiet polls (CPU <= {args.max_busy:g}%, no actively working competitors); "
        f"poll {args.poll_minutes:g} min, give up after {args.max_hours:g} h")

    streak = 0
    while time.time() < deadline:
        done = suite_finished()
        snap = snapshot(sample_seconds=4.0)
        idle, why = is_idle(snap, max_busy=args.max_busy)
        streak = streak + 1 if (done and idle) else 0
        log(f"suite_finished={done}  idle={idle}  streak={streak}/{args.consecutive}  "
            f"({why})")

        if streak >= args.consecutive:
            log("preconditions met on consecutive polls; starting the timing benchmark")
            r = subprocess.run(
                [sys.executable, str(HERE / "run_timing.py"),
                 "--repeats", str(args.repeats), "--max-busy", str(args.max_busy)],
                cwd=str(HERE.parent))
            log(f"timing benchmark exited with code {r.returncode}")
            return r.returncode

        time.sleep(args.poll_minutes * 60)

    log(f"gave up after {args.max_hours:g} h; the machine never went quiet for "
        f"{args.consecutive} consecutive polls. Timings were NOT measured, "
        "deliberately, rather than recorded as if clean.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
