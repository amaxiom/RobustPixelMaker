# Why the timings in this directory are provisional

`TIMING_SUMMARY.md` exists, and every number in it is stamped `clean_run: false`. This
note records why: the automatic watch never found a quiet machine, and the measurement
was subsequently taken under load on request, with the contamination declared rather
than hidden.

## What was attempted

`run_timing.py` measures fit-time scaling across sample count, image side, patch size,
bootstrap count and representation count, plus a head-to-head against the competitors.
It refuses to run on a busy machine, because a wall-clock number taken under contention
describes the load rather than the code, and a number that looks authoritative but is
not is worse than no number at all.

`await_idle_timing.py` polled every ten minutes from 2026-09-03 17:07 to 2026-09-05
09:15, waiting for the machine to satisfy the precondition. It never did.

## What the wait actually showed

| measurement | value |
|---|---|
| Polls taken | 238, over 40 hours |
| Polls where CPU utilisation met the gate (25%) | 122 (51%) |
| Polls where it did not | 116, spanning 25 to 98% |
| Polls blocked by an actively working competing process | 238 (100%) |
| Longest run of consecutive quiet polls | 0 of the 3 required |

Throughout, an unrelated long-running Python job on this workstation held between one
and four and a half cores of the eight available. System CPU utilisation dipped below
the threshold about half the time, but never with the machine actually free, and never
for three consecutive polls.

## Why the gate was not simply relaxed

Twice during the wait the readings tempted a shortcut, and both times checking rather
than assuming showed the gate was right.

Fluctuating utilisation (34%, 94%, 41%, 93%) suggested the machine was intermittently
idle and the threshold too strict. Per-process sampling showed two competing processes
consuming 287% and 450% of a core, roughly 92% of the machine, with five consecutive
sustained samples reading a steady 94%. The low readings were single-sample noise
between test batches. The gate was correct; the samples were not.

That investigation did expose two genuine defects in the gate itself, both fixed rather
than worked around (commit `51f0e61`). Process **existence** was being treated as
contention, so a dormant interpreter, including the watcher's own zero-percent process,
would have blocked measurement until the deadline expired. And a **single** sample was
being treated as a verdict, so one lucky low reading could have launched a measurement
directly into the next burst. Neither fix lowered the threshold.

## What was eventually recorded

On 2026-09-06 the benchmark was run with `--force`, which measures anyway and stamps
`clean_run: false` into the CSV, the JSON and the summary. Those numbers are in
`TIMING_SUMMARY.md` and `timing_results.csv`. They are usable for *shape* (how cost
scales, and how methods rank against each other) and not usable as absolute costs: the
machine was 16.5% busy at the start and 75.9% at the end, and the largest
median-to-minimum spread was x1.78, which is direct evidence of interference during the
run.

## What would produce quotable timings

Run this when the machine is genuinely free:

```bash
python benchmarks/run_timing.py --repeats 3
```

It will refuse if the machine is busy, and say why. To record provisional numbers
anyway, `--force` runs the measurement and stamps `clean_run: false` into the CSV, the
JSON and the summary, so a forced run can never be mistaken for a clean one.

## What this does not affect

Nothing else in the benchmark suite depends on it. Every scientific result, the scores,
coverage, stability, recovery and the representation ablation, is a property of the
selections rather than of wall-clock time, and all of it reproduced on a loaded machine
without difficulty. The `seconds` column present in the other result files remains
usable for ordering methods within a single run, and unusable as an absolute cost, which
is exactly what it said before this attempt began.
