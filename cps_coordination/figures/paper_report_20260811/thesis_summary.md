# CPS coordination -- thesis/defense summary

Generated: 2026-08-11T15:21:36
Sweep root: `experiments/cps_eval/scale_10k_20260807_123741`
Combos: 4 (k=0/dynamic, k=0/static, k=3/dynamic, k=3/static)
Episodes per combo: 2000, 2000, 2000, 2000

State which dataset/commit these numbers came from in any write-up that cites this report, per this repo's "Self-Review Before Reporting Results" rule.

## Metric-by-metric interpretation

`std` is the sample standard deviation of that metric across episodes within a combo (episode-to-episode variance, not a standard error of the mean) -- `nan` with fewer than 2 valid episode observations.

### `success_rate`

Fraction of all aircraft that landed successfully.

**Current range across combos:** 0.9943 to 0.9989

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.9970 | nan |
| k=0/static | 0.9989 | nan |
| k=3/dynamic | 0.9943 | nan |
| k=3/static | 0.9986 | nan |

### `gamma`

Combined-runway throughput: total successful landings per hour of elapsed simulation time, summed across all runways in scope.

**Current range across combos:** 9.6345 to 10.0213

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 9.6345 | 0.5373 |
| k=0/static | 10.0213 | 0.5103 |
| k=3/dynamic | 9.8211 | 0.4704 |
| k=3/static | 9.8224 | 0.5003 |

### `c_sep`

Separation compliance: fraction of consecutive same-runway landing pairs that meet the RECAT-EU minimum separation (within tolerance).

**Current range across combos:** 0.9628 to 0.9791

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.9628 | 0.0397 |
| k=0/static | 0.9724 | 0.0344 |
| k=3/dynamic | 0.9791 | 0.0301 |
| k=3/static | 0.9747 | 0.0320 |

### `delta_epsilon_vs_static`

Tracking degradation (Eq. tracking_degradation, RQ2.2's literal metric): mean |RTA error under CPS| minus mean |RTA error under a frozen, once-assigned static TTA|. Negative means CPS tracks the assigned arrival time MORE accurately than a static schedule would.

**Current range across combos:** -644.1232 to -396.0104

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | -644.1232 | 126.0274 |
| k=0/static | -639.8090 | 96.5914 |
| k=3/dynamic | -396.0104 | 106.4914 |
| k=3/static | -601.7488 | 95.9788 |

### `delta_epsilon_vs_uncoordinated`

Secondary reference (NOT Groot et al.'s published data): mean |RTA error under CPS| minus mean |RTA error solo/uncoordinated| under the identical frozen worker.

**Current range across combos:** 0.2061 to 4.7255

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.4694 | 5.1649 |
| k=0/static | 0.2061 | 1.4961 |
| k=3/dynamic | 4.7255 | 42.1600 |
| k=3/static | 0.5529 | 1.6583 |

### `r_rec`

Recovery success rate: of aircraft that received a genuine mid-trajectory TTA update, the fraction that still landed within the RTA tolerance despite the update.

**Current range across combos:** 0.9943 to 0.9989

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.9971 | 0.0112 |
| k=0/static | 0.9989 | 0.0066 |
| k=3/dynamic | 0.9943 | 0.0153 |
| k=3/static | 0.9987 | 0.0075 |

### `rho_ripple`

Delay ripple index: mean per-episode lag-1 autocorrelation of consecutive aircraft's RTA errors (sorted by landing time). Positive means one aircraft's delay tends to be followed by a similarly-signed delay in the next; near zero means delays don't propagate through the landing sequence.

**Current range across combos:** -0.0643 to 0.0384

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.0384 | 0.2398 |
| k=0/static | -0.0643 | 0.1954 |
| k=3/dynamic | 0.0054 | 0.2101 |
| k=3/static | -0.0568 | 0.2046 |

### `stall_unrecovered`

Headline stall risk metric: fraction of ALL aircraft that were flagged stalled (distance-to-IAF plateaued) AND never landed -- the actually-costly subset, reported alongside success_rate.

**Current range across combos:** 0.0000 to 0.0003

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.0000 | 0.0000 |
| k=0/static | 0.0000 | 0.0000 |
| k=3/dynamic | 0.0003 | 0.0036 |
| k=3/static | 0.0000 | 0.0000 |

### `stall_recovery_rate`

Of aircraft flagged stalled, the fraction that still landed successfully -- a mitigation-effectiveness diagnostic, not a headline risk metric on its own.

**Current range across combos:** 0.7808 to 1.0000

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 1.0000 | 0.0000 |
| k=0/static | nan | nan |
| k=3/dynamic | 0.7808 | 0.4187 |
| k=3/static | 1.0000 | nan |

### `stall_rate`

Diagnostic only: fraction of all aircraft flagged as stalled by CPSManager. Answers 'did progress plateau', not 'did it fail' -- an aircraft can legitimately stall during path-stretching and still converge. NOT the headline risk metric (see stall_unrecovered).

**Current range across combos:** 0.0000 to 0.0015

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.0001 | 0.0018 |
| k=0/static | 0.0000 | 0.0000 |
| k=3/dynamic | 0.0015 | 0.0076 |
| k=3/static | 0.0000 | 0.0009 |

## k-sensitivity summary ("does k-CPS relaxation help?")

### mode = dynamic (k ∈ [0, 3])

| metric | k=0 | k=3 | direction (k=min → k=max) |
|---|---|---|---|
| `success_rate` | 0.9970 | 0.9943 | decreases |
| `gamma` | 9.6345 | 9.8211 | increases |
| `c_sep` | 0.9628 | 0.9791 | increases |
| `delta_epsilon_vs_static` | -644.1232 | -396.0104 | increases |
| `r_rec` | 0.9971 | 0.9943 | decreases |
| `rho_ripple` | 0.0384 | 0.0054 | decreases |

### mode = static (k ∈ [0, 3])

| metric | k=0 | k=3 | direction (k=min → k=max) |
|---|---|---|---|
| `success_rate` | 0.9989 | 0.9986 | decreases |
| `gamma` | 10.0213 | 9.8224 | decreases |
| `c_sep` | 0.9724 | 0.9747 | increases |
| `delta_epsilon_vs_static` | -639.8090 | -601.7488 | increases |
| `r_rec` | 0.9989 | 0.9987 | decreases |
| `rho_ripple` | -0.0643 | -0.0568 | increases |

## Throughput arithmetic (Finding 4 reasoning)

Raw spawn schedule rate is **37.50 ac/h** (25 arrivals / 0.6667h window), which will look inconsistent with the measured Gamma values below unless the gap is explained explicitly -- do so up front rather than let it read as a bug.

| combo | measured Γ (ac/h) | spawn/Γ ratio | why: mean episode span |
|---|---|---|---|
| k=0/dynamic | 9.63 | 3.89x | see `verify_metrics_sanity.py` output for this combo |
| k=0/static | 10.02 | 3.74x | see `verify_metrics_sanity.py` output for this combo |
| k=3/dynamic | 9.82 | 3.82x | see `verify_metrics_sanity.py` output for this combo |
| k=3/static | 9.82 | 3.82x | see `verify_metrics_sanity.py` output for this combo |

The gap is explained by queuing against `max_concurrent_aircraft`'s slot cap plus in-sector dwell/holding time, NOT a residual bug -- confirmed via `verify_metrics_sanity.py`'s independent from-scratch Gamma recomputation and spawn-rate-vs-measured-rate arithmetic. Re-run that script for the exact per-combo numbers backing this claim.

## Flagged-but-unresolved call-outs (anticipated defense questions)

- **k=3/static**: `stall_rate` rounds to 0.0000 but `stall_recovery_rate` is defined (1.0000) rather than '--'. Likely a handful of sub-rounding stall events, not a bug, but confirm the raw stall count before citing the recovery-rate figure for this combo specifically.
- **Throughput arithmetic**: Γ measures well under the raw spawn schedule rate for every combo (see the throughput-arithmetic section above) -- this is explained by queuing/dwell time, not a bug, but be ready to walk through the arithmetic live if asked why Γ looks low relative to the spawn rate.
