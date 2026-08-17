# CPS coordination -- thesis/defense summary

Generated: 2026-08-14T21:36:05
Sweep root: `experiments/cps_eval/scale_10k_20260813_220308`
Combos: 6 (k=0/dynamic, k=0/static, k=1/dynamic, k=1/static, k=3/dynamic, k=3/static)
Episodes per combo: 2000, 2000, 2000, 2000, 2000, 2000

State which dataset/commit these numbers came from in any write-up that cites this report, per this repo's "Self-Review Before Reporting Results" rule.

## Metric-by-metric interpretation

`std` is the sample standard deviation of that metric across episodes within a combo (episode-to-episode variance, not a standard error of the mean) -- `nan` with fewer than 2 valid episode observations.

### `success_rate`

Fraction of all aircraft that landed successfully.

**Current range across combos:** 0.9975 to 0.9999

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.9975 | nan |
| k=0/static | 0.9999 | nan |
| k=1/dynamic | 0.9984 | nan |
| k=1/static | 0.9999 | nan |
| k=3/dynamic | 0.9994 | nan |
| k=3/static | 0.9999 | nan |

### `gamma`

Combined-runway throughput: total successful landings per hour of elapsed simulation time, summed across all runways in scope.

**Current range across combos:** 23.5237 to 26.3020

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 23.5237 | 1.1025 |
| k=0/static | 26.3020 | 1.0892 |
| k=1/dynamic | 24.3723 | 0.8201 |
| k=1/static | 26.3020 | 1.0892 |
| k=3/dynamic | 24.8024 | 0.7539 |
| k=3/static | 26.3020 | 1.0892 |

### `c_sep`

Separation compliance: fraction of consecutive same-runway landing pairs that meet the RECAT-EU minimum separation (within tolerance).

**Current range across combos:** 0.8091 to 0.9039

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.8759 | 0.0517 |
| k=0/static | 0.9039 | 0.0389 |
| k=1/dynamic | 0.8169 | 0.0507 |
| k=1/static | 0.9039 | 0.0389 |
| k=3/dynamic | 0.8091 | 0.0518 |
| k=3/static | 0.9039 | 0.0389 |

### `delta_epsilon_vs_static`

Tracking degradation (Eq. tracking_degradation, RQ2.2's literal metric): mean |RTA error under CPS| minus mean |RTA error under a frozen, once-assigned static TTA|. Negative means CPS tracks the assigned arrival time MORE accurately than a static schedule would.

**Current range across combos:** -320.8483 to -190.4440

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | -320.8483 | 75.2090 |
| k=0/static | -210.2674 | 56.6651 |
| k=1/dynamic | -294.5982 | 108.5928 |
| k=1/static | -210.2674 | 56.6651 |
| k=3/dynamic | -190.4440 | 89.1300 |
| k=3/static | -210.2674 | 56.6651 |

### `delta_epsilon_vs_uncoordinated`

Secondary reference (NOT Groot et al.'s published data): mean |RTA error under CPS| minus mean |RTA error solo/uncoordinated| under the identical frozen worker.

**Current range across combos:** 0.1410 to 5.0545

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.5869 | 8.2585 |
| k=0/static | 0.1410 | 0.9511 |
| k=1/dynamic | 5.0545 | 34.2393 |
| k=1/static | 0.1410 | 0.9511 |
| k=3/dynamic | 2.6336 | 25.0208 |
| k=3/static | 0.1410 | 0.9511 |

### `r_rec`

Recovery success rate: of aircraft that received a genuine mid-trajectory TTA update, the fraction that still landed within the RTA tolerance despite the update.

**Current range across combos:** 0.9976 to 0.9999

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.9976 | 0.0075 |
| k=0/static | 0.9999 | 0.0011 |
| k=1/dynamic | 0.9984 | 0.0064 |
| k=1/static | 0.9999 | 0.0011 |
| k=3/dynamic | 0.9994 | 0.0035 |
| k=3/static | 0.9999 | 0.0011 |

### `rho_ripple`

Delay ripple index: mean per-episode lag-1 autocorrelation of consecutive aircraft's RTA errors (sorted by landing time). Positive means one aircraft's delay tends to be followed by a similarly-signed delay in the next; near zero means delays don't propagate through the landing sequence.

**Current range across combos:** -0.2094 to -0.0595

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | -0.0699 | 0.1474 |
| k=0/static | -0.2094 | 0.1400 |
| k=1/dynamic | -0.0666 | 0.1394 |
| k=1/static | -0.2094 | 0.1400 |
| k=3/dynamic | -0.0595 | 0.1351 |
| k=3/static | -0.2094 | 0.1400 |

### `stall_unrecovered`

Headline stall risk metric: fraction of ALL aircraft that were flagged stalled (distance-to-IAF plateaued) AND never landed -- the actually-costly subset, reported alongside success_rate.

**Current range across combos:** 0.0000 to 0.0005

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.0001 | 0.0013 |
| k=0/static | 0.0000 | 0.0000 |
| k=1/dynamic | 0.0005 | 0.0034 |
| k=1/static | 0.0000 | 0.0000 |
| k=3/dynamic | 0.0002 | 0.0022 |
| k=3/static | 0.0000 | 0.0000 |

### `stall_recovery_rate`

Of aircraft flagged stalled, the fraction that still landed successfully -- a mitigation-effectiveness diagnostic, not a headline risk metric on its own.

**Current range across combos:** 0.6406 to 0.6800

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.6800 | 0.4761 |
| k=0/static | nan | nan |
| k=1/dynamic | 0.6483 | 0.4646 |
| k=1/static | nan | nan |
| k=3/dynamic | 0.6406 | 0.4626 |
| k=3/static | nan | nan |

### `stall_rate`

Diagnostic only: fraction of all aircraft flagged as stalled by CPSManager. Answers 'did progress plateau', not 'did it fail' -- an aircraft can legitimately stall during path-stretching and still converge. NOT the headline risk metric (see stall_unrecovered).

**Current range across combos:** 0.0000 to 0.0014

| combo | value | std (across episodes) |
|---|---|---|
| k=0/dynamic | 0.0003 | 0.0022 |
| k=0/static | 0.0000 | 0.0000 |
| k=1/dynamic | 0.0014 | 0.0060 |
| k=1/static | 0.0000 | 0.0000 |
| k=3/dynamic | 0.0006 | 0.0039 |
| k=3/static | 0.0000 | 0.0000 |

## k-sensitivity summary ("does k-CPS relaxation help?")

### mode = dynamic (k ∈ [0, 1, 3])

| metric | k=0 | k=1 | k=3 | direction (k=min → k=max) |
|---|---|---|---|---|
| `success_rate` | 0.9975 | 0.9984 | 0.9994 | increases |
| `gamma` | 23.5237 | 24.3723 | 24.8024 | increases |
| `c_sep` | 0.8759 | 0.8169 | 0.8091 | decreases |
| `delta_epsilon_vs_static` | -320.8483 | -294.5982 | -190.4440 | increases |
| `r_rec` | 0.9976 | 0.9984 | 0.9994 | increases |
| `rho_ripple` | -0.0699 | -0.0666 | -0.0595 | increases |

### mode = static (k ∈ [0, 1, 3])

| metric | k=0 | k=1 | k=3 | direction (k=min → k=max) |
|---|---|---|---|---|
| `success_rate` | 0.9999 | 0.9999 | 0.9999 | no change |
| `gamma` | 26.3020 | 26.3020 | 26.3020 | no change |
| `c_sep` | 0.9039 | 0.9039 | 0.9039 | no change |
| `delta_epsilon_vs_static` | -210.2674 | -210.2674 | -210.2674 | no change |
| `r_rec` | 0.9999 | 0.9999 | 0.9999 | no change |
| `rho_ripple` | -0.2094 | -0.2094 | -0.2094 | no change |

## Throughput arithmetic (Finding 4 reasoning)

Raw spawn schedule rate is **75.00 ac/h** (50 arrivals / 0.6667h window), which will look inconsistent with the measured Gamma values below unless the gap is explained explicitly -- do so up front rather than let it read as a bug.

| combo | measured Γ (ac/h) | spawn/Γ ratio | why: mean episode span |
|---|---|---|---|
| k=0/dynamic | 23.52 | 3.19x | see `verify_metrics_sanity.py` output for this combo |
| k=0/static | 26.30 | 2.85x | see `verify_metrics_sanity.py` output for this combo |
| k=1/dynamic | 24.37 | 3.08x | see `verify_metrics_sanity.py` output for this combo |
| k=1/static | 26.30 | 2.85x | see `verify_metrics_sanity.py` output for this combo |
| k=3/dynamic | 24.80 | 3.02x | see `verify_metrics_sanity.py` output for this combo |
| k=3/static | 26.30 | 2.85x | see `verify_metrics_sanity.py` output for this combo |

The gap is explained by queuing against `max_concurrent_aircraft`'s slot cap plus in-sector dwell/holding time, NOT a residual bug -- confirmed via `verify_metrics_sanity.py`'s independent from-scratch Gamma recomputation and spawn-rate-vs-measured-rate arithmetic. Re-run that script for the exact per-combo numbers backing this claim.

## Flagged-but-unresolved call-outs (anticipated defense questions)

- No `stall_rate≈0` / `stall_recovery_rate`-defined inconsistency detected in this sweep root's combos.
- **Throughput arithmetic**: Γ measures well under the raw spawn schedule rate for every combo (see the throughput-arithmetic section above) -- this is explained by queuing/dwell time, not a bug, but be ready to walk through the arithmetic live if asked why Γ looks low relative to the spawn rate.
