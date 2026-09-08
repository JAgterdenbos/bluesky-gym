# Phase III significance testing -- notes

## Self-review gate (derived vs. published means)

All derived per-episode series' means match the published `tab:throughput_results`/`tab:delay_ripple` mean values within 1e-3.


## Cross-condition matching verification

Solo-pass values differ across conditions:
  k0_static vs k0_dynamic: 45120/96722 solo-pass rta_error_solo values differ
  k1_dynamic vs k0_dynamic: 62888/96773 solo-pass rta_error_solo values differ
  k1_static vs k0_dynamic: 45120/96722 solo-pass rta_error_solo values differ
  k3_dynamic vs k0_dynamic: 65284/95932 solo-pass rta_error_solo values differ
  k3_static vs k0_dynamic: 45120/96722 solo-pass rta_error_solo values differ

Decision: comparisons run as **independent (Mann-Whitney U / Welch t-test)**.


## Consistency check against results.tex / discussion.tex / conclusion.tex

- discussion.tex claims C_sep's static/dynamic gap 'opens up between k=0 and k=1, then holds roughly flat from k=1 to k=3'. Test result: k0-vs-k1 p_holm=3.19e-295 (sig=True, effect=1.268); k1-vs-k3 p_holm=7.34e-06 (sig=True, effect=0.142). 
  Both are statistically significant at n=2000 (high power can make even a small residual k1-vs-k3 shift detectable) -- compare the effect sizes above, not just significance, before deciding whether 'holds roughly flat' still holds as a *practical*-significance claim.


- conclusion.tex claims throughput 'converg[es] toward the static value as k increases'. Test result: k1-vs-k3 (dynamic) p_holm=3.34e-41 (sig=True); static-vs-dynamic gap at k=1 effect=-0.429 vs. at k=3 effect=-0.213. 
  Effect size shrinks from k=1 to k=3, consistent with the convergence claim.
