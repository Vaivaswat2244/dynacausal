Runs: 20 fault runs, of which 11 impactful; 3 no-fault runs.

#### Subset: all fault runs (n=20)

**1. Accuracy at fault time (window starting at inject)**

| method | AC@1 | AC@3 | AC@5 | Avg@5 | MRR | n |
|---|---|---|---|---|---|---|
| baro_inject | 0.2000 | 0.7500 | 0.9500 | 0.6700 | 0.4954 | 20 |
| dynacausal_causal | 0.4000 | 0.6500 | 0.7500 | 0.6300 | 0.5748 | 20 |
| dynacausal_causal_raw | 0.2500 | 0.6500 | 0.7500 | 0.5600 | 0.4679 | 20 |
| dynacausal_d1train | 0.0500 | 0.1500 | 0.3500 | 0.1700 | 0.2157 | 20 |
| dynacausal_percase | 0.4000 | 0.4500 | 0.6500 | 0.4800 | 0.4997 | 20 |
| metric_jump | 0.0000 | 0.1500 | 0.1500 | 0.1100 | 0.1509 | 20 |
| error_count | 0.1000 | 0.6000 | 0.6000 | 0.5000 | 0.3833 | 20 |

**2. Accuracy without fault time (method's own choice of when)**

| method | AC@1 | AC@3 | AC@5 | Avg@5 | MRR | n |
|---|---|---|---|---|---|---|
| baro_cpd | 0.2000 | 0.7000 | 0.9000 | 0.6400 | 0.4772 | 20 |
| dynacausal_causal | 0.5000 | 0.6500 | 0.7500 | 0.6500 | 0.6368 | 20 |
| dynacausal_causal_raw | 0.3000 | 0.6500 | 0.9000 | 0.6500 | 0.5329 | 20 |
| dynacausal_d1train | 0.1500 | 0.2000 | 0.5000 | 0.2800 | 0.3044 | 20 |
| dynacausal_percase | 0.3000 | 0.5000 | 0.5500 | 0.4500 | 0.4520 | 20 |
| metric_jump | 0.0500 | 0.1500 | 0.2500 | 0.1400 | 0.1864 | 20 |
| error_count | 0.1000 | 0.6000 | 0.6000 | 0.5000 | 0.3833 | 20 |

**3. Time to diagnosis**

| method | median s (diagnosed runs) | diagnosed | never within fault | share never | n |
|---|---|---|---|---|---|
| dynacausal_causal | 0 | 9 | 11 | 0.5500 | 20 |
| dynacausal_causal_raw | 15 | 11 | 9 | 0.4500 | 20 |
| dynacausal_d1train | 7.5000 | 2 | 18 | 0.9000 | 20 |
| dynacausal_percase | 0 | 9 | 11 | 0.5500 | 20 |
| metric_jump | n/a | 0 | 20 | 1.0000 | 20 |
| error_count | 0.0000 | 2 | 18 | 0.9000 | 20 |

**4. Share of windows with top score above threshold**

| method | threshold | no-fault runs | n win | pre-fault | n win | during fault | n win |
|---|---|---|---|---|---|---|---|
| dynacausal_causal | 0.5 | 0.4762 | 42 | n/a | 0 | 0.5684 | 95 |
| dynacausal_causal | 0.7 | 0.3571 | 42 | n/a | 0 | 0.4842 | 95 |
| dynacausal_causal | 0.9 | 0.2619 | 42 | n/a | 0 | 0.4105 | 95 |
| dynacausal_causal_raw | 0.5 | 0.8810 | 42 | n/a | 0 | 0.7474 | 95 |
| dynacausal_causal_raw | 0.7 | 0.8810 | 42 | n/a | 0 | 0.7158 | 95 |
| dynacausal_causal_raw | 0.9 | 0.8333 | 42 | n/a | 0 | 0.6632 | 95 |
| dynacausal_d1train | 0.5 | 1.0000 | 66 | 1.0000 | 6 | 1.0000 | 95 |
| dynacausal_d1train | 0.7 | 1.0000 | 66 | 1.0000 | 6 | 0.8421 | 95 |
| dynacausal_d1train | 0.9 | 0.0000 | 66 | 0.0000 | 6 | 0.0000 | 95 |
| dynacausal_percase | 0.5 | 0.3939 | 66 | 0.0000 | 6 | 0.4000 | 95 |
| dynacausal_percase | 0.7 | 0.3333 | 66 | 0.0000 | 6 | 0.3579 | 95 |
| dynacausal_percase | 0.9 | 0.1818 | 66 | 0.0000 | 6 | 0.2421 | 95 |

**5a. Per fault type** (at = at fault time, free = without fault time; baro_inject has only at, baro_cpd only free, the other side counts every run as last)

| method | group | n | at AC@1 | at AC@3 | at AC@5 | at MRR | free AC@1 | free AC@3 | free AC@5 | free MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| baro_cpd | cpu | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.4000 | 0.8000 | 1.0000 | 0.6167 |
| baro_cpd | delay | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.8000 | 0.8000 | 0.3556 |
| baro_cpd | loss | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.2000 | 0.4000 | 0.8000 | 0.4200 |
| baro_cpd | mem | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.2000 | 0.8000 | 1.0000 | 0.5167 |
| baro_inject | cpu | 5 | 0.4000 | 0.8000 | 1.0000 | 0.6167 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | delay | 5 | 0.2000 | 0.8000 | 1.0000 | 0.4833 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | loss | 5 | 0.2000 | 0.6000 | 0.8000 | 0.4650 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | mem | 5 | 0.0000 | 0.8000 | 1.0000 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| dynacausal_causal | cpu | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dynacausal_causal | delay | 5 | 0.2000 | 1.0000 | 1.0000 | 0.6000 | 0.6000 | 0.8000 | 1.0000 | 0.7400 |
| dynacausal_causal | loss | 5 | 0.2000 | 0.4000 | 0.6000 | 0.3786 | 0.2000 | 0.4000 | 0.4000 | 0.3952 |
| dynacausal_causal | mem | 5 | 0.2000 | 0.2000 | 0.4000 | 0.3208 | 0.2000 | 0.4000 | 0.6000 | 0.4119 |
| dynacausal_causal_raw | cpu | 5 | 0.4000 | 0.8000 | 0.8000 | 0.5619 | 0.8000 | 0.8000 | 1.0000 | 0.8500 |
| dynacausal_causal_raw | delay | 5 | 0.2000 | 0.6000 | 1.0000 | 0.4333 | 0.4000 | 0.8000 | 0.8000 | 0.5583 |
| dynacausal_causal_raw | loss | 5 | 0.4000 | 0.8000 | 0.8000 | 0.5917 | 0.0000 | 0.6000 | 1.0000 | 0.3900 |
| dynacausal_causal_raw | mem | 5 | 0.0000 | 0.4000 | 0.4000 | 0.2848 | 0.0000 | 0.4000 | 0.8000 | 0.3333 |
| dynacausal_d1train | cpu | 5 | 0.2000 | 0.2000 | 0.6000 | 0.3519 | 0.2000 | 0.2000 | 0.6000 | 0.3519 |
| dynacausal_d1train | delay | 5 | 0.0000 | 0.0000 | 0.0000 | 0.1313 | 0.0000 | 0.2000 | 0.6000 | 0.2011 |
| dynacausal_d1train | loss | 5 | 0.0000 | 0.2000 | 0.4000 | 0.1925 | 0.2000 | 0.2000 | 0.4000 | 0.3278 |
| dynacausal_d1train | mem | 5 | 0.0000 | 0.2000 | 0.4000 | 0.1872 | 0.2000 | 0.2000 | 0.4000 | 0.3369 |
| dynacausal_percase | cpu | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dynacausal_percase | delay | 5 | 0.2000 | 0.4000 | 0.8000 | 0.3667 | 0.0000 | 0.2000 | 0.2000 | 0.2139 |
| dynacausal_percase | loss | 5 | 0.4000 | 0.4000 | 0.4000 | 0.4756 | 0.2000 | 0.4000 | 0.4000 | 0.3700 |
| dynacausal_percase | mem | 5 | 0.0000 | 0.0000 | 0.4000 | 0.1567 | 0.0000 | 0.4000 | 0.6000 | 0.2241 |
| metric_jump | cpu | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0867 | 0.0000 | 0.0000 | 0.2000 | 0.1186 |
| metric_jump | delay | 5 | 0.0000 | 0.0000 | 0.0000 | 0.1071 | 0.0000 | 0.2000 | 0.2000 | 0.1786 |
| metric_jump | loss | 5 | 0.0000 | 0.2000 | 0.2000 | 0.1841 | 0.0000 | 0.0000 | 0.2000 | 0.1186 |
| metric_jump | mem | 5 | 0.0000 | 0.4000 | 0.4000 | 0.2256 | 0.2000 | 0.4000 | 0.4000 | 0.3298 |
| error_count | cpu | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| error_count | delay | 5 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 |
| error_count | loss | 5 | 0.2000 | 0.6000 | 0.6000 | 0.4333 | 0.2000 | 0.6000 | 0.6000 | 0.4333 |
| error_count | mem | 5 | 0.2000 | 0.8000 | 0.8000 | 0.5167 | 0.2000 | 0.8000 | 0.8000 | 0.5167 |

**5b. Per service** (at = at fault time, free = without fault time; baro_inject has only at, baro_cpd only free, the other side counts every run as last)

| method | group | n | at AC@1 | at AC@3 | at AC@5 | at MRR | free AC@1 | free AC@3 | free AC@5 | free MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| baro_cpd | checkoutservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.2500 | 0.7500 | 1.0000 | 0.5208 |
| baro_cpd | currencyservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.7500 | 0.7500 | 0.4000 |
| baro_cpd | emailservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.7500 | 1.0000 | 1.0000 | 0.8750 |
| baro_cpd | productcatalogservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.5000 | 0.7500 | 0.2569 |
| baro_cpd | recommendationservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.5000 | 1.0000 | 0.3333 |
| baro_inject | checkoutservice | 4 | 0.2500 | 0.7500 | 1.0000 | 0.5208 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | currencyservice | 4 | 0.0000 | 0.7500 | 0.7500 | 0.4062 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | emailservice | 4 | 0.7500 | 1.0000 | 1.0000 | 0.8750 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | productcatalogservice | 4 | 0.0000 | 0.7500 | 1.0000 | 0.3542 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | recommendationservice | 4 | 0.0000 | 0.5000 | 1.0000 | 0.3208 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| dynacausal_causal | checkoutservice | 4 | 0.5000 | 0.7500 | 1.0000 | 0.6875 | 0.2500 | 0.5000 | 0.7500 | 0.4792 |
| dynacausal_causal | currencyservice | 4 | 0.7500 | 1.0000 | 1.0000 | 0.8333 | 0.5000 | 0.7500 | 0.7500 | 0.6667 |
| dynacausal_causal | emailservice | 4 | 0.2500 | 0.5000 | 0.5000 | 0.4357 | 0.5000 | 0.5000 | 0.7500 | 0.5857 |
| dynacausal_causal | productcatalogservice | 4 | 0.2500 | 0.5000 | 0.7500 | 0.4653 | 0.5000 | 0.7500 | 0.7500 | 0.6667 |
| dynacausal_causal | recommendationservice | 4 | 0.2500 | 0.5000 | 0.5000 | 0.4524 | 0.7500 | 0.7500 | 0.7500 | 0.7857 |
| dynacausal_causal_raw | checkoutservice | 4 | 0.2500 | 0.5000 | 0.7500 | 0.4792 | 0.2500 | 0.7500 | 0.7500 | 0.5000 |
| dynacausal_causal_raw | currencyservice | 4 | 0.5000 | 1.0000 | 1.0000 | 0.7083 | 0.5000 | 0.7500 | 1.0000 | 0.6875 |
| dynacausal_causal_raw | emailservice | 4 | 0.5000 | 0.7500 | 0.7500 | 0.6250 | 0.2500 | 0.7500 | 1.0000 | 0.5208 |
| dynacausal_causal_raw | productcatalogservice | 4 | 0.0000 | 0.7500 | 0.7500 | 0.2727 | 0.0000 | 0.0000 | 0.7500 | 0.2062 |
| dynacausal_causal_raw | recommendationservice | 4 | 0.0000 | 0.2500 | 0.5000 | 0.2545 | 0.5000 | 1.0000 | 1.0000 | 0.7500 |
| dynacausal_d1train | checkoutservice | 4 | 0.0000 | 0.0000 | 0.2500 | 0.1437 | 0.0000 | 0.0000 | 0.5000 | 0.1590 |
| dynacausal_d1train | currencyservice | 4 | 0.0000 | 0.0000 | 0.7500 | 0.2107 | 0.0000 | 0.0000 | 1.0000 | 0.2500 |
| dynacausal_d1train | emailservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.1270 | 0.0000 | 0.0000 | 0.0000 | 0.1409 |
| dynacausal_d1train | productcatalogservice | 4 | 0.2500 | 0.7500 | 0.7500 | 0.4583 | 0.7500 | 1.0000 | 1.0000 | 0.8333 |
| dynacausal_d1train | recommendationservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.1389 | 0.0000 | 0.0000 | 0.0000 | 0.1389 |
| dynacausal_percase | checkoutservice | 4 | 0.5000 | 0.7500 | 1.0000 | 0.6458 | 0.2500 | 0.7500 | 0.7500 | 0.5000 |
| dynacausal_percase | currencyservice | 4 | 0.7500 | 0.7500 | 0.7500 | 0.7812 | 0.2500 | 0.5000 | 0.7500 | 0.4562 |
| dynacausal_percase | emailservice | 4 | 0.2500 | 0.2500 | 0.5000 | 0.3458 | 0.2500 | 0.2500 | 0.2500 | 0.3444 |
| dynacausal_percase | productcatalogservice | 4 | 0.2500 | 0.2500 | 0.5000 | 0.3729 | 0.2500 | 0.2500 | 0.2500 | 0.3447 |
| dynacausal_percase | recommendationservice | 4 | 0.2500 | 0.2500 | 0.5000 | 0.3528 | 0.5000 | 0.7500 | 0.7500 | 0.6146 |
| metric_jump | checkoutservice | 4 | 0.0000 | 0.2500 | 0.2500 | 0.1944 | 0.2500 | 0.2500 | 0.2500 | 0.3213 |
| metric_jump | currencyservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.1042 | 0.0000 | 0.0000 | 0.0000 | 0.0958 |
| metric_jump | emailservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0972 | 0.0000 | 0.0000 | 0.0000 | 0.0941 |
| metric_jump | productcatalogservice | 4 | 0.0000 | 0.2500 | 0.2500 | 0.1649 | 0.0000 | 0.5000 | 0.5000 | 0.2668 |
| metric_jump | recommendationservice | 4 | 0.0000 | 0.2500 | 0.2500 | 0.1936 | 0.0000 | 0.0000 | 0.5000 | 0.1540 |
| error_count | checkoutservice | 4 | 0.5000 | 0.7500 | 0.7500 | 0.6458 | 0.5000 | 0.7500 | 0.7500 | 0.6458 |
| error_count | currencyservice | 4 | 0.0000 | 0.5000 | 0.5000 | 0.2917 | 0.0000 | 0.5000 | 0.5000 | 0.2917 |
| error_count | emailservice | 4 | 0.0000 | 0.5000 | 0.5000 | 0.2917 | 0.0000 | 0.5000 | 0.5000 | 0.2917 |
| error_count | productcatalogservice | 4 | 0.0000 | 0.7500 | 0.7500 | 0.3958 | 0.0000 | 0.7500 | 0.7500 | 0.3958 |
| error_count | recommendationservice | 4 | 0.0000 | 0.5000 | 0.5000 | 0.2917 | 0.0000 | 0.5000 | 0.5000 | 0.2917 |

#### Subset: impactful fault runs (n=11)

**1. Accuracy at fault time (window starting at inject)**

| method | AC@1 | AC@3 | AC@5 | Avg@5 | MRR | n |
|---|---|---|---|---|---|---|
| baro_inject | 0.0000 | 0.6364 | 0.9091 | 0.5455 | 0.3477 | 11 |
| dynacausal_causal | 0.4545 | 0.7273 | 0.8182 | 0.6909 | 0.6237 | 11 |
| dynacausal_causal_raw | 0.1818 | 0.6364 | 0.7273 | 0.5091 | 0.4038 | 11 |
| dynacausal_d1train | 0.0909 | 0.2727 | 0.5455 | 0.2909 | 0.2786 | 11 |
| dynacausal_percase | 0.4545 | 0.4545 | 0.5455 | 0.4727 | 0.5298 | 11 |
| metric_jump | 0.0000 | 0.1818 | 0.1818 | 0.1273 | 0.1591 | 11 |
| error_count | 0.0000 | 0.6364 | 0.6364 | 0.5091 | 0.3485 | 11 |

**2. Accuracy without fault time (method's own choice of when)**

| method | AC@1 | AC@3 | AC@5 | Avg@5 | MRR | n |
|---|---|---|---|---|---|---|
| baro_cpd | 0.0000 | 0.5455 | 0.8182 | 0.4909 | 0.3146 | 11 |
| dynacausal_causal | 0.6364 | 0.8182 | 0.8182 | 0.7818 | 0.7576 | 11 |
| dynacausal_causal_raw | 0.3636 | 0.5455 | 0.9091 | 0.6364 | 0.5523 | 11 |
| dynacausal_d1train | 0.2727 | 0.3636 | 0.7273 | 0.4727 | 0.4293 | 11 |
| dynacausal_percase | 0.3636 | 0.4545 | 0.5455 | 0.4545 | 0.4845 | 11 |
| metric_jump | 0.0000 | 0.1818 | 0.3636 | 0.1636 | 0.1765 | 11 |
| error_count | 0.0000 | 0.6364 | 0.6364 | 0.5091 | 0.3485 | 11 |

**3. Time to diagnosis**

| method | median s (diagnosed runs) | diagnosed | never within fault | share never | n |
|---|---|---|---|---|---|
| dynacausal_causal | 0 | 5 | 6 | 0.5455 | 11 |
| dynacausal_causal_raw | 15 | 7 | 4 | 0.3636 | 11 |
| dynacausal_d1train | 7.5000 | 2 | 9 | 0.8182 | 11 |
| dynacausal_percase | 0 | 5 | 6 | 0.5455 | 11 |
| metric_jump | n/a | 0 | 11 | 1.0000 | 11 |
| error_count | n/a | 0 | 11 | 1.0000 | 11 |

**4. Share of windows with top score above threshold**

| method | threshold | no-fault runs | n win | pre-fault | n win | during fault | n win |
|---|---|---|---|---|---|---|---|
| dynacausal_causal | 0.5 | 0.4762 | 42 | n/a | 0 | 0.5577 | 52 |
| dynacausal_causal | 0.7 | 0.3571 | 42 | n/a | 0 | 0.4423 | 52 |
| dynacausal_causal | 0.9 | 0.2619 | 42 | n/a | 0 | 0.3077 | 52 |
| dynacausal_causal_raw | 0.5 | 0.8810 | 42 | n/a | 0 | 0.7115 | 52 |
| dynacausal_causal_raw | 0.7 | 0.8810 | 42 | n/a | 0 | 0.6731 | 52 |
| dynacausal_causal_raw | 0.9 | 0.8333 | 42 | n/a | 0 | 0.6154 | 52 |
| dynacausal_d1train | 0.5 | 1.0000 | 66 | 1.0000 | 3 | 1.0000 | 52 |
| dynacausal_d1train | 0.7 | 1.0000 | 66 | 1.0000 | 3 | 0.8077 | 52 |
| dynacausal_d1train | 0.9 | 0.0000 | 66 | 0.0000 | 3 | 0.0000 | 52 |
| dynacausal_percase | 0.5 | 0.3939 | 66 | 0.0000 | 3 | 0.4038 | 52 |
| dynacausal_percase | 0.7 | 0.3333 | 66 | 0.0000 | 3 | 0.3654 | 52 |
| dynacausal_percase | 0.9 | 0.1818 | 66 | 0.0000 | 3 | 0.2308 | 52 |

**5a. Per fault type** (at = at fault time, free = without fault time; baro_inject has only at, baro_cpd only free, the other side counts every run as last)

| method | group | n | at AC@1 | at AC@3 | at AC@5 | at MRR | free AC@1 | free AC@3 | free AC@5 | free MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| baro_cpd | cpu | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.6667 | 1.0000 | 0.3611 |
| baro_cpd | delay | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.6667 | 0.6667 | 0.3148 |
| baro_cpd | loss | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.0000 | 0.6667 | 0.2000 |
| baro_cpd | mem | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 1.0000 | 1.0000 | 0.4167 |
| baro_inject | cpu | 3 | 0.0000 | 0.6667 | 1.0000 | 0.3611 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | delay | 3 | 0.0000 | 0.6667 | 1.0000 | 0.3611 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | loss | 3 | 0.0000 | 0.3333 | 0.6667 | 0.2750 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | mem | 2 | 0.0000 | 1.0000 | 1.0000 | 0.4167 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| dynacausal_causal | cpu | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dynacausal_causal | delay | 3 | 0.3333 | 1.0000 | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dynacausal_causal | loss | 3 | 0.0000 | 0.3333 | 0.6667 | 0.2500 | 0.3333 | 0.6667 | 0.6667 | 0.5556 |
| dynacausal_causal | mem | 2 | 0.5000 | 0.5000 | 0.5000 | 0.5556 | 0.0000 | 0.5000 | 0.5000 | 0.3333 |
| dynacausal_causal_raw | cpu | 3 | 0.0000 | 0.6667 | 0.6667 | 0.2698 | 0.6667 | 0.6667 | 1.0000 | 0.7500 |
| dynacausal_causal_raw | delay | 3 | 0.3333 | 0.6667 | 1.0000 | 0.5278 | 0.6667 | 0.6667 | 0.6667 | 0.7083 |
| dynacausal_causal_raw | loss | 3 | 0.3333 | 0.6667 | 0.6667 | 0.4861 | 0.0000 | 0.3333 | 1.0000 | 0.3167 |
| dynacausal_causal_raw | mem | 2 | 0.0000 | 0.5000 | 0.5000 | 0.2955 | 0.0000 | 0.5000 | 1.0000 | 0.3750 |
| dynacausal_d1train | cpu | 3 | 0.3333 | 0.3333 | 0.6667 | 0.4722 | 0.3333 | 0.3333 | 0.6667 | 0.4722 |
| dynacausal_d1train | delay | 3 | 0.0000 | 0.0000 | 0.0000 | 0.1402 | 0.0000 | 0.3333 | 0.6667 | 0.2315 |
| dynacausal_d1train | loss | 3 | 0.0000 | 0.3333 | 0.6667 | 0.2315 | 0.3333 | 0.3333 | 0.6667 | 0.4537 |
| dynacausal_d1train | mem | 2 | 0.0000 | 0.5000 | 1.0000 | 0.2667 | 0.5000 | 0.5000 | 1.0000 | 0.6250 |
| dynacausal_percase | cpu | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dynacausal_percase | delay | 3 | 0.3333 | 0.3333 | 0.6667 | 0.4333 | 0.0000 | 0.3333 | 0.3333 | 0.2454 |
| dynacausal_percase | loss | 3 | 0.3333 | 0.3333 | 0.3333 | 0.4259 | 0.3333 | 0.3333 | 0.3333 | 0.4167 |
| dynacausal_percase | mem | 2 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 | 0.0000 | 0.5000 | 0.1714 |
| metric_jump | cpu | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0889 | 0.0000 | 0.0000 | 0.3333 | 0.1421 |
| metric_jump | delay | 3 | 0.0000 | 0.0000 | 0.0000 | 0.1136 | 0.0000 | 0.3333 | 0.3333 | 0.2303 |
| metric_jump | loss | 3 | 0.0000 | 0.3333 | 0.3333 | 0.2421 | 0.0000 | 0.0000 | 0.3333 | 0.1303 |
| metric_jump | mem | 2 | 0.0000 | 0.5000 | 0.5000 | 0.2083 | 0.0000 | 0.5000 | 0.5000 | 0.2167 |
| error_count | cpu | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| error_count | delay | 3 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 |
| error_count | loss | 3 | 0.0000 | 0.6667 | 0.6667 | 0.3611 | 0.0000 | 0.6667 | 0.6667 | 0.3611 |
| error_count | mem | 2 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 |

**5b. Per service** (at = at fault time, free = without fault time; baro_inject has only at, baro_cpd only free, the other side counts every run as last)

| method | group | n | at AC@1 | at AC@3 | at AC@5 | at MRR | free AC@1 | free AC@3 | free AC@5 | free MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| baro_cpd | currencyservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.7500 | 0.7500 | 0.4000 |
| baro_cpd | productcatalogservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.5000 | 0.7500 | 0.2569 |
| baro_cpd | recommendationservice | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 | 0.3333 | 1.0000 | 0.2778 |
| baro_inject | currencyservice | 4 | 0.0000 | 0.7500 | 0.7500 | 0.4062 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | productcatalogservice | 4 | 0.0000 | 0.7500 | 1.0000 | 0.3542 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| baro_inject | recommendationservice | 3 | 0.0000 | 0.3333 | 1.0000 | 0.2611 | 0.0000 | 0.0000 | 0.0000 | 0.0833 |
| dynacausal_causal | currencyservice | 4 | 0.7500 | 1.0000 | 1.0000 | 0.8333 | 0.5000 | 0.7500 | 0.7500 | 0.6667 |
| dynacausal_causal | productcatalogservice | 4 | 0.2500 | 0.5000 | 0.7500 | 0.4653 | 0.5000 | 0.7500 | 0.7500 | 0.6667 |
| dynacausal_causal | recommendationservice | 3 | 0.3333 | 0.6667 | 0.6667 | 0.5556 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dynacausal_causal_raw | currencyservice | 4 | 0.5000 | 1.0000 | 1.0000 | 0.7083 | 0.5000 | 0.7500 | 1.0000 | 0.6875 |
| dynacausal_causal_raw | productcatalogservice | 4 | 0.0000 | 0.7500 | 0.7500 | 0.2727 | 0.0000 | 0.0000 | 0.7500 | 0.2062 |
| dynacausal_causal_raw | recommendationservice | 3 | 0.0000 | 0.0000 | 0.3333 | 0.1726 | 0.6667 | 1.0000 | 1.0000 | 0.8333 |
| dynacausal_d1train | currencyservice | 4 | 0.0000 | 0.0000 | 0.7500 | 0.2107 | 0.0000 | 0.0000 | 1.0000 | 0.2500 |
| dynacausal_d1train | productcatalogservice | 4 | 0.2500 | 0.7500 | 0.7500 | 0.4583 | 0.7500 | 1.0000 | 1.0000 | 0.8333 |
| dynacausal_d1train | recommendationservice | 3 | 0.0000 | 0.0000 | 0.0000 | 0.1296 | 0.0000 | 0.0000 | 0.0000 | 0.1296 |
| dynacausal_percase | currencyservice | 4 | 0.7500 | 0.7500 | 0.7500 | 0.7812 | 0.2500 | 0.5000 | 0.7500 | 0.4562 |
| dynacausal_percase | productcatalogservice | 4 | 0.2500 | 0.2500 | 0.5000 | 0.3729 | 0.2500 | 0.2500 | 0.2500 | 0.3447 |
| dynacausal_percase | recommendationservice | 3 | 0.3333 | 0.3333 | 0.3333 | 0.4037 | 0.6667 | 0.6667 | 0.6667 | 0.7083 |
| metric_jump | currencyservice | 4 | 0.0000 | 0.0000 | 0.0000 | 0.1042 | 0.0000 | 0.0000 | 0.0000 | 0.0958 |
| metric_jump | productcatalogservice | 4 | 0.0000 | 0.2500 | 0.2500 | 0.1649 | 0.0000 | 0.5000 | 0.5000 | 0.2668 |
| metric_jump | recommendationservice | 3 | 0.0000 | 0.3333 | 0.3333 | 0.2247 | 0.0000 | 0.0000 | 0.6667 | 0.1636 |
| error_count | currencyservice | 4 | 0.0000 | 0.5000 | 0.5000 | 0.2917 | 0.0000 | 0.5000 | 0.5000 | 0.2917 |
| error_count | productcatalogservice | 4 | 0.0000 | 0.7500 | 0.7500 | 0.3958 | 0.0000 | 0.7500 | 0.7500 | 0.3958 |
| error_count | recommendationservice | 3 | 0.0000 | 0.6667 | 0.6667 | 0.3611 | 0.0000 | 0.6667 | 0.6667 | 0.3611 |

**9. Pipeline** (per window = scaling + graph + forward pass; load = reading and bucketing one run's files; for BARO the load column is its whole run time in s)

| method | ms/window median | ms/window p95 | load s/run median | windows scored | windows skipped | peak RSS MB | share of inputs outside 0..1 |
|---|---|---|---|---|---|---|---|
| baro_cpd | n/a | n/a | 427.5065 | n/a | n/a | n/a | n/a |
| baro_inject | n/a | n/a | 0.2202 | n/a | n/a | n/a | n/a |
| dynacausal_causal | 5.4891 | 9.2047 | 0.8147 | 323 | 184 | 428.9961 | 0.0000 |
| dynacausal_causal_raw | 5.6054 | 11.0951 | 0.8281 | 323 | 184 | 432.2617 | 0.1100 |
| dynacausal_d1train | 5.6401 | 8.5966 | 0.8062 | 507 | 0 | 431.9492 | 0.0897 |
| dynacausal_percase | 5.3347 | 9.4124 | 0.8062 | 507 | 0 | 432.1562 | 0.0000 |

