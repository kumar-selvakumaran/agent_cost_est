# E3 sensitivities (primary metric: task-weighted QCRPS)

Best method: P5. Secondary targets, leave-family-out and ablations refit the method (single seed); exclusion analyses re-evaluate the main predictions on the subset.

| analysis                                              | method   |   n_runs |   qcrps |     lo |     hi |   diff_vs_P1 |   diff_lo |   diff_hi |   rel_gain |   degradation_loso_minus_cv |
|:------------------------------------------------------|:---------|---------:|--------:|-------:|-------:|-------------:|----------:|----------:|-----------:|----------------------------:|
| primary target (reference)                            | P1       |    98501 |  0.3686 | 0.3651 | 0.3722 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| primary target (reference)                            | P5       |    98501 |  0.314  | 0.3109 | 0.3173 |      -0.0546 |   -0.0569 |   -0.0524 |     0.1481 |                    nan      |
| summarization-flagged runs excluded                   | P1       |    97054 |  0.3675 | 0.3639 | 0.3711 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| summarization-flagged runs excluded                   | P5       |    97054 |  0.3133 | 0.3102 | 0.3164 |      -0.0542 |   -0.0563 |   -0.052  |     0.1474 |                    nan      |
| hidden-reasoning models excluded (visible = billed)   | P1       |    71063 |  0.3471 | 0.3432 | 0.3511 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| hidden-reasoning models excluded (visible = billed)   | P5       |    71063 |  0.3154 | 0.3117 | 0.3192 |      -0.0317 |   -0.0335 |   -0.0299 |     0.0914 |                    nan      |
| target y_tok                                          | P1       |    98501 |  0.5326 | 0.5276 | 0.5374 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| target y_tok                                          | P5       |    98501 |  0.4582 | 0.4537 | 0.4626 |      -0.0744 |   -0.0774 |   -0.0715 |     0.1398 |                    nan      |
| target y_hi                                           | P1       |    98501 |  0.3606 | 0.3573 | 0.364  |     nan      |  nan      |  nan      |   nan      |                    nan      |
| target y_hi                                           | P5       |    98501 |  0.3078 | 0.3049 | 0.3109 |      -0.0528 |   -0.055  |   -0.0506 |     0.1463 |                    nan      |
| leave-family-out: code_repair                         | P1       |    20476 |  0.6639 | 0.6555 | 0.672  |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: code_repair             | P1       |    20476 |  0.3127 | 0.3067 | 0.3188 |     nan      |  nan      |  nan      |   nan      |                      0.3512 |
| leave-family-out: code_repair                         | P5       |    20476 |  0.5896 | 0.5812 | 0.5977 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: code_repair             | P5       |    20476 |  0.2856 | 0.2798 | 0.2915 |     nan      |  nan      |  nan      |   nan      |                      0.304  |
| leave-family-out: competitive_programming             | P1       |       81 |  1.7498 | 1.4565 | 2.0648 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: competitive_programming | P1       |       81 |  0.6909 | 0.583  | 0.8076 |     nan      |  nan      |  nan      |   nan      |                      1.0589 |
| leave-family-out: competitive_programming             | P5       |       81 |  0.9995 | 0.7959 | 1.2297 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: competitive_programming | P5       |       81 |  0.6831 | 0.5683 | 0.8079 |     nan      |  nan      |  nan      |   nan      |                      0.3165 |
| leave-family-out: data_qa                             | P1       |    32946 |  0.4678 | 0.4617 | 0.4743 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: data_qa                 | P1       |    32946 |  0.4387 | 0.4324 | 0.4457 |     nan      |  nan      |  nan      |   nan      |                      0.029  |
| leave-family-out: data_qa                             | P5       |    32946 |  0.6359 | 0.6234 | 0.6469 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: data_qa                 | P5       |    32946 |  0.421  | 0.4144 | 0.4283 |     nan      |  nan      |  nan      |   nan      |                      0.2149 |
| leave-family-out: math                                | P1       |     2071 |  1.5728 | 1.5416 | 1.6031 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: math                    | P1       |     2071 |  0.3502 | 0.3303 | 0.3705 |     nan      |  nan      |  nan      |   nan      |                      1.2227 |
| leave-family-out: math                                | P5       |     2071 |  1.4291 | 1.4007 | 1.4567 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: math                    | P5       |     2071 |  0.2536 | 0.2388 | 0.2687 |     nan      |  nan      |  nan      |   nan      |                      1.1754 |
| leave-family-out: other                               | P1       |    31363 |  0.5029 | 0.4928 | 0.5132 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: other                   | P1       |    31363 |  0.4761 | 0.4661 | 0.4852 |     nan      |  nan      |  nan      |   nan      |                      0.0268 |
| leave-family-out: other                               | P5       |    31363 |  0.5183 | 0.5085 | 0.5286 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: other                   | P5       |    31363 |  0.3159 | 0.309  | 0.3222 |     nan      |  nan      |  nan      |   nan      |                      0.2024 |
| leave-family-out: shell                               | P1       |    11564 |  0.8167 | 0.8044 | 0.8284 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: shell                   | P1       |    11564 |  0.3426 | 0.3372 | 0.3481 |     nan      |  nan      |  nan      |   nan      |                      0.4741 |
| leave-family-out: shell                               | P5       |    11564 |  0.6507 | 0.6427 | 0.6585 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| grouped-task CV on same runs: shell                   | P5       |    11564 |  0.2979 | 0.2935 | 0.3024 |     nan      |  nan      |  nan      |   nan      |                      0.3528 |
| ablation (a) model + source only                      | P5       |    98501 |  0.3693 | 0.3659 | 0.3725 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| ablation (b) + numeric text stats                     | P5       |    98501 |  0.3285 | 0.3251 | 0.3319 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| ablation (c) + keywords                               | P5       |    98501 |  0.3277 | 0.3245 | 0.3311 |     nan      |  nan      |  nan      |   nan      |                    nan      |
| ablation (d) + TF-IDF/SVD (full)                      | P5       |    98501 |  0.314  | 0.311  | 0.3171 |     nan      |  nan      |  nan      |   nan      |                    nan      |
