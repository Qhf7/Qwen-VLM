# B0w-5crop 消融

| prompt_id | aggregation | sample_auroc | sample_aupr | sample_f1max | macro_class_auroc | accuracy_at_0.5 | hard_yes_rate_at_0.5 | ng_gt_ok_classes | n_evaluable_classes | ok_score_median | ng_score_median | score_min | score_max | score_dynamic_range | n_samples | five_view_latency_p50_s | five_view_latency_p95_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0_direct | max | 0.725000 | 0.787922 | 0.750000 | 0.700000 | 0.500000 | 0.000000 | 7.000000 | 10.000000 | 0.204541 | 0.224070 | 0.162388 | 0.362985 | 0.200597 | 20.000000 |  |  |
| P0_5crop | max | 0.570000 | 0.669486 | 0.714286 | 0.700000 | 0.500000 | 0.000000 | 7.000000 | 10.000000 | 0.295880 | 0.292637 | 0.242220 | 0.362985 | 0.120765 | 20.000000 | 39.323621 | 50.293989 |
