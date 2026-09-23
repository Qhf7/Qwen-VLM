# 跨版本评测运行目录

评测入口：`e1/evaluate_versions_local.py`。本目录使用 fold1 全量 3600-view dev，配置为 `mean_w0.5`，并同时累计 `gate=0.5` 与 `gate=0`。完整运行会在每个版本目录写入指标、逐图记录、直方图、切片表和选择记录，最终写入 `summary.csv/json`。

本机 MPS 已用 2 张图 smoke test 验证；完整 3600-view 任务已切到训练服务器 CUDA 后台运行，避免在本机逐张调用 2B 视觉基座耗时过长。服务器端日志位于 `/tmp/v1.log`…`/tmp/v5.log`，完成后可将本目录同步回本机：

```bash
scp -P 34855 -r root@connect.westc.seetacloud.com:/root/autodl-tmp/vlm/artifacts/version_audit_20260923 ./artifacts/
```
