# MMPFN 一键评测

统一入口同时支持 VT-Bench 和 MulTaBench。每个数据集通过注册表描述，新增数据集不需要再复制一份 runner。

## 当前注册的数据集

- VT-Bench：11 个判别任务数据集。
- MulTaBench：论文表格中 `Text=0, Img=1` 的 11 个 image-tabular 数据集。

查看完整清单：

```bash
python -m mmpfn.run_benchmark_suite --list-datasets
```

## 先做一次小型烟雾测试

Mango Mass 只有 546 条，适合先验证整条链路：

```bash
cd /mnt/hdd/zhangyg/projects/tab/MultiModalPFN-main
/home/debian/miniconda/envs/tabopen/bin/python -u -m mmpfn.run_benchmark_suite \
  --datasets mt_mango_mass \
  --modes full image_only tabular_only \
  --folds 0 \
  --gpu 1 \
  --data-root /mnt/hdd/zhangyg/projects/tab/benchmark_data \
  --output-dir /mnt/hdd/zhangyg/projects/tab/results/mmpfn_benchmark_suite \
  --dino-checkpoint /mnt/hdd/zhangyg/projects/tab/models/dinov2_vitb14_pretrain.pth \
  --download-multabench
```

## 一键运行 11 个 MulTaBench 数据集

```bash
bash run_mmpfn_benchmarks.sh
```

默认跑 fold 0 和三种模式。重新执行同一命令会跳过已有 `metrics.json` 的实验，从中断处继续。正式复现五个 fold 时直接使用 Python 入口并传入：

```bash
--folds 0 1 2 3 4
```

也可以直接选择数据集和模式：

```bash
# 一个数据集，只跑 Full
bash run_mmpfn_benchmarks.sh --datasets mt_mango_mass --modes full

# 两个数据集，只跑两个单模态
bash run_mmpfn_benchmarks.sh \
  --datasets mt_mango_mass mt_cbis_ddsm \
  --modes image_only tabular_only

# 一个数据集跑三种模式（省略 --modes 时默认就是三种）
bash run_mmpfn_benchmarks.sh --datasets mt_mango_mass
```

MulTaBench 按论文协议默认将每个 fold 的训练 context 限制为最多 10,000 条；显式传入 `--max-train-context 0` 才会强制全量训练。

## 运行已有 VT-Bench 导出

已有旧格式导出需要给出其真实目录，例如：

```bash
/home/debian/miniconda/envs/tabopen/bin/python -u -m mmpfn.run_benchmark_suite \
  --datasets vt_pneumonia \
  --dataset-root vt_pneumonia=/mnt/hdd/zhangyg/projects/tab/raw/mimic/pneumonia/features \
  --modes full image_only tabular_only \
  --folds 0 --gpu 1 \
  --data-root /mnt/hdd/zhangyg/projects/tab/benchmark_data \
  --output-dir /mnt/hdd/zhangyg/projects/tab/results/mmpfn_benchmark_suite \
  --dino-checkpoint /mnt/hdd/zhangyg/projects/tab/models/dinov2_vitb14_pretrain.pth
```

没有旧格式 adapter 的 VT-Bench 数据集可转换为统一格式：一个目录内包含 `metadata.json` 和 `train.npz`、`val.npz`、`test.npz`。每个 NPZ 包含对齐的 `x`、`y`、`image_paths`。

## 输出与监控

- 单次日志：`results/mmpfn_benchmark_suite/logs/<dataset>/fold_<n>/<mode>_seed_<n>.log`
- 单次结果：`results/mmpfn_benchmark_suite/<dataset>/fold_<n>/<mode>/seed_<n>/metrics.json`
- 性能总表：`results/mmpfn_benchmark_suite/summary/results.csv`
- 负迁移表：`results/mmpfn_benchmark_suite/summary/negative_transfer.csv`
- 失败清单：`results/mmpfn_benchmark_suite/summary/failures.json`

实时看当前日志示例：

```bash
tail -n 30 -f /mnt/hdd/zhangyg/projects/tab/results/mmpfn_benchmark_suite/logs/mt_mango_mass/fold_0/full_seed_42.log
```

所有下载、缓存、预处理和结果默认均位于 `/mnt/hdd/zhangyg/projects/tab` 下。Kaggle 认证只从现有环境读取，不会把 token 写入项目。
