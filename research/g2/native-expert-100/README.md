# RoboTwin handover_block 原生专家 CPU 回放

本目录验收第三步的动态回放。Aloha-AgileX 与 ARX-X5 各使用自己的官方专家规划缓存、夹爪控制和原生碰撞资产。现有 [100 条数据处理报告](../../robotwin100_course/REPORT.md) 已核验 23,814 帧并筛查 47,628 个单臂目标。已有 [Aloha 原生回放](../../../docs/pace_g2_native_expert_replay.md) 提供历史证据。[ARX 历史重定向回放](../../../docs/pace_g2_arx_expert_replay.md) 的动作来自 Aloha，未进入本台账。

## 输入与执行约定

- 数据集 [固定 revision 3dc3b79](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/3dc3b798668feb99ac61cc9086d84cbcc3d79186/dataset/handover_block)。`inputs/` 记录各自 ZIP 成员、CRC32、SHA-256 和原始 seed；HDF5 复用已有下载及哈希。碰撞与 URDF 来自固定版本的 [embodiments.zip](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/blob/3dc3b798668feb99ac61cc9086d84cbcc3d79186/embodiments.zip) 及仓库已有 CPU 环境。固定 [官方实现 ea8b211](https://github.com/RoboTwin-Platform/RoboTwin/tree/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f) 用于恢复原始控制、初始化和任务判据。数据集未记录当时的生成源码提交。
- 规划 PKL 恢复双臂位置和速度。原任务阶段顺序恢复双臂同步、夹爪插值及保持。对每个 HDF5 保存点核验 14 维驱动指令；保存回调可能在同一物理步重复触发。物理步由原缓存和源代码的调度规则推导。每步骤采集仿真实测关节、指驱动、末端、物块以及接触冲量。
- CPU 执行器沿用原尺寸、关节驱动、夹爪范围、URDF 和碰撞网格，4 ms/步，10 g 物块，摩擦系数 0.5，恢复系数 0。初始化 2,801 步单独记录。跳过渲染和在线规划，执行驱动过程中由 PhysX 更新实测 qpos。原文件未提供物块 3D 初态快照，依据每个 seed 和相机初始化的随机数消耗重建；`scene-audit.json` 用原始首帧 RGB 的投影边界交叉核对。
- 以**原轨迹最后一个物理步**计算原版 `handover_block.check_success`。物块和托盘功能点的 x、y 差分别小于 30 mm，z 差小于 10 mm，右夹爪归一化请求大于 0.8。留存两指实测位置以独立核对张开过程。复用的 Aloha 历史回执包含诊断保持，任务结果仍取当时的专家序列末尾。
- 新回放日志的接触列包括掩码、物块最大接触深度（m）和四指冲量（N·s）。掩码位 0/1 是左手双指，2/3 是右手双指，4 是物块与托盘。台账同时提供双指抓取并抬升至少 20 mm、双方同时双指、右手双指接管和托盘接触的首次 tick，以及更宽的任一手指接触指标。两条复用的 Aloha 日志按手保存每步冲量，接触对名称只在变化时记录。稀疏事件无法证实的精细里程碑留空；手级指标和原任务判据可独立复算。

## 复现

在项目根目录运行，使用已有 RoboTwin CPU 环境（SAPIEN 3.0.0b1、NumPy、h5py、PyYAML 和 Pillow）。`RAW` 指向已有的 `{aloha-agilex,arx-x5}/episodeN.hdf5`。先设置操作目录和资产路径。

```bash
RAW=/path/to/existing/robotwin100/raw
RUN=/path/to/new/output
ASSETS=/path/to/RoboTwin/assets
SOURCE=/path/to/pinned-source
AUDIT=research/robotwin100_course/sources/embodiment_config_audit.json
mkdir -p "$RUN" "$SOURCE"
export PYTHONPATH="$PWD/scripts"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python - "$SOURCE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

root = Path(sys.argv[1])
manifest = Path("research/g2/expert-source-20261105/manifest.json")
for record in json.loads(manifest.read_text())["files"]:
    if "sha256" not in record:
        continue
    dest = root / record["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(record["url"], timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(record["path"])
        dest.write_bytes(data)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(dest)
(root / "manifest.json").write_bytes(manifest.read_bytes())
PY

for robot in aloha-agilex arx-x5; do
  python scripts/g2_native_expert_inputs.py \
    research/robotwin/sample_manifest.json "$RUN/cache" "$robot"
done
python scripts/g2_native_expert_batch.py prepare "$RAW" "$RUN/cache" "$RUN/prepared" \
  --manifest research/robotwin100_course/data/download_manifest.json \
             research/robotwin100_course/data/download_manifest-arx.json
python scripts/g2_native_expert_scene_audit.py "$RAW" "$RUN/prepared" "$RUN/scene-audit.json"

# 每个 episode 独立子进程。中断后，使用相同输出目录重复命令即可续跑。
python scripts/g2_native_expert_batch.py batch "$RUN/prepared" "$RUN/results" \
  "$ASSETS" "$AUDIT" "$SOURCE" --episodes 0 --timeout 240
python scripts/g2_native_expert_batch.py batch "$RUN/prepared" "$RUN/results" \
  "$ASSETS" "$AUDIT" "$SOURCE" --timeout 240
python scripts/g2_verify_native_expert_100.py "$RUN/results" "$RUN/expert_replay_100_results.csv" \
  --prepared "$RUN/prepared" --manifests "$RUN/cache" --source-manifest "$SOURCE/manifest.json"
```

`batch` 保留已落盘的终态回执和中断尝试。恢复同一目录前先结束该目录正在运行的批次。针对基础设施故障另开输出目录，使用 `--robots` 和 `--episodes` 指定受影响的样本。保留旧故障回执和新尝试。每条执行进程受独立超时约束。单条失败在台账保留具体分类及诊断。

复算已归档的正式台账只需 Python 和 NumPy。另传入 `--prepared "$RUN/prepared"` 可独立逐值核对所有原生关节驱动、夹爪请求及帧时刻。

```bash
E=research/g2/native-expert-100
PYTHONPATH=scripts python scripts/g2_verify_native_expert_100.py \
  "$E/results" "$E/expert_replay_100_results.csv" --check-csv --manifests "$E/inputs" \
  --source-manifest research/g2/expert-source-20261105/manifest.json \
  --evidence-manifest "$E/evidence-hashes.json" \
  --hdf5-manifests research/robotwin100_course/data/download_manifest.json \
                   research/robotwin100_course/data/download_manifest-arx.json
```
