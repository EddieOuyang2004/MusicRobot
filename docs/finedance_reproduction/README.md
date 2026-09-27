# FineDance 分割、音频配对与 GMR 数据复现

本目录冻结已发布 GMR 数据使用的分割记录。原始 FineDance 数据和音频不上传；
拿到相同原始数据后，可以恢复与 GMR 动作一一对应的音频，通常无需重新执行 GMR。

## 发布记录

- `segmentation_manifest.json`：原始完整分割清单的逐字节副本，含参数、排除原因、
  185 个来源录音的分段边界、2,530 个片段的 ID、标签和 train/val/test 划分。
- `clip_audio_map.csv`：每个片段的原始 WAV、60 FPS 起止帧、44.1 kHz 起止采样点、
  输出音频路径及 GMR 文件名。结束位置均为 **exclusive**。
- `checksums.json`：原始 WAV/动作、分割后的 WAV/SMPL、已发布 GMR 的 SHA-256，
  以及分割脚本 Git blob、文件校验值和本次复现验证的库版本。
- 完整分割为 2,530 段；GMR 成功 2,522 段，失败 8 段。没有 GMR 文件的片段在 CSV
  中 `gmr_artifact` 为空，在校验清单中 `gmr_sha256` 为 null；失败原因见
  [failures.json](../../realtime/humanoid_robot/data/finedance_gmr_v2/failures.json)。

清单中的原电脑绝对路径仅用于历史溯源；复现命令使用你自己的 `--input-root`。
原始清单不改写路径，以保留 GMR 元数据引用的
`f866692fa7c6557361035ff58c01a5b681940b562f5fdd506f93c6bb13e9b014` 校验值。

## 最快方式：下载已有 GMR，只恢复配对音频

在仓库根目录执行，先安装 Git LFS 并下载动作：

```powershell
git lfs install
git pull
git lfs pull --include="realtime/humanoid_robot/data/finedance_gmr_v2/*.pkl"
```

解压合法取得的 FineDance 原始数据，至少应有 `music_wav/001.wav` 等原始 WAV。
安装项目 Python 环境。以下命令使用 `.venv/Scripts/python.exe`；其他平台可换为相应
Python 可执行文件。本次逐字节验证使用 NumPy 1.26.4、SciPy 1.11.4、SoundFile 0.14.0，
记录在 `checksums.json` 的 `verification_environment` 中；它们不是 GMR 求解器的环境版本。

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/restore_finedance_audio.py `
  --input-root realtime/humanoid_robot/data/finedance `
  --output-root realtime/humanoid_robot/data/finedance_aistpp
```

默认只恢复 **2,522 个成功 GMR 片段**的 WAV，输出为
`finedance_aistpp/audio/<clip_id>.wav`，与 `finedance_gmr_v2/<clip_id>.pkl` 同名配对。
这一步不需要 SMPL 模型、GMR、GPU 或原始动作文件。每首完整录音只重采样一次再切片。
脚本验证原始 WAV 和重建 WAV 的 SHA-256；若来源不同或恢复结果不一致会报错。
已存在且一致的 WAV 可重复运行；不一致的文件不会被覆盖。

可先检查全部 2,530 个片段，不写出音频：

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/restore_finedance_audio.py --verify-only --all-clips
```

`--source-id 001` 可只处理一首录音（可重复指定）；`--all-clips` 包含八个 GMR 失败片段。
音频恢复命令只写 WAV，不生成完整 SMPL 数据集或声称已完成分割。若需要重建 GMR 或完整
检索目录的输入，请执行下一节的完整分割，并使用一个新的空目录。

## 完整重建分割后的 SMPL 与音频

原始目录应包含 `motion/<source_id>.npy`、`music_wav/<source_id>.wav` 和
`label_json/<source_id>.json`。使用发布时的明确参数；输出目录必须不存在或为空：

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/segment_finedance.py `
  --input-root realtime/humanoid_robot/data/finedance `
  --output-root realtime/humanoid_robot/data/finedance_aistpp `
  --source-fps 30 --min-seconds 8 --max-seconds 12 --target-seconds 10 `
  --root-y-offset 1.3 --max-duration-mismatch-seconds 0.1
```

不要加 `--allow-duration-mismatch`、自定义 `--split-json` 或改变目标时长。
默认 cross_genre 划分中 ignore 优先，val 为空；接受 185 首，分割 2,530 段，
其中 train 2,380、test 150。完整清单保留了 18 个排除来源及原因。
参数算法详见 [分割说明](../finedance_segmentation.md)。

运动从 30 FPS 重采样为 60 FPS，片段为 8–12 秒、目标 10 秒的连续平衡分段。
它不是简单每隔 10 秒裁一次，也不是根据节拍重新切分。音频从完整原 WAV 重采样至
44,100 Hz，单声道复制为双声道，以 PCM-16 保存。每个 60 FPS 帧对应 **735 个采样点**。
先切原音频再分别重采样会改变边界滤波结果，因此不可替代本脚本。

例：`finedance_001_0000586_0001172` 来自 `music_wav/001.wav`：
- 帧区间 `[586, 1172)`，时间 `[586/60, 1172/60)` 秒。
- 重采样后的采样区间 `[430710, 861420)`，共 430,710 个采样点。
- 对应 WAV：`audio/finedance_001_0000586_0001172.wav`。
- 对应 GMR：`finedance_001_0000586_0001172.pkl`。
- 音频时长是 `N/60`；最后一个动作姿态时间为 `(N-1)/60`，两者不能混淆。

跨电脑重建时，清单中的绝对路径可能不同，导致整个 JSON 的哈希不同；应比较片段 ID、
边界、划分和 WAV/SMPL 文件校验值，不应仅靠整个新清单的哈希判断失败。

## 确需重新求解 GMR 时

请按 [GMR 转换说明](../finedance_gmr_conversion.md) 配置 GMR 环境、
neutral SMPL 模型和 Unitree G1 资产。先以一个新输出目录验证一段，再执行全量任务：

```powershell
realtime/humanoid_robot/.venv-gmr/Scripts/python.exe `
  realtime/humanoid_robot/src/build_finedance_gmr_dataset.py `
  --input-root realtime/humanoid_robot/data/finedance_aistpp `
  --output-root realtime/humanoid_robot/data/finedance_gmr_rebuilt `
  --motion-id finedance_001_0000000_0000586 --jobs 1
```

全量重建时去掉 `--motion-id`，加 `--resume`。这是真正的约束优化和碰撞校验，
不能承诺像音频恢复一样快速。发布清单的 `v2_build` 保存 GMR commit、依赖版本、
模型哈希、投影参数与源 SMPL 哈希。跨平台/依赖变化可能使求解结果不同，不承诺重新求解
的 GMR 二进制逐字节相同。需要精确已发布动作时，使用 Git LFS 下载并验证 `gmr_sha256`。
更改路径或环境也可能使转换器缓存失效；不要为了复用缓存而修改校验字段。

## 本次实际验证（2026-09-27）

- 从 185 首原始 WAV 在内存中重建全部 2,530 段，所有 SHA-256 与原导出 WAV 一致，
  本机耗时 19.88 秒；落盘时间和其他电脑的耗时另计。
- 独立重跑录音 001 的完整分割，10 段 SMPL 和 WAV 均与发布参考逐字节一致。
- 验证向新目录恢复音频，以及既有分割测试。此次没有重新执行全量 GMR 求解。
