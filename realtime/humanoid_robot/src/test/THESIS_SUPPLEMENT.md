# Thesis 补测运行说明（measurement schema v2）

本次只补测因果实时与最终输出记录。现有 `output/thesis_final/` 的 1,111 次运行、特征包和 `docs/thesis/experiment_results/READY_FOR_THESIS` 不改写。

## 你现在运行什么

旧的 `output/thesis_supplement/` 是冻结的 120 Hz 基线，不要用新代码对它执行
`-Resume`。修复后的双频率验证必须写入独立目录。

第一轮 pilot 已完整保留并完成离线分析：120 Hz 为 7/21 runs 全门禁通过，60 Hz
为 18/21；两者的实时工作预算和最终 clearance 均通过，失败来自输出速度/加速度。
复盘确认 0.5 mm 内部 clearance 余量会把仍满足 6.0 mm 合同的当前姿态误当成
不可用 anchor，继而跳回陈旧姿态。该回退已修复，并为 2000 rad/s² 验收上限设置
1600 rad/s² 内部限制以吸收实际发送时刻抖动。旧 pilot 不删除、不选择性重测。

第二轮 pilot 也已完整保留。关节导数尖峰已经消失，但 120 Hz 为 12/21、60 Hz
为 5/21 全门禁通过：失败主要来自运行期间的系统负载抖动（相同 collision 路径的
p99 从约 4.14 ms 漂移到 35.96 ms），另有一个 120 Hz run 的 6 帧硬 clearance
残留。最终 fallback 现已改为逐候选重新验证并拒绝陈旧的不安全缓存。修复后对同一
silence-recovery 配置的定向复测结果为：120 Hz work p99 5.82 ms、miss ratio
0.00139、残留 0；60 Hz work p99 6.36 ms、miss ratio 0、残留 0。完整回归
186/186 通过。

第三轮 pilot 已完整保留。两个频率的最终 clearance 均为 0，但两个频率 21/21
runs 都未通过实时门禁。核查确认两个 cohort 没有重叠执行；活动电源方案却仍是
Windows“平衡”（GUID `381b4222-f694-41f0-9685-ff5bb260df2e`），而机器已有
“高性能”方案。相同代码、输入和 seed 在独立定向运行中明显更快，因此 v3 作为环境
不合格的压力观测保留，不能作为通过结果。实验入口现会在每次 run 前验证活动电源
方案，并把它写入冻结 manifest；错误方案会在运行前终止，不再浪费完整 cohort。

第四轮 60 Hz pilot 已完整保留并完成离线分析：17/21 runs 全门禁通过；21/21
runs 的实时门禁、最终 clearance、关节限位、输出速度和加速度全部通过。唯一失败项是
4 个 runs 各出现 1 次 `hold_last_events`。根因是动作播放到末尾时，强制切换只搜索
最新检索排名，没有使用已经完整准备好的缓存动作。强制切换现会在最新排名没有可用
动作时选择不同于当前动作的安全缓存候选；原失败配置的 60 秒定向复测中
`hold_last_events=0`、work p99 13.24 ms、miss ratio 0.00465，最终安全残留为 0。
完整回归 188/188 通过。由于该修复会改变控制输出，v4 不能与修复后的 runs 混用，
需要一个全新的 v5 cohort。

第五轮 60 Hz pilot 已完整保留并通过离线门禁：21/21 runs 全部通过，且
`hold_last_events` 总数为 0。跨全部短跑和长跑，最差 control-work p99 为
12.512 ms（预算 16.667 ms），最差 deadline-miss ratio 为 0.003096（门限
0.01），最大最终速度为 15.410 rad/s，最大最终加速度为 1544.018 rad/s²；最终
clearance 和关节限位违规均为 0。活动电源方案已由冻结 manifest 确认为“高性能”。

第五轮 120 Hz pilot 也已完整保留并完成离线分析，但只有 6/21 runs 全门禁通过。
`hold_last_events` 已在 21/21 runs 中归零，最终 clearance 和关节限位违规也全部为
0；主要失败来自 8.333 ms 预算：最差 control-work p99 为 15.572 ms，最差
deadline-miss ratio 为 0.18544。前 15 个短跑的 limiter/collision 阶段 p99 为
7.78--12.04 ms，单阶段已接近或超过整个周期预算。另有 1 个短跑的最终加速度为
2046.42 rad/s²，略超 2000 rad/s² 门限。后三个短跑与三个长跑在机器状态稳定后
通过，但该结果对系统状态的依赖过强，不能支持可靠 120 Hz 的结论。120 Hz v5 因此
作为容量/压力测试的失败证据保留，不再通过选择性重跑追求通过；论文主运行频率改为
已经 21/21 通过的 60 Hz。

60 Hz v5 正式 cohort 和完整离线分析现已完成，不需要再跑实验。1 次 smoke、130
次短跑和 5 次长跑均一次完成，无失败或重试；135/135 个正式 runs 通过全部已评价
门禁。跨所有正式 runs，最差 control-work p99 为 14.483 ms，最差
deadline-miss ratio 为 0.006929，最大最终速度为 15.404 rad/s，最大最终加速度为
1673.415 rad/s²；`hold_last_events`、ready-pool underrun、最终 clearance 和最终
关节限位违规总数均为 0。冻结结果位于
`docs/thesis/experiment_results/reanalysis_rescue_60_v5/`，其中
`READY_FOR_THESIS` 表示完整性与分析完成；不要对正式目录执行选择性重跑，也不要再
启动 120 Hz 正式 cohort。

## 修复后的开发级证据（不是论文统计结果）

2026-09-07 在同一台 CPU-only 机器上，以 stitched 输入、2 秒计时预热和 10 秒
因果回放完成了两次冒烟验证：

| 频率 | control-work p99 | deadline miss | 输出间隔 p99 | 最终碰撞残留 | 最大速度 / 加速度 |
|---:|---:|---:|---:|---:|---:|
| 120 Hz | 4.09 ms（预算 8.33 ms） | 0/962 | 10.22 ms | 0 | 14.71 rad/s / 1815.33 rad/s² |
| 60 Hz | 4.88 ms（预算 16.67 ms） | 0/481 | 18.71 ms | 0 | 15.33 rad/s / 1622.62 rad/s² |

这只证明代码已达到“值得重跑 pilot”的门槛，不能替代多 seed、长跑或正式 cohort。
修复包括：按真实输出间隔执行动力学限制、不允许 deadline miss 后立即补发；对完整
floating-base pose 做碰撞投影并最终复核；检索和动作 grounding/entry-feature 准备使用
低优先级独立进程；因果文件音频按墙钟异步输入；扩大动作缓存并限制 BLAS/OpenMP
线程。当前环境没有 CUDA ONNX provider，因此没有把 GPU 作为本次修复的依赖。
Windows 的非实时唤醒仍造成输出周期抖动；短冒烟中的 120 Hz 通过不能覆盖多 run
行为，已被 v5 pilot 的结果否定。论文只能把 120 Hz 表述为原始目标和未通过的容量
测试，不能表述为可靠或硬实时 8.33 ms 周期保证。

以下命令仅用于复现原始补测流程，不应在代码修改后指向旧结果目录。

在仓库根目录打开 PowerShell，执行：

```powershell
cd C:\Users\Eddie\Desktop\MusicRobot
powershell -ExecutionPolicy Bypass -File `
  realtime/humanoid_robot/src/test/run_thesis_experiments.ps1 `
  -Stage supplement -Resume
```

脚本先检查 CPUExecutionProvider、输入文件与版本哈希、至少 10 GB 空闲空间，然后串行执行：

| 阶段 | 数量 | 每次输入评价长度 |
|---|---:|---:|
| 开发 smoke | 1 | 10 秒 |
| Full / authored timing × 13 输入 × seeds 0–4 | 130 | 40 秒（前 6 秒预热） |
| Full × stitched × seeds 0–4 | 5 | 606 秒（后约 600 秒评价） |

13 个输入为 stitched、全部 10 首 CC0、tempo-jump、silence-recovery。短输入不足时确定性循环扩展；长跑循环 stitched。输入循环边界会作为突变事件枚举。40/606 秒是输入音频时轴长度，另有启动校准、加载、结束写盘开销；因果模式不提前馈送音频。

60 Hz 正式配置的评价时长合计约 2.29 小时，含进程预热、数据准备和分析开销建议
预留 3–4 小时（实际依机器负载而变）。接通电源、暂时关闭自动睡眠及高负载程序，
只启动一个入口。请保持代码、Python 依赖、模型和数据不变；不要同时运行旧
`-Stage all`、120 Hz 正式实验或其他高负载任务。

## 中断、错误与完成

- 中断后重复同一条命令。仅复用状态完整、命令一致、版本一致且四类产物校验通过的运行。
- 每次运行最多自动尝试两次；失败产物保留到该 suite 的 `runs/failed_attempts/`。仍失败则停止，避免后续运行被无效测量污染。`run_status.json` 保留尝试历史，不能将历史状态条目当成互相独立的失败次数。
- 若出现版本不一致错误，不要删除 manifest 或改旧结果。将错误告诉我；恢复原版本或另建 cohort 才能继续。
- `runner.lock` 是 OS 文件锁，退出或崩溃后自动释放。不要通过删除它来并发运行。
- 即使延迟、安全指标未达标，完整且有效的运行也会保留，**不会为追求通过而反复重测**。smoke 检查的是测量完整性，不是性能门槛；静音片段没有节拍也不算产物损坏。
- 脚本全部成功后在所选输出目录写入 `SUPPLEMENT_COMPLETE`。此标记仅表示 smoke + 135 次记录完整，不表示所有验收约束通过。

若终端报错，保留整个输出目录，把最后一段错误发给我。全部结束后告诉我“补测跑完了”即可，无需上传共享目录里的文件。

## 新文件位置

- 正式补测：`realtime/humanoid_robot/src/test/output/thesis_rescue_formal_60_v5/`
- 固定版本与全部输入哈希：上述正式目录的 `manifest.json`
- 分阶段状态：正式目录下的 `smoke/run_status.json`、`short/run_status.json`、`long/run_status.json`
- 每次运行：trace CSV、timing JSON、pose NPZ、log；完成状态另保存这四类文件的 SHA-256。
- 最终离线汇总：`docs/thesis/experiment_results/reanalysis_rescue_60_v5/`

## 测量解释

- `--experiment-causal-file-input` 为可选实验开关，默认文件回放仍使用旧行为。新模式仅向在线分析器馈送已到达墙钟位置的音频，不运行整曲节拍预分析。
- NPZ schema 见 `thesis_supplement.schema.json`。墙钟相对“音频开始（扣除启动静音）”记录，可在校准期为负值，必须逐帧严格递增。`time_seconds` 是已接收音频位置，允许重复，不得用于实际关节导数。
- 记录最终 MuJoCo qpos、关节顺序／qpos 地址、XML 哈希，以及检测、修正、最终残留 clearance 违规、最终限位四个独立计数。检查针对既有几何体配对和 clearance 阈值，不能扩张为全机器人所有接触的闭环安全保证。
- 最终安全审计带来额外 `mj_forward` 等开销，记录于 `experiment_final_safety_audit`，也计入总控制耗时。
- 节拍同时保存对应时间、真实发送时间、音频接收位置和 controller 是否接受。新主 BAS 使用**所有在线检测节拍的真实发送时间**，不是回填时间，也不只选择被控制器接受的节拍。没有音乐节拍时 BAS 为缺失值，不人为记 0。
- 输出速度、加速度、jerk 使用实际墙钟差分；旧记录无法恢复真实时间，保留为名义采样率诊断并明确标注。
- [EDGE 官方 PFC](https://github.com/Stanford-TML/EDGE/blob/main/eval/eval_pfc.py) 使用截断向下分量的根加速度，经片段最大值归一化，再乘左右足的水平位移。旧公式仅标为 `pfc_legacy_proxy`。新 `pfc_edge_g1_adapted_30fps` 在 30 FPS 网格采用相同代数形式，但 G1 每侧只有一个足部锚点，而非 SMPL 踝／趾最小值；不能直接与官方数字比较。

## 已完成的离线步骤

离线阶段没有启动 matcher 实验。它核验了冻结的 60 Hz 正式 cohort，然后重算指标、
按音源跨 seed 聚合、bootstrap CI、配对 permutation test 及 suite 内 Holm 修正，
重建插值后的 SMPL 特征，并按每 seed 固定 40 项分别计算 FID/Div（每项 Div 使用全部
780 对）。同时报告 raw 和 ground-truth 标准化特征尺度；120 Hz v5 pilot 作为独立
容量测试引用，不混入正式 60 Hz cohort。保留短动作循环、低频 trace 边界和检索
真实动作复用的限制。

新版汇总、完成标记、Chapter 6--7、表格和图均已更新。论文中旧预分析回放与新因果
验证分开，120 Hz 未达标及 BAS 不显著结果如实报告。

仅查看计划、不跑实验可使用 `-Stage supplement -Resume -DryRun` 或 `-Stage reanalyse -DryRun`。这两个 dry-run 不生成完成标记，也不初始化实时 matcher。
