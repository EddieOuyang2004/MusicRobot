# 第7章写作依据与跨章口径核对

核对日期：2026-09-08。正文沿用已有章节的英文 LaTeX，保留“工作总结、局限性、后续工作”三节。以下为写作辅助说明，不属于论文正文。新的 60 Hz 正式实验和 120 Hz v5 pilot 均已冻结；历史日志未改写。

## 采用的证据

- 正文：`chapter1_introduction.tex` 至 `chapter6_experimental_evaluation.tex`。
- 当前实现：音频／动作目录构建、SMPL／GMR 重定向、速度谷值关键姿态、分层检索与选择、异步准备、节拍控制、过渡与根连续性、最终 qpos 输出及实验记录路径；同时核对了离线拍手和实时机械臂子项目的开发背景。
- 当前目录：`realtime/humanoid_robot/data/music_catalog/catalog.json` 与 `catalog_features.npz`；重定向 manifest 及生成／准入代码均以 pipeline v4 为准。
- 主要实验依据：`docs/thesis/experiment_results/reanalysis_rescue_60_v5/consolidated_results.json`、同目录 `READY_FOR_THESIS`、`chapter6_numbers.tex` 和 `chapter6_tables/`。冻结汇总 SHA-256 为 `cb84e7c070c81570f9617bf605339c0cbe98a4fc0036f8b89e9e2e94ab242de4`。
- 新因果正式 cohort：`realtime/humanoid_robot/src/test/output/thesis_rescue_formal_60_v5/`；135 次正式运行包括130次短测试与5次长测试，另有1次 smoke。135/135 通过全部已评价门禁。
- 120 Hz 容量测试：`realtime/humanoid_robot/src/test/output/thesis_rescue_pilot_120_v5/`；21 次 pilot 中6次通过、15次失败，不与60 Hz正式结果合并。

## 用户列出的五项问题：当前结论

| 问题 | 核对后的口径及第7章处理 |
|---|---|
| DSP embedding 或 EffNet | 当前目录明确记录 `discogs-effnet-onnx`，不是纯 DSP embedding。采用同一 EffNet 模型的1280维 embedding、400维 style activations，并融合24维 DSP 节奏／音色特征。DSP-only 后端作为需另建目录的替代配置，不混入最终结论。 |
| 旧完整入口 smoke 没有收到匹配结果 | 2026-08-12 的报告是历史运行事实。后续已有完整运行、匹配及切换记录，应由第6章的新 cohort 支撑当前结论。新数据支持60 Hz最终输出合同，但不支持120 Hz可靠性。旧报告若继续公开展示，应另加“历史报告／已由后续实验补充”的说明和结果入口。 |
| pipeline v2/v3/v4 | 以生成脚本、准入检查和 `data/aistpp_gmr/manifest.json` 的 pipeline v4 为准。第7章明确写出版本；JSON 的 `schema_version: 2` 是另一套版本编号，不等于重定向 pipeline v2。 |
| 不同日期延迟不同 | 第6章已有冻结平台、CPUExecutionProvider、6秒窗口、1秒匹配间隔、60 Hz正式配置、120 Hz容量测试及新旧 cohort 分离。第7章不把历史微基准拼成趋势，也不把6秒上下文当成推理耗时。后续比较还应统一审计开关并平衡运行顺序。 |
| 相位结果仅三个动作 | 原来的 KR／LO／WA 单动作相位报告仍是三个个案。新实验已有更广的运行与 BAS 评估，但 BAS 与 authored-phase MAE 不是同一个指标，也不能证明已完成“每类型至少三个动作”的独立相位测试。第7章将该分层测试保留为后续工作，未声称已经补齐。 |

## 第7章的实证边界

- 工作总结按数据准备、音乐检索、动作选择、节拍控制、连续切换、实时实现六条主线展开；不把方法设计意图当成已证实效果。
- 保留留一音乐检索 Recall@1 = 0.317、Recall@5 = 0.783；类别差异举例为 HO = 1.000，JB／JS／KR = 0.000，每类仅6个音乐身份。
- 因果短测试 full/authored 的 harmonic BAS 为0.313／0.306，差值0.0067，95% CI [-0.0078, 0.0228]，Holm校正 p = 1.0000。因此不能声称 beat-sync 显著改善 BAS。该指标参考所有在线检测节拍的实际发送时刻，不是人工标注节拍，也不只挑选控制器接受的节拍。
- 不宣称多样性、检索层次或切入点评分的全部收益均通过显著性检验。第6章选出的 replay 消融中，相应差异未通过 Holm 校正；因果补测仅覆盖 full/authored timing。
- 60 Hz因果正式运行的速度、加速度、最终关节范围和6 mm clearance门禁为135/135通过；最大速度15.404 rad/s、最大加速度1673.415 rad/s²，残余间隙和最终关节越界均为零。该合同仍不等同于动力学或实机安全。
- 因果 full 短测试的每运行 work p99 中位数为4.657 ms；长测试为8.724 ms。所有正式run满足16.67 ms p99和0.01 miss-ratio门限。120 Hz pilot最差p99为15.572 ms、最差miss ratio为0.18544，仅6/21通过。
- 第7章将扩大到120 Hz的控制余量列为系统优化方向，再讨论拍号、学习式切入点、片段拼接、接触／平衡、G1实机和用户偏好。

## 仍需全论文统一、但不在本次正文中擅自改写的地方

1. **系统名称**：第2章主要使用 BeatWeaver，第1、3–6章主要使用 Humanoid Matcher。第7章暂跟随后者。正式合稿时应选定一个对外名称。
2. **研究问题编号**：第1章 RQ2–RQ5 分别侧重动作兼容、选择稳定性、相位控制、切换及实时性；第6章重新组织为节奏／多样性、最终输出、时延、消融。第7章按主题总结，避免继续引用冲突编号；合稿时应统一两章问题表述与映射。
3. **因果性表述**：第5章将双时间尺度路径统一描述为因果；当前文件输入默认仍会整曲预分析节拍，必须启用 `--experiment-causal-file-input` 才是本次补测协议。麦克风在线路径、默认文件回放与因果补测应分别表述。
4. **方法要求与实验结论**：第3、5章关于连续性和限制器的措辞有时接近保证。正式合稿时应明确这些是设计机制／目标；第6章仅验证60 Hz、头less、运动学输出合同，不扩张为动力学或实机保证。
5. **架构图与旧规划**：已有架构图仍含 DSP-current、旧融合权重等历史表述，`figure_plan_zh.md` 的“尚无第6章／完成标记”也是当时状态。它们不能作为当前实验版本的证据。

## 文件使用

- 正文：`chapter7_conclusions_future_work.tex`，可由主论文 `\input` 或 `\include`。
- 独立预览：`chapter7_preview.tex` 将章节计数设为6，因此正确显示 Chapter 7 与7.1–7.3。
- 编译在 `docs/thesis` 下执行 `../../tmp/thesis_tools/tectonic.exe chapter7_preview.tex`。正文未引入新文献或新实验，也不依赖第6章数字宏即可编译；若将来更新实验，应同步复核本说明和正文中的数字。
