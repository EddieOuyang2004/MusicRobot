# 论文插图与图表规划

审阅日期：2026-09-05。依据当前工作区的第 1–5 章、参考文献、现有架构图，以及项目的数据构建、检索、控制、重定向、诊断和实验代码。本文是插图规划，尚未修改论文正文或制作最终插图；工作区仍在更新，正式制图时应再次核对配置与数据版本。以下用 LaTeX section label 定位，避免正文修改后行号失效。

最有价值的补充，是让读者看懂三条关系：**人体表示如何成为机器人轨迹；音乐证据如何成为切换决定；节拍如何通过连续时间调整落在动作关键姿态上。** 建议先完成下文的 12 张核心图，再根据篇幅选择扩展图。文献原图集中选择少数真正解释机制的例子，方法章节主要使用自己的代码和数据制图。

## 1. 当前状态与需要先统一的内容

- 论文目录目前有第 1–5 章正文，第 6–7 章仅在章节安排中提及。第 6 章建议在本文中属于预留规划。
- 目前第 1、2、4、5 章没有 figure/table 环境；第 3 章有一个由 `fbox/parbox` 组成的文字框架构图。两套已有 `.drawio/.svg/.png` 架构图还没有通过 `includegraphics` 接入论文。
- 项目历史包括冻结的离线拍手轨迹、实时机械臂和当前实时人形机器人。论文主体是最后一条；机械臂历史可放附录或一段开发背景，无须占用核心方法图篇幅。
- 当前目录实测为 **348 个音频片段、60 个音乐 ID、411 个动作**，数组形状分别为 `348×1280`、`348×24`、`348×400`。音频编码后端为 `discogs-effnet-onnx`。
- `thesis_final` 下已有正式实验子目录及部分 SVG 输出，但本次检查时 `docs/thesis/experiment_results/READY_FOR_THESIS` 不存在。可以设计实验图及使用明确标注的个案，最终统计结论应以通过汇总检查的结果为依据。

### 已有架构图值得复用，但需修订

文件：[英文模型图](C:/Users/Eddie/Desktop/MusicRobot/docs/thesis/figures/humanoid_matcher_model_english.png)、[中文架构图](C:/Users/Eddie/Desktop/MusicRobot/docs/thesis/figures/humanoid_matcher_architecture.png)。两者均有同名 drawio 编辑源。

| 当前图中的问题 | 建议修订 |
|---|---|
| 写着 `DSP embedding (current)`、EffNet optional | 当前冻结目录应写 Discogs EffNet：1280-D embedding 和 400-D style activations；DSP embedding 作为需另建目录的替代方案 |
| 最终匹配公式为 `.55 music + .45 motion` | 当前代码和第 5 章均为 `.65 music + .35 motion` |
| 相似度直接写 cosine | 方法细图应与代码一致，说明标准化、单位化及 `(1+cos)/2` 映射；总架构图可不写公式 |
| 缺少 genre-first restriction 和 abstention | 在分层检索细图中补充风格限制、弱节拍/非舞蹈音频拒绝分支 |
| 英文图把实时事件流画在六秒查询描述符之后 | 从音频输入处分成短块事件通道与滚动窗口检索通道，避免暗示节拍检测必须等待检索完成 |
| `Motion Compatibility Heads` 容易被看作训练出来的网络头 | 改为 `Compatibility Scores`，注明 deterministic scoring |
| 碰撞检查画成无条件输出步骤 | 标注 optional；CLI 默认 off，正式实验协议使用 always，图注说明采用哪一配置 |
| 图中有普通四拍小节措辞 | 注明是每四个 accepted alignment beats 的逻辑边界，不是经过 downbeat/meter 识别的真实乐理小节 |
| 大横图文字、公式和跨栏连线过密 | 第 3 章只保留模块与数据流，公式拆到第 4–5 章，缩到实际论文宽度后检查字体 |

此外，第 2 章主要使用 BeatWeaver，其他章节多用 Humanoid Matcher。先统一对外系统名，代码名称可保留为实现名称。图中输出使用 `G1 kinematic pose playback`；当前 `mj_forward` 路径没有力矩闭环和平衡控制。

## 2. 十二张核心图：插入位置、内容和制作办法

### C01 — SMPL 人体表示与机器人表示

**位置：** 第 2 章 `subsec:lit_motion_representations`，首次介绍 SMPL 后。建议图题：*Human and robot motion representations*。

**构图：** 三联图：(a) 中性 SMPL 网格及半透明骨架；(b) 同一模型的一个舞蹈姿态，标出 pelvis、左右肩/肘/腕、髋/膝/足；(c) 对应 G1 模型。SMPL 标出 24 关节、6890 顶点、pose 与 shape 的概念；G1 标出浮动根和 29 个驱动关节。不同身体的对应部位使用同色，不把 24 和 29 画成一对一连线。

**解决的问题：** 审核人能立即理解，SMPL 是有表面的参数化人体模型，而人类姿态参数不能直接当机器人关节指令。

**制作：** 优先用本地 SMPL_NEUTRAL 和 AIST++ 姿态自渲染。AITViewer 适合快速显示 SMPL、骨架和导出图片；如需统一材质、正交相机和高质量排版，可用 Blender。G1 使用本项目 MuJoCo XML 渲染。最后用矢量编辑工具组合、标注。[SMPL 官方资源](https://smpl.is.tue.mpg.de/)提供模型与 Blender 相关工具入口；[AITViewer 官方文档](https://eth-ait.github.io/aitviewer/)说明了 SMPL 支持与图像导出功能。

**注意：** 基础概念图可以说明 shape 变化，但本项目采用固定中性体型，不能画成在线估计或调整体型。Python 包名 `smplx`、GMR 配置名 `smplx_to_g1` 不意味着输入数据变成了 SMPL-X。根位姿另有 3 维平移与 4 维四元数，不能把四元数的四个数说成四个旋转自由度。

### C02 — 文献路线对照图

**位置：** 第 2 章 `sec:lit_synthesis`，研究空缺总结前。建议图题：*Design alternatives for music-driven motion*。

**构图：** 四条独立横向流程，统一符号和配色：音乐条件的人体舞蹈生成；因果流式人体运动生成；直接音频条件机器人策略；本文的音频检索、机器人动作库和连续播放。分别标注输入上下文、输出表示、主要在线计算以及是否需要独立 retargeting。离线训练/准备放在浅灰背景，在线路径放在彩色背景。

**解决的问题：** 解释本文在已有研究中的位置。不要画成所有离线模型都经过同一处理流程，也不要给全部生成模型统一贴上“非实时”标签。

**制作：** diagrams.net 自绘概念综合图，各路线引用相应论文。概念图下方明确这是作者归纳，而不是任何一篇论文的原始架构。

**配套表：** 方法行可包括 FACT、Bailando、EDGE、Beat-It、DiscoForcing、RoboPerform 和本文；列为生成/检索方式、输入上下文、人体/机器人输出、节拍或关键帧控制、运动学/动力学/实机证据、与本文的关系。非同一协议下的延迟数值不要直接拼成性能排行榜。

### C03 — Beat-It 的节拍与关键姿态示意

**位置：** 第 2 章 `subsec:lit_diffusion` 或 `subsec:lit_beat_keypose`，只放一次并交叉引用。建议图题：*Explicit beat and key-pose conditioning in Beat-It*。

**选图：** 优先考察论文 **Fig. 1**。该图把输入节拍、关键姿态、生成动作及运动速度放到同一示意中，与本文的关键姿态时间对齐最相关。[ECCV 原论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02909.pdf)、[作者项目页](https://zikaihuangscut.github.io/Beat-It/)。

**解决的问题：** 比单纯的舞蹈效果截图更容易解释“动作在节拍时刻发生”具体指什么。

**制作/获取：** 按可用授权引用原图，或依据概念重新绘制一个简化示意并引用。正文须说明 Beat-It 用条件生成动作，本文用已存轨迹的速度和相位调整实现对齐；低速谷值只是本文的关键姿态候选，不自动等于所有语义舞蹈重拍。

### C04 — 端到端系统架构

**位置：** 替换第 3 章 `sec:design_architecture` 的现有文字框图，保留现有 `fig:humanoid_matcher_architecture` label。

**构图：** 上方离线双分支：audio → features；SMPL → key-pose profiles 与 GMR → preflight → frozen catalogue。下方在线双分支：rolling query → retrieval → stable selection → preparation；short audio blocks → beat events → phase controller。汇合到 entry search、root-aligned blend、output conditioning、G1 playback。音频库、动作库用不同数据存储符号。

**解决的问题：** 一张图交代模块职责、离线/在线边界与数据依赖。

**制作：** 从已有 `.drawio` 结构重排，去除公式、长函数名和全部默认阈值。建议竖向或紧凑两层布局。不同箭头分别表示数据、节拍事件、候选准备完成；图例说明可选碰撞处理。

### C05 — 多时间尺度与异步执行时序

**位置：** 第 3 章 `sec:design_state_ownership`，第 5 章 `subsec:ch5_dual_timescale` 引用。建议图题：*Causal audio processing and asynchronous motion preparation*。

**构图：** 同一时间轴上排列 audio blocks、rolling window、retrieval worker、preparation pool、foreground playback 五条泳道。标出 `[t−6,t]` 查询窗口、约一秒一次的提交机会、busy 时跳过、完成结果被消费、候选 ready，以及边界提交。正式控制配置标 60 Hz；120 Hz 仅作为未通过的容量测试另行标注。

**解决的问题：** “实时”不是每个模块都在 60 Hz 运行；控制线程不能等编码、加载或接地完成。六秒历史窗口也不是六秒的算法计算耗时。

**制作：** diagrams.net 或 TikZ 自绘。第 3 章用概念时间轴，若使用实际 trace 定位时刻，则把它作为实测时序个案并记录 run ID。不要把不同通道的事件先后关系画成串行串接。

### C06 — 离线音频窗口与三种表示

**位置：** 第 4 章 `subsec:audio_array_storage` 后，覆盖 audio_windowing、learned_audio_representation、audio_dsp_features 三节。

**构图：** (a) 波形上方画 6 s 窗、2 s hop；(b) robust normalisation 后分到 EffNet 与 DSP；(c) EffNet 同时输出 1280-D embedding、400-D style activations，DSP 输出 24-D rhythm–timbre；(d) offline mean/std 的保存及 online 复用关系。Style activations 单独画 unit normalisation，避免误画成同样 z-score。

**解决的问题：** 把分散在文字里的窗口、维数和处理规则联系起来。

**制作：** 波形与窗位置由真实 WAV 用 Python/Matplotlib 输出，流程部分 diagrams.net 绘制。图内只保留维数和主要操作，具体 MFCC/contrast 维数拆分留在表格。网络结构本身不是本文贡献，无需重画完整 EfficientNet。

### C07 — 速度谷值、关键姿态与实际姿势

**位置：** 第 4 章 `subsec:keypose_detection` 后。建议图题：*Velocity-valley proposals for salient motion poses*。

**构图：** 同一时间轴三层：原始/平滑速度；归一化曲线及被保留的谷值；与 4–6 个所选谷值对应的 SMPL 姿态小图。标示 prominence、最小间隔和边界排除的一个例子。可叠加参考音乐节拍，但须说明两者不必逐点重合。

**制作：** 调用 `aistpp_velocity_keypoints.py` 的 `detect_aistpp_file`，直接取得 `raw_velocity`、`smoothed_velocity`、`frame_indices` 和 scores；再按真实帧渲染 SMPL，Matplotlib 合成。

**现成工具的限制：** `test/visualize_motion_keypoints.py` 调用的是通用 `motion_keypoints` salience 路径，不能直接把它的旧 HTML 称为当前 AIST++ velocity-valley 结果。真正的谷值图应从 `aistpp_velocity_keypoints.py` 生成。

**需核对的图文问题：** 第 4 章目前将角速度公式中的旋转定义为 world rotation；实际 `smpl_angular_velocity` 直接将原始局部 axis-angle 转成旋转矩阵，没有先做世界旋转 FK。制图前应统一文字定义与实际实现。纵轴也应写 equivalent motion speed 或 normalised velocity，不能将近似等效速度表述成精确测量的全身质心速度。

### C08 — 同帧人体到 G1 的重定向对照

**位置：** 第 4 章 `sec:offline_retargeting`，FK 和 GMR 说明附近。建议图题：*AIST++ motion retargeting to Unitree G1*。

**构图：** 三行、四列时间采样。行分别为 source video（有对应视频时）、SMPL world skeleton/mesh、G1 rendered pose；每列固定相同 source frame/time。增加小插图说明 Y-up → Z-up，及 24 关节中选取 14 个 world transforms 作为 GMR 目标。

**解决的问题：** 读者能直接看出动作语义保留、身体比例变化和约束造成的差别。

**制作：** 优先选本地已生成诊断的 `gWA_sBM_cAll_d26_mWA0_ch07` 等样本。`audit_aistpp_retargeting.py` 已生成 `comparison.html`、`layers.npz` 和 `report.json`，可用于检查同帧对齐；论文成图再用统一相机渲染。当前 HTML 按各帧骨架范围自动缩放，若直接截图会掩盖位移或尺度差异，须改成固定尺度或明确只比较姿势。

**注意：** 原视频必须确实对应同一 motion/camera，不能只凭动作名字相似拼图。没有视频时用 SMPL/G1 双行即可。生产链直接传 SMPL world transforms 给 GMR，BVH 只是可选诊断链，不能画成主路径必经步骤。自己的碰撞、连续性修正与上游 GMR 的方法贡献应分别标注。[GMR 官方方法说明](https://jaraujo98.github.io/retargeting_matters/)。

### C09 — 分层检索与动作重排序

**位置：** 第 5 章 `sec:ch5_hierarchical_retrieval` 末尾。建议图题：*From audio similarity to motion-compatible ranking*。

**构图：** 左侧真实 query 的三种表示；中间展示若干 segment scores 如何按 track 做 Top-3 mean、genre restriction 和 abstention；右侧展示同一首音乐关联的多个动作，按 tempo/key-pose/activity 重新排序。用一个具体案例说明“音乐最像”不必是“动作最终最合适”。

**制作：** diagrams.net 绘流程，Matplotlib 绘组件分数的水平分组条形图。数值来自同一次 `MusicMotionMatcher.match()` 输出。若人为构造数值，仅标为 illustrative example，不作为实验结果。

**取舍：** 主图放层次、门控与排序前后；所有权重集中在正文或小表，避免重复英文旧图的大量微小公式。风格激活与相似度分数不能标成经过校准的正确概率。

### C10 — 稳定选择与切换的状态/时间复合图

**位置：** 第 5 章 `subsec:ch5_commitment_readiness` 后。建议图题：*Evidence accumulation and beat-boundary commitment*。

**构图：** (a) 简化状态图：current、evidence、pending、prepared、transitioning，加 rejected、not ready、clip end 的回退路径；(b) 时间轴绘 A/B 候选分数、0.08 margin、连续三个 completed retrieval wins、pending、ready 和 accepted-beat 边界。

**解决的问题：** 为什么不是 Top-1 一变就切，以及为什么已经 pending 却还没切。用两类箭头区分 relevance 与 diversity；四个 held bars 的多样性策略不能误画成所有相关性切换的强制等待。

**制作：** diagrams.net/TikZ；个案曲线由 trace 和实际 ranking 输出生成。`current/evidence/...` 是论文的逻辑状态抽象，实际 readiness 可以与证据积累并行发生，不要把它画成单一程序枚举的严格串行实现。

### C11 — 强拍接受与连续相位修正

**位置：** 第 5 章 `subsec:ch5_phase_correction` 后。建议图题：*Motion-aware beat acceptance and gradual phase correction*。

**构图：** 四条共享时间轴：raw beat confidence/contrast 与 accepted/rejected 标记；authored key-pose events；展开后的连续相位与目标事件；speed multiplier 和界限。标一个过早的细分拍被拒绝、一个可行强拍被接受、随后相位误差逐步消耗的过程。

**制作：** 复用实际控制器记录；现有 `analyze_motion_audio_phase.py` 有 phase、speed、accepted beat 和误差 CSV，适合起步。比较 authored 与 adaptive 时用相同音乐/动作/初始条件，若要加入 phase-snap 对照则需明确另实现该基线。

**指标解释：** 旧 CSV 的 `original_phase = audio_time / motion_duration` 衡量相对原编舞时钟的偏离，并不直接等于听觉节拍对齐好坏。图中应另给 beat-to-key-pose 事件误差或 BAS；循环相位 1→0 是表示回绕，不是实际姿态瞬移，必要时使用 unwrapped phase。一个个案不足以代替第 6 章统计比较。

### C12 — 状态兼容入口、根对齐与连续混合

**位置：** 第 5 章 `sec:ch5_continuous_transition`，入口优化章节先引用。建议图题：*State-aware entry selection and root-aligned transition*。

**构图：** (a) 当前 G1 姿态及 3 个候选入口，标左右脚接触和运动方向；(b) 候选的 pose/velocity/contact/root/salience cost 分项；(c) 鸟瞰 root 轨迹在 yaw/XY 对齐前后的位置；(d) 选中的过渡按 0、25%、50%、75%、100% 显示姿态，旁边画 smoothstep 权重。正文空间不足可拆成入口图与过渡图。

**解决的问题：** 姿态相近但速度反向或支撑脚不一致为何不理想；关节看起来连续但 root 仍会瞬移为何要单独处理。

**制作：** 候选和分项由 `build_motion_entry_features`、`select_motion_entry` 导出；机器人帧用 MuJoCo；轨迹和 cost 用 Matplotlib；组合用矢量排版。不要只对静止端点插值，当前代码在 blend 时两个 controller 都继续推进。仅保存最佳入口的 trace 不足以恢复所有候选分项，需要重算或补充导出。

## 3. 可选扩展图：按篇幅选择

| 图 | 插入位置 | 画什么、为什么 | 制作与数据 |
|---|---|---|---|
| 第 1 章系统效果概览 | `sec:intro_system` | 音乐波形、连续 4–6 帧 G1、小型“检索→调时→过渡”示意，让非本领域读者先知道最终做了什么 | 使用真实 matcher 输出帧；不重复完整架构 |
| AIST++ 数据示例 | 第 2 章 `subsec:lit_datasets` | 同步视频、3D 人体、音频，帮助理解数据从何而来 | 官方 AI Choreographer Fig. 1 的数据侧或作者自绘；若 C08 已足够可省 |
| 生成模型代表性框图 | 第 2 章生成方法节 | FACT 或 EDGE 选一个机制图，说明 cross-modal attention 或 conditional diffusion | 作者页或官方论文；不要每篇论文各塞一张完整架构 |
| Streaming causality 示例 | 第 2 章实时控制节 | 已提交历史、当前可用音频和不可见未来；或 DiscoForcing Fig. 1/3 中对应内容 | 原图授权引用或概念重绘，版面有限可合入 C02 |
| 目录结构关系图 | 第 4 章 `subsec:catalogue_join` | Track 1:N Segments，Track 1:N Motions，每个动作连 profile、GMR artifact、preflight | 用实际 JSON schema 简化为关系图；比把不同单位强行画成漏斗更清楚 |
| 目录覆盖与动作分布 | 第 4 章 `subsec:motion_profile` 或 catalogue_join | 按 genre 的动作数量，duration/key-pose density/equivalent speed 分布，揭示有限库的覆盖 | 直接读取 catalog.json；横向条形图＋箱线/散点图，通常比饼图更有用 |
| 碰撞检查和连续性诊断 | 第 4 章 collision_retargeting / retarget_continuity | 真实异常帧局部放大、碰撞对、相邻帧安全插值，以及修正前后姿态 | audit report、layers、保存的修正前后数据；没有真实对照时标清 schematic，不能制造“改进前” |
| 输出限幅机制 | 第 5 章 `sec:ch5_output_conditioning` | target/output 角度、速度、加速度分面图和干预标记 | 真实 pose NPZ、trace 与 limits；不同单位使用独立纵轴面板 |
| 失败与回退个案 | 第 5 章 observability 或第 6 章 | silence、rejection、not-ready、terminal hold 的一条运行时间轴 | 挑实际失败记录，解释系统边界，避免只展示成功个案 |

AIST++ 的官方[项目页](https://google.github.io/aichoreographer/)和[论文 Fig. 1](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_AI_Choreographer_Music_Conditioned_3D_Dance_Generation_With_AIST_ICCV_2021_paper.pdf)适合解释同步数据与 FACT 的关系。EDGE 的作者页有 *The EDGE Model, Explained* 示意，适合说明音乐编码、扩散和片段连接。[EDGE 项目页](https://edge-dance.github.io/)。DiscoForcing 作者页明确展示了 Fig. 1 的流式系统以及 Fig. 3 的静音/音乐切换场景。[DiscoForcing 项目页](https://discoforcing.github.io/)。

## 4. 文献图片的取舍和来源清单

| 文献/资源 | 建议用的内容 | 在本文的价值 | 建议 |
|---|---|---|---|
| [SMPL 官方站](https://smpl.is.tue.mpg.de/) | 人体 mesh、pose/shape、skeleton | 建立表示基础 | 优先自渲染 C01；基础模型论文原始年份是 2015，当前 bib 使用 2023 重印条目，需统一引用版本 |
| [AI Choreographer / AIST++](https://google.github.io/aichoreographer/) | 数据及 FACT 概览 | 解释成对音频/动作与人体生成 | 数据图有用，完整 transformer 细节可省 |
| [Bailando 原论文](https://arxiv.org/abs/2203.13055) | choreographic memory / discrete codebook 概念 | 解释动作单元复用与本文固定库的区别 | 优先放对照表或自绘小示意，无需为其运行训练模型 |
| [EDGE](https://edge-dance.github.io/) | 模型解释、编辑/时间约束示意 | 解释生成中的连续片段连接 | 与 FACT 二选一作为额外生成机制图 |
| [Beat-It](https://zikaihuangscut.github.io/Beat-It/) | Fig. 1 的 beat、keyframe、velocity 关系 | 与本文相位控制关系最直接 | 文献原图首选之一 |
| [Retargeting Matters / GMR](https://jaraujo98.github.io/retargeting_matters/) | source scaling、两阶段优化、重定向缺陷 | 支撑 embodiment gap 与离线验证动机 | 第 2 章解释上游方法；第 4 章用自己的同帧 G1 结果 |
| [RoboPerform 原论文](https://arxiv.org/abs/2512.23650) | 音频条件策略总览、机器人实验场景 | 与本文预计算库＋运动学播放形成方法对照 | 完整 teacher/student 图较复杂，优先表格和 C02；若用照片，清楚标成他人实机成果 |
| [DiscoForcing](https://discoforcing.github.io/) | 因果流式系统、变化输入演示 | 支撑实时性与历史连续性的讨论 | 优先因果时间轴，不必整幅复制 denoising 网络 |

不建议把他人论文中的性能柱状图作为自己的实验比较：采样协议、表示空间和硬件往往不同。引用原结果时保留其条件，并明确是 reported results。

取得原图时，优先使用论文 PDF 中的矢量内容或作者发布的原始图片，少用网页截图。每张图记录论文版本、图号、来源和使用条件；原样使用标 `Reproduced from`，修改或重绘标 `Adapted from` 或 `Author's schematic based on` 并引用。作者代码开源并不自动说明论文图片也使用同一许可证。SMPL 的模型包与可分享身体表示有不同条款，按实际下载资产的条款处理。[官方 SMPL-Body 说明](https://smpl.is.tue.mpg.de/license.html)。

## 5. 比图片更适合用表格的内容

| 表格 | 位置 | 推荐列 |
|---|---|---|
| 文献方法比较 | 第 2 章 synthesis | 方法、输入上下文、生成/检索、输出表示、节拍控制、部署证据、与本文关系 |
| 数据表示与单位 | 第 4 章 canonical_artifact | SMPL poses N×72；translation N×3；G1 root_pos N×3 m；root_rot N×4 wxyz；dof_pos N×29 rad；FPS、dof_names、provenance |
| 当前目录统计 | 第 4 章 catalogue_join | source/admitted/excluded motions、music IDs、audio variants、segments、各数组维度；注明 build 时间和版本 |
| 特征组成 | 第 4 章 audio_dsp_features | 特征族、维数、计算窗口、标准化方式、参与哪个分数 |
| 状态归属与失败策略 | 第 3 章 state_ownership | 组件、拥有的状态、失败条件、回退行为、诊断字段 |
| 在线关键参数 | 第 5 章末或附录 | 参数、当前值、单位、作用、CLI/协议来源；分开默认运行与正式实验配置 |

建议前三张表优先。表格使用 LaTeX booktabs，数值通过代码导出，避免把关键参数散落在多张图里而难以同步更新。

## 6. 为第 6 章预留的结果图

| 结果图 | 对应研究问题 | 数据与设计 |
|---|---|---|
| 留一音乐检索的 genre confusion matrix ＋ Recall@K | RQ1 | 每个 query music ID 的所有参考片段/相关动作均排除；按 music 为主要统计单位，窗口指标作诊断；与 same-segment sanity check 分开 |
| 拒绝能力与非目录音乐结果 | RQ1 | 正负例的 acceptance/rejection、错误类型；ROC/PR 需要连续 score，固定阈值的二值输出用混淆表 |
| 动作兼容性消融 | RQ2 | full 与 no_motion_compatibility 配对差值，分别展示节奏、活动与不可行速度候选比例；若要单独验证每一分项，还需额外注册实验 |
| 切换稳定性和变化响应 | RQ3 | 同一音乐变化流上的 motion ID 色带、pending/switch/blend milestone；附所有运行的响应延迟分布，不能只展示一个成功片段 |
| 节拍对齐与速度约束 | RQ4 | full 与 authored_timing 的 BAS/事件误差配对图，同时给出 speed saturation、beat acceptance，防止只靠少接受节拍提高指标 |
| 入口与过渡质量 | RQ5 | full 与 fixed_entry_simple_transition 的 root jump、joint speed/acceleration/jerk、contact mismatch；各单位单独画 |
| 多阶段耗时与实时预算 | RQ5 | waveform-to-match、preparation、control work 分开；正式 control work 标 60 Hz 对应16.67 ms，并将120 Hz pilot作为独立失败容量测试；用实际记录分布，不把窗口长度当计算延迟 |
| 长时间运行 | RQ3/RQ5 | 十分钟的 deadline miss、ready pool underrun、覆盖/切换、limiter/collision 干预；若没有时间序列内存记录，不能仅凭汇总内存画“内存稳定曲线” |

**现成自动图需要重新整理。** 实验 runner 已有 `latency_cdf`、`bas_distribution`、`selection_coverage`、`switch_speed_jerk`、`change_response_timeline` 输出。本次看到的正式子目录使用 SVG fallback：名为 `latency_cdf.svg` 的内容实际是按 run 排序的 p99 折线，不是真正 ECDF；`bas_distribution.svg` 也只是逐 run 的数值线。最终需要重新画正确的坐标及分组。

Matplotlib 分支的 latency CDF 统计对象是“每次运行的控制耗时 p99”，不能描述成“所有控制帧的耗时分布”。`switch_speed_jerk` 把 rad/s 与 rad/s³ 放在同一幅量级曲线上，不适合作为最终论文图，建议拆成共享横轴的分面图。数百个 run 的 unique motions 柱状图也应按 condition 聚合或展示分布，避免密集不可读的标签。

汇总验收完成后，优先读取 `docs/thesis/experiment_results/consolidated_results.json` 和对应原始数据重新绘图。保留失败运行的统计口径，并报告样本数、seed、置信区间算法和聚合单位；不能把同一首音乐的大量相邻帧当作独立实验样本。

## 7. 工具组合与现有素材入口

| 类型 | 推荐工具 | 原因/使用边界 |
|---|---|---|
| 系统架构、流程、状态与泳道 | diagrams.net；数学排版要求高时 TikZ | 项目已具备 drawio 源文件，复用成本最低；保留编辑源，导出 PDF/SVG。[官方导出说明](https://www.drawio.com/docs/manual/export/export-diagram/) |
| SMPL 网格、骨架、关键姿态 | AITViewer；需要精细材质时 Blender | 用真实模型与姿态渲染。AITViewer 已提供 SMPL 与截图、视频、headless 支持。[官方文档](https://eth-ait.github.io/aitviewer/) |
| G1 姿态、脚接触、碰撞对 | 项目 MuJoCo 模型＋Python 渲染 | 与实验模型一致；保留确定的 qpos、相机和帧号。[MuJoCo Python 文档](https://mujoco.readthedocs.io/en/stable/python.html) |
| 曲线、统计、heatmap、cost 条形图 | Python＋NumPy/Pandas＋Matplotlib | 从真实 CSV/JSON/NPZ 可复现出图；导出 PDF/SVG 和 PNG 预览。[savefig 官方文档](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html) |
| 图版组合、箭头、局部放大 | Inkscape 或其他熟悉的矢量编辑器 | 只负责排版标注，不手动改数据点 |
| 精确数值与方法对照 | LaTeX booktabs | 文字可搜索，格式与论文一致 |

核心技术图使用程序绘制或真实模型渲染。生成式图片可用于非技术装饰，但不适合表示精确关节拓扑、重定向结果、碰撞修正或实验曲线。

可复用的本地入口：

- [已有英文 drawio](C:/Users/Eddie/Desktop/MusicRobot/docs/thesis/figures/humanoid_matcher_model_english.drawio)：提取布局和配色，先更新内容。
- [AIST++ 谷值检测](C:/Users/Eddie/Desktop/MusicRobot/realtime/humanoid_robot/src/aistpp_velocity_keypoints.py)：C07 的数值来源。
- [SMPL FK 与坐标转换](C:/Users/Eddie/Desktop/MusicRobot/realtime/humanoid_robot/src/aistpp_smpl.py)：C01/C08 骨架和 14 个目标的定义。
- [重定向诊断代码](C:/Users/Eddie/Desktop/MusicRobot/realtime/humanoid_robot/src/audit_aistpp_retargeting.py)：同帧源视频、SMPL、G1 对照。
- [已有 Waacking 对照页](C:/Users/Eddie/Desktop/MusicRobot/realtime/humanoid_robot/data/aistpp_gmr/diagnostics/gWA_sBM_cAll_d26_mWA0_ch07/comparison.html)：挑选动作帧前核对同步。
- [目录与匹配逻辑](C:/Users/Eddie/Desktop/MusicRobot/realtime/humanoid_robot/src/music_motion_catalog.py)：窗口、编码、分数、稳定选择、目录统计。
- [在线 matcher](C:/Users/Eddie/Desktop/MusicRobot/realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py)：异步处理、入口分项、准备和 trace。
- [根运动、混合、限幅](C:/Users/Eddie/Desktop/MusicRobot/realtime/humanoid_robot/src/robot_motion.py)：C12 与输出机制图。
- [相位诊断](C:/Users/Eddie/Desktop/MusicRobot/realtime/humanoid_robot/src/test/analyze_motion_audio_phase.py)：时间轴与相位 CSV。
- [实验 runner](C:/Users/Eddie/Desktop/MusicRobot/realtime/humanoid_robot/src/test/run_humanoid_matcher_experiments.py)、[正式实验说明](C:/Users/Eddie/Desktop/MusicRobot/realtime/humanoid_robot/src/test/README.md)：第 6 章图表的数据契约。

## 8. 实施顺序与论文格式

建议分三轮：

1. **先形成可读的方法主线：** C04 架构 → C01 SMPL/G1 → C08 重定向 → C07 关键姿态 → C11 节拍相位 → C12 过渡。六张图对应读者最难自行想象的内容。
2. **再解释决策和研究定位：** C09 检索 → C10 稳定切换 → C05 多时间尺度 → C06 音频表示 → C02 文献路线 → C03 Beat-It。同期完成文献比较、表示单位和目录统计三张表。
3. **最后填入统计证据：** 等正式汇总完成，再做第 6 章分组和配对比较图；按剩余篇幅补 intro teaser、目录覆盖、失败案例和碰撞细节。

每幅图采用英文标签，与论文符号统一；字母分图 `(a)–(d)`，相同语义使用固定颜色。结构图和统计图优先 PDF 矢量输出，3D 渲染用足够分辨率的 PNG；至少按论文实际宽度检查一次，建议图内文字最终不小于约 8–9 pt。既有两幅大横图不能只压缩到一页宽而不重排。

建议保留“编辑源/绘图脚本＋数据配置＋最终 PDF/PNG＋图注来源”四项。已有 figures 目录可以继续使用，但新文件名应带章节和语义，例如 `ch4_velocity_valleys.pdf`、`ch5_transition_entry.pdf`；这是命名建议，这些图尚未生成。

各章 preview wrapper 当前未加载 `graphicx`，正式插图时添加；需要子图时再加载 `subcaption`，表格需要 `booktabs`。普通 LaTeX 编译优先包含 PDF/PNG，SVG 保留作编辑源。每张图前用一两句正文指出“要观察什么”，图注说明输入样本、参数、颜色和比较条件，让图本身能被独立理解。
