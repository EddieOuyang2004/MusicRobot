# Humanoid Matcher 模型图说明

## 建议图题

**图 X　Humanoid Matcher 音乐驱动人形机器人实时动作检索与控制模型架构**

该模型由离线知识库构建、在线跨模态检索、稳定动作选择、节拍相位控制、连续过渡和安全运动输出六部分组成。系统将最近 6 s 的音乐窗口编码为音乐嵌入、节奏—音色特征及可选风格标签，通过分层相似度计算检索 AIST++ 音乐及其关联动作；随后结合动作与当前音乐的速度、关键姿态密度和活跃度兼容性进行排序。候选动作必须满足连续胜出或多样性轮换条件，并在 accepted beat 构成的四拍小节边界进入切换。切换阶段对根位姿、关节位置和动作相位进行连续化处理，最终经关节动力学限制与运行时自碰撞检查后写入 MuJoCo Unitree G1 29-DoF 模型。

## 可编辑文件

- `humanoid_matcher_architecture.drawio`：主编辑文件，可用免费的 [diagrams.net](https://app.diagrams.net/) 打开。所有框、连接线、文字和颜色均可单独修改。
- `humanoid_matcher_architecture.svg`：矢量版本，可用 Inkscape、Adobe Illustrator、Affinity Designer 或 PowerPoint 编辑。
- `humanoid_matcher_architecture.png`：论文预览图，不建议作为后续编辑源。

## 与代码的对应关系

- 离线音乐目录、描述符和匹配公式：`realtime/humanoid_robot/src/music_motion_catalog.py`
- AIST++ 速度谷关键姿态：`realtime/humanoid_robot/src/aistpp_velocity_keypoints.py`
- AIST++ → GMR → G1 数据集：`realtime/humanoid_robot/src/build_aistpp_gmr_dataset.py`、`gmr_retarget_smpl_headless.py`
- 在线检索、稳定选择、动作预加载、入口评分与切换：`realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py`
- 实时音乐分析、节拍相位控制、GMR 采样与 MuJoCo 播放：`realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py` 及 `realtime/robot_arm/src/realtime_music_adaptive_player.py`
- 音乐姿态调制：`realtime/humanoid_robot/src/music_pose_modulator.py`
- 根轨迹连续性、动作混合和关节动力学限制：`realtime/humanoid_robot/src/robot_motion.py`

## 论文表述注意

Humanoid Matcher 不是单一端到端神经网络，而是一个由可选神经音频嵌入、手工 DSP 特征、基于相似度的检索、规则化稳定决策和运动学约束控制构成的混合模型。当前提交的 `catalog.json` 使用 DSP embedding；Discogs EffNet ONNX 是代码支持的可选后端，不应写成当前实验的必选模型。MuJoCo 部分调用 `mj_forward` 进行运动学预览，并不包含力矩控制、动力学平衡或真实机器人闭环控制。
