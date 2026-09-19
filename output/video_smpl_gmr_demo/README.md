# 原始视频 / SMPL / GMR → G1 对比

- `original_smpl_gmr.mp4`：12 秒，30 fps，1600 × 720，包含原始音乐。
- `original_smpl_gmr_half_speed.mp4`：半速播放，24 秒；画面时间是原始动作时间。
- `provenance.json`：输入路径、SHA-256、采样帧号、后处理配置与诊断数据。

动作：`gBR_sBM_cAll_d05_mBR0_ch08`，原视频相机 c01。
原视频按时间戳采样；SMPL / GMR 为 60 fps，输出每隔一帧取样。
使用同名片段的共同零点，没有额外手动偏移；未通过相机标定独立验证亚帧同步。

SMPL 使用与重定向一致的 neutral 模型和零 shape 系数。两幅 3D 画面使用相同相机与米制尺度，各自水平根节点居中，保留原始高度和所有关节旋转。原始相机视角与 3D 相机未标定对齐。

右栏直接回放已有 GMR 数据，其中包括速度限制、关节限位和碰撞修正，不是未处理的 GMR 输出。MuJoCo 使用 mj_forward 做运动学渲染，没有测试动力学控制器或真实机器人跟踪。悬空现象保留于画面，没有逐帧贴地。

全片 720 个保存帧中，G1 关节角度越界帧数为 0（模型关节范围，容差 1e-6 rad）；最大关节速度为 9.4248 rad/s，与配置速度上限一致。二阶差分最大关节加速度为 1130.97 rad/s²；未给定加速度阈值，所以不将它标为加速度越界。以上指标无法单独解释重定向偏差的因果关系。

在项目根目录重新生成：

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/demo_video_smpl_gmr.py
```

附加依赖：`pillow`、`imageio-ffmpeg`；使用现有 MuJoCo、NumPy 和 SciPy。
