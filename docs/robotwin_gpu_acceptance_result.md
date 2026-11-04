# RoboTwin 独立环境 GPU 验收结果

修订 `2ffb4fc` 的最小 GPU 验收通过。脚本输出 `ROBOTWIN_GRAPHICS_GATE_PASS`，`acceptance.json` 的 `passed` 为 `true`，各阶段退出码均为 0。

| 检查 | 结果 |
| --- | --- |
| Vulkan | 宿主与 Apptainer 容器识别到同一张分配的 NVIDIA GPU，容器使用 EGL ICD |
| SAPIEN | 输出两张 160×120 RGB 图像，移动测试物体后有 2184 个像素变化 |
| CuRobo | 标准 UR5e 关节规划通过，保存了轨迹文件 |

这次验收覆盖容器图形和关节规划。RoboTwin 的笛卡尔规划、完整任务和 PACE 策略另行验收。
