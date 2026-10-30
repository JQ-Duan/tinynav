# 特征提取、匹配与几何检查

本目录承接 `frontend/sensors/DataPreparer` 输出的 `PreparedFrame`，沿用原项目的 SuperPoint → LightGlue → PnP RANSAC 路径。

| 文件 | 职责 |
| --- | --- |
| `feature_types.hpp/.cpp` | 特征点、描述子、有效掩码、匹配索引及数据检查 |
| `feature_extractor.hpp/.cpp` | SuperPoint 输入准备、模型调用、坐标还原 |
| `feature_matcher.hpp/.cpp` | LightGlue 调用、无效匹配和重复匹配过滤 |
| `geometry_checker.hpp/.cpp` | 利用当前深度和匹配点估计相对位姿，筛选几何内点 |
| `visual_frontend.hpp/.cpp` | 串联上述三个步骤，接收已经准备好的帧 |

函数声明放在 `.hpp`，实现放在 `.cpp`。TensorRT 的加载、显存管理和推理执行放在 `infrastructure/inference`。

## 数据与坐标约定

- 只对左图进行跨时刻特征匹配。此阶段的深度已经对齐左图，单位为米。
- SuperPoint 按引擎输入尺寸缩放灰度图；保留原包装的 `0..255` 输入范围，不再除以 255。
- 特征坐标按 `(x + 0.5) * 原图宽 / 模型宽 - 0.5` 还原，纵向独立计算；后续几何计算使用原图内参。
- 保留模型的全部特征槽位和有效掩码，避免破坏 LightGlue 输出索引和后续 VIO 的轨迹关联。
- LightGlue 的图像尺寸顺序为 `[宽, 高]`。默认特征阈值为 `0.0005`，匹配阈值为 `0.1`。
- 用**当前帧深度、当前帧匹配点**恢复当前相机下的三维点，以**参考帧像素**进行 PnP。默认有效深度严格满足 `0.1 < 深度 < 10` 米，至少 6 对点、6 个内点，EPNP RANSAC 阈值 2 像素、置信度 0.999、100 次迭代。
- 输出 `T_reference_current` 满足 `点_reference = T_reference_current * 点_current`。这是相机之间的相对位姿，不是机器人或世界坐标下的位姿。
- 几何失败时不返回有效位姿或内点约束。额外检查重复对应、退化点集、正深度与重投影误差；缺少视觉信息不代表机器人静止。

## 调用关系

```cpp
// 需启用 TensorRT 构建，并提供与目标设备匹配的引擎路径。
tinynav::infrastructure::TensorRtSession superpoint(superpoint_path);
tinynav::infrastructure::TensorRtSession lightglue(lightglue_path);
tinynav::frontend::FeatureExtractor extractor(superpoint);
tinynav::frontend::FeatureMatcher matcher(lightglue);
tinynav::frontend::VisualFrontend visual(extractor, matcher);

// prepared_frame 来自 DataPreparer；首次没有参考帧。
auto first = visual.process(prepared_frame);
// first.status == FeaturesOnly 时，调用方可保存 first.current_features。

// 后续由调用方明确提供参考关键帧特征。
auto result = visual.process(next_prepared_frame, &reference_features);
// result.status == Ready 时：
// result.geometry->T_reference_current：相对位姿；
// result.geometry->inlier_matches：可交给 VIO 的匹配；
// result.matches：几何检查前的候选，不能直接当有效视觉约束。
```

`VisualFrontend` 不自动更换参考帧，也不选择关键帧；参考帧由后续 VIO/关键帧管理逻辑决定。IMU 仍随调用方持有的 `PreparedFrame` 进入后续阶段。本模块不修改时间戳、不积分 IMU，也不确认或丢弃 `FrameAssembler` 中的帧。调用方串行调用同一个模型实例。

当前根目录 `main.cpp` 仍只启动传感器接收。这里已提供可调用的处理链，尚未接入 ROS 主循环和 VIO。

## 构建与验证

普通构建不要求 TensorRT，可以验证数据适配和几何算法：

```bash
source /opt/ros/humble/setup.bash
cmake -S . -B build
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

真实模型推理需要 TensorRT 8.5 或更新版本、CUDA 开发环境及可加载的模型引擎；启用时缺失依赖会直接报错。

```bash
cmake -S . -B build-trt -DTINYNAV_ENABLE_TENSORRT=ON
# 非标准安装目录可加 -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build build-trt --parallel 2
./build-trt/tensorrt_feature_smoke /path/to/superpoint.plan /path/to/lightglue.plan /path/to/left.png
```

推理封装使用 profile 0 的最大输入尺寸，按张量名称检查形状、类型、布局与返回值。仅支持已解析形状的线性 device 张量；不支持的导出形式会报错，不会切换到其他算法。

当前检查到的原项目 ONNX 文件是 Git LFS 指针，实体引擎为 `aarch64` 版本。本机测试使用受控模型输出验证适配逻辑，并使用真实 OpenCV PnP 验证几何；这些测试不证明真实模型推理或设备上的实时性能。真实模型可用上述 smoke 程序检查提取和同图匹配。

对应原实现：`tinynav-main/tinynav/core/models_trt.py`、`math_utils.py`、`perception_node.py`。TensorRT API 参考：[NVIDIA C++ API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/_nv_infer_runtime_8h_source.html)。
