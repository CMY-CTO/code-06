# search_grid

## 项目简介

`search_grid` 是一个用于处理和分析 3D 动作数据的项目。该项目集成了多种数据处理和可视化方法，并提供了对 3D 动作数据的聚类和降维功能。

## 文件说明

- `search_grid.py`：主要的脚本文件，包含数据过滤、转换、分段和聚类可视化的功能。

## 安装

请确保您的环境中已经安装了以下依赖项：

```bash
pip install numpy torch smplx scikit-learn matplotlib pandas

使用说明
1. 数据过滤
filter_joints 函数用于根据给定的骨骼类型过滤关节数据。

2. 轴角转换为位置
axis_angle_to_position 函数使用 SMPL-X 模型将轴角数据转换为关节位置。

3. 计算加速度
compute_acceleration 函数计算关节位置的加速度。

4. 数据分段
segment_data 函数将数据文件进行分段处理。

5. 聚类和可视化
cluster_and_visualize 函数对分段后的数据进行聚类，并使用 t-SNE 进行降维可视化。

示例
以下是一个基本的使用示例：

```python

import search_grid

# 数据文件路径
npz_files = ['data1.npz', 'data2.npz']
model_path = 'path/to/smplx_model'
segment_path = 'path/to/segments'
t = 10  # 分段时间
skeleton_type = 'upper'
keypoint_type = 'position'

# 数据分段
search_grid.segment_data(npz_files, t, segment_path, model_path, skeleton_type, keypoint_type)

# 聚类和可视化
k = 5  # 聚类数
output_path = 'path/to/output'
search_grid.cluster_and_visualize(segment_path, k, output_path, t, skeleton_type, keypoint_type)
```

许可证
本项目使用 MIT 许可证，详情请参见 LICENSE 文件。