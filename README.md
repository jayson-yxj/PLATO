# PLATO - Point-Line Augmented Tracking and Optimization

**基于点线特征的增量式Structure from Motion系统**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 项目简介

PLATO是一个现代化的3D重建系统，结合了**点特征**和**线特征**进行增量式Structure from Motion (SFM)。系统采用深度学习特征提取器（SuperPoint、SOLD2）和传统方法（LSD），实现高精度的相机位姿估计和3D场景重建。

### 核心特性

- 🎯 **点线融合**：同时利用点特征和线特征，提高重建鲁棒性
- 🚀 **深度学习**：集成SuperPoint、SuperGlue、SOLD2等先进模型
- ⚡ **性能优化**：局部BA、关键帧策略，实现准实时处理
- 📊 **实时可视化**：Open3D实时显示3D重建过程
- 🔧 **模块化设计**：清晰的代码结构，易于扩展

---

## 🏗️ 项目结构

```
PLATO/
├── plato/                      # 核心代码包
│   ├── data/                   # 数据加载模块
│   │   ├── image_loader.py     # 图像加载器
│   │   └── __init__.py
│   ├── features/               # 特征提取和匹配
│   │   ├── extractors/         # 特征提取器
│   │   │   ├── base.py         # 基类定义
│   │   │   ├── superpoint.py   # SuperPoint提取器
│   │   │   ├── sold2.py        # SOLD2/LSD提取器
│   │   │   └── __init__.py
│   │   ├── descriptors/        # 描述子模块
│   │   └── __init__.py
│   ├── matching/               # 特征匹配模块
│   │   └── matchers/
│   ├── geometry/               # 几何计算模块
│   ├── optimization/           # BA优化模块
│   ├── sfm/                    # SFM核心模块
│   ├── utils/                  # 工具函数
│   │   ├── batch_processor.py  # 批量处理
│   │   ├── progress.py         # 进度显示
│   │   └── __init__.py
│   └── __init__.py
├── tests/                      # 测试脚本
│   ├── test_image_loader.py    # 图像加载测试
│   ├── test_feature_extractors.py  # 特征提取测试
│   └── benchmark_performance.py     # 性能基准测试
├── third_party/                # 第三方库
│   ├── SuperGluePretrainedNetwork/
│   └── SOLD2/
├── data/                       # 数据目录
│   ├── test_images/            # 测试图像
│   ├── eth3d/                  # ETH3D数据集
│   └── tum_rgbd/               # TUM RGB-D数据集
├── scripts/                    # 脚本工具
│   └── verify_setup.py         # 环境验证
├── requirements.txt            # 依赖列表
├── setup.py                    # 安装脚本
└── README.md                   # 本文件
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐，用于GPU加速)
- 8GB+ RAM
- 2GB+ GPU显存（可选）

### 2. 安装

```bash
# 克隆项目
cd /path/to/LineSFM/PLATO

# 安装依赖
pip install -r requirements.txt

# 安装PLATO包（开发模式）
pip install -e .
```

### 3. 验证安装

```bash
# 运行环境验证脚本
python scripts/verify_setup.py
```

### 4. 运行示例

```bash
# 测试图像加载
python tests/test_image_loader.py

# 测试特征提取
python tests/test_feature_extractors.py

# 性能基准测试
python tests/benchmark_performance.py

# 运行完整SFM（在上级目录）
cd ..
python LSD_SFM_BA.py
```

---

## 📚 使用指南

### 数据加载

```python
from plato.data.image_loader import ImageLoader

# 创建加载器
loader = ImageLoader('data/test_images')

# 加载单张图像
image = loader[0]

# 批量加载
images = [loader[i] for i in range(5)]

# 获取图像信息
info = loader.get_image_info(0)
print(f"图像路径: {info['path']}")
print(f"图像大小: {info['size_bytes']} bytes")
```

### 特征提取

#### SuperPoint 点特征

```python
from plato.features.extractors.superpoint import SuperPointDetector

# 创建提取器
detector = SuperPointDetector(
    max_keypoints=2000,
    conf_thresh=0.015
)

# 提取特征
keypoints, descriptors = detector.extract(image)
print(f"检测到 {len(keypoints)} 个特征点")
print(f"描述子形状: {descriptors.shape}")  # (N, 256)
```

#### SOLD2 线特征

```python
from plato.features.extractors.sold2 import SOLD2Detector

# 创建提取器
detector = SOLD2Detector(
    device='cuda',
    min_length=10.0
)

# 提取线段
lines, descriptors = detector.extract(image)
print(f"检测到 {len(lines)} 条线段")
print(f"描述子形状: {descriptors.shape}")  # (N, 128)
```

#### LSD 线特征（传统方法）

```python
from plato.features.extractors.sold2 import LSDDetector

# 创建提取器
detector = LSDDetector(min_length=20.0)

# 提取线段（无描述子）
lines, _ = detector.extract(image)
print(f"检测到 {len(lines)} 条线段")
```

### 批量处理

```python
from plato.utils.batch_processor import BatchProcessor
from plato.features.extractors.superpoint import SuperPointDetector

# 创建处理器和提取器
processor = BatchProcessor(num_workers=4)
extractor = SuperPointDetector()

# 批量处理图像
results = processor.process_images(
    images,
    extractor.extract,
    desc="提取特征",
    parallel=True  # 启用并行处理
)

# 处理结果
for i, (kp, desc) in enumerate(results):
    print(f"图像 {i}: {len(kp)} 个特征点")
```

---

## 🔬 模块详解

### 1. 数据加载模块 (`plato/data/`)

**ImageLoader** - 图像加载器

- 支持多种图像格式（JPG, PNG, BMP等）
- 自动排序和索引
- 批量加载功能
- 图像信息查询

### 2. 特征提取模块 (`plato/features/extractors/`)

**基类设计**

- `FeatureExtractor`: 所有提取器的抽象基类
- `PointFeatureExtractor`: 点特征提取器基类
- `LineFeatureExtractor`: 线特征提取器基类

**实现的提取器**

| 提取器 | 类型 | 描述子 | 速度 | 精度 |
|--------|------|--------|------|------|
| SuperPoint | 点 | 256维 | ~50ms | 高 |
| SOLD2 | 线 | 128维 | ~150ms | 高 |
| LSD | 线 | 无 | ~5ms | 中 |

### 3. 批量处理模块 (`plato/utils/`)

**BatchProcessor** - 批量处理器

- 顺序处理：适合单线程任务
- 并行处理：多线程加速（2-4倍）
- 进度显示：tqdm进度条
- 错误处理：单个失败不影响整体

**注意事项**

- SuperPoint和SOLD2支持并行处理
- LSD不支持多线程（OpenCV限制）

---

## 📊 性能基准

### 测试环境

- CPU: Intel i7-10700K
- GPU: NVIDIA RTX 3080
- 图像尺寸: 640x480
- CUDA: 11.3

### 性能数据

| 模块 | 时间 | 特征数 | GPU |
|------|------|--------|-----|
| SuperPoint | 50ms | 1000-2000 | ✅ |
| SOLD2 | 150ms | 80-150 | ✅ |
| LSD | 5ms | 50-200 | ❌ |
| SuperGlue | 80ms | - | ✅ |
| 局部BA (5帧) | 100ms | - | ❌ |
| 全局BA (10帧) | 2000ms | - | ❌ |

### 优化效果

| 优化项 | 提速比 | 说明 |
|--------|--------|------|
| 局部BA | 10-20x | 只优化最近N个关键帧 |
| 关键帧策略 | 3-5x | 减少70%的BA计算 |
| 并行处理 | 2-4x | 多线程特征提取 |

---

## 🧪 测试

### 运行所有测试

```bash
cd PLATO

# 图像加载测试
python tests/test_image_loader.py

# 特征提取测试
python tests/test_feature_extractors.py

# 性能基准测试
python tests/benchmark_performance.py
```

### 测试覆盖

- ✅ 图像加载功能
- ✅ SuperPoint特征提取
- ✅ SOLD2线特征提取
- ✅ LSD线特征提取
- ✅ 批量处理功能
- ✅ 性能基准测试

---

## 🛠️ 开发指南

### 添加新的特征提取器

1. **继承基类**

```python
from plato.features.extractors.base import PointFeatureExtractor

class MyExtractor(PointFeatureExtractor):
    def __init__(self, ...):
        # 初始化
        pass
    
    def extract(self, image):
        # 实现特征提取
        keypoints = ...
        descriptors = ...
        return keypoints, descriptors
    
    def get_name(self):
        return "MyExtractor"
```

2. **添加到模块**

将文件保存到 `plato/features/extractors/my_extractor.py`

3. **编写测试**

在 `tests/test_feature_extractors.py` 中添加测试用例

4. **更新文档**

在README中添加使用说明

### 代码风格

- 遵循PEP 8规范
- 使用类型注解
- 编写完整的docstring
- 添加单元测试

---

## 📖 API文档

### ImageLoader

```python
class ImageLoader:
    """图像加载器"""
    
    def __init__(self, data_dir: str)
    def __len__(self) -> int
    def __getitem__(self, index: int) -> np.ndarray
    def load_image(self, index: int) -> np.ndarray
    def load_batch(self, indices: List[int]) -> List[np.ndarray]
    def get_image_info(self, index: int) -> dict
```

### FeatureExtractor

```python
class FeatureExtractor(ABC):
    """特征提取器基类"""
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> Tuple
    
    @abstractmethod
    def get_name(self) -> str
```

### BatchProcessor

```python
class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, num_workers: int = 4)
    def process(self, items: List, process_fn: Callable, desc: str) -> List
    def process_parallel(self, items: List, process_fn: Callable, desc: str) -> List
    def process_images(self, images: List, process_fn: Callable, 
                      desc: str, parallel: bool = True) -> List
```

---

## 🐛 常见问题

### Q1: SOLD2初始化失败

**问题**: `ImportError: kornia.feature.SOLD2 不可用`

**解决**: 
```bash
pip install kornia
```

### Q2: CUDA out of memory

**问题**: GPU显存不足

**解决**:
- 减少 `max_keypoints` 参数
- 使用CPU模式: `device='cpu'`
- 减少批处理大小

### Q3: LSD多线程崩溃

**问题**: `double free or corruption`

**解决**: LSD不支持多线程，使用 `parallel=False`

```python
processor.process_images(images, lsd.extract, parallel=False)
```

### Q4: SuperGlue加载慢

**问题**: 首次运行需要下载模型

**解决**: 模型会自动下载到 `~/.cache/torch/`，首次运行需要等待

---

## 📝 更新日志

### v0.1.0 (2024-12-23)

**新增功能**
- ✨ 完整的特征提取模块（SuperPoint、SOLD2、LSD）
- ✨ 批量处理系统（支持并行）
- ✨ 图像加载器（支持批量加载）
- ✨ 完整的测试套件
- ✨ 性能基准测试

**性能优化**
- ⚡ 局部BA（10-20倍加速）
- ⚡ 关键帧策略（减少70%计算）
- ⚡ 并行特征提取（2-4倍加速）

**Bug修复**
- 🐛 修复LSD多线程崩溃问题
- 🐛 修复SOLD2坐标转换问题

---

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

### 贡献流程

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

本项目使用了以下开源项目：

- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) - 点特征提取
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) - 特征匹配
- [SOLD2](https://github.com/cvg/SOLD2) - 线特征提取
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Open3D](http://www.open3d.org/) - 3D数据处理

---

## 📧 联系方式

- 项目主页: [GitHub](https://github.com/jayson-yxj/PLATO)
- 问题反馈: [Issues](https://github.com/yourusername/PLATO/issues)
- 邮箱: 3630115255@qq.com

---

## 🌟 Star History

如果这个项目对你有帮助，请给我们一个 ⭐️！

---

**Made with ❤️ by PLATO Team**