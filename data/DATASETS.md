# 数据集准备指南

## 1. ETH3D数据集

### 下载链接
- 官网：https://www.eth3d.net/
- 训练集：https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z
- 测试集：https://www.eth3d.net/data/multi_view_test_dslr_undistorted.7z

### 数据集特点
- 高质量DSLR图片
- 包含ground truth 3D点云
- 适合SFM评估

### 推荐序列（小规模测试）
- [ ] courtyard (91 images, ~500MB)
- [ ] delivery_area (68 images, ~400MB)
- [ ] facade (85 images, ~450MB)


### 下载命令
```bash
cd PLATO/data
mkdir -p eth3d
cd eth3d

# 1. 下载图像数据（必须）
wget https://www.eth3d.net/data/delivery_area_dslr_undistorted.7z

# 2. 下载ground truth点云（推荐）
wget https://www.eth3d.net/data/delivery_area_scan_clean.7z

# 3. 解压
7z x delivery_area_dslr_undistorted.7z
7z x delivery_area_scan_clean.7z
```

## 2. TUM-RGBD数据集

### 下载链接
- 官网：https://vision.in.tum.de/data/datasets/rgbd-dataset
- 直接下载：https://vision.in.tum.de/data/datasets/rgbd-dataset/download

### 推荐序列（1-2个）
- [ ] freiburg1_desk (2.7GB)
- [ ] freiburg1_room (2.1GB)

### 下载命令
```bash
cd PLATO/data
mkdir -p tum_rgbd
cd tum_rgbd

# 下载freiburg1_desk
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xzf rgbd_dataset_freiburg1_desk.tgz


```

## 3. 自建测试集

### 当前测试图片
- 位置：`data/test_images/`
- 数量：8-10张
- 来源：项目中的images/, images2/, images3/

### 扩展计划
- [ ] 添加低纹理场景（墙面、天空）
- [ ] 添加结构化场景（建筑、走廊）
- [ ] 添加不同光照条件