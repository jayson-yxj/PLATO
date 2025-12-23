"""
图像加载器模块

提供图像加载、预处理和批量操作功能。
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from functools import lru_cache


class ImageLoader:
    """
    图像加载器类
    
    提供图像加载、预处理和批量操作功能。支持索引访问、切片和迭代。
    
    Args:
        data_dir: 图像目录路径
        cache_size: 缓存大小，0表示不缓存
    
    Examples:
        >>> loader = ImageLoader('data/images')
        >>> img = loader[0]  # 加载第一张图片
        >>> imgs = loader[0:3]  # 加载前3张图片
        >>> for img in loader:  # 遍历所有图片
        ...     process(img)
        >>> print(len(loader))  # 图片总数
    """
    
    def __init__(self, data_dir: str, cache_size: int = 0):
        self.data_dir = Path(data_dir)
        self.image_paths = self._scan_images()
        self.cache_size = cache_size
        
        # 如果启用缓存，创建缓存版本的加载函数
        if cache_size > 0:
            self._load_cached = lru_cache(maxsize=cache_size)(self._load_image_impl)
        else:
            self._load_cached = self._load_image_impl
    
    def _scan_images(self) -> List[Path]:
        """
        扫描目录里的所有图像文件
        
        Returns:
            排序后的图像路径列表
        """
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        for ext in extensions:
            image_paths.extend(self.data_dir.glob(ext))
            # 也支持大写扩展名
            image_paths.extend(self.data_dir.glob(ext.upper()))
        return sorted(set(image_paths))  # 去重并排序
    
    def _load_image_impl(self, index: int) -> np.ndarray:
        """
        实际加载图像的实现
        
        Args:
            index: 图像索引
        
        Returns:
            加载的图像数组
        
        Raises:
            FileNotFoundError: 图像文件不存在或无法加载
        """
        img_path = self.image_paths[index]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {img_path}")
        return img
    
    def load_image(self, index: int) -> np.ndarray:
        """
        加载指定索引的图像
        
        Args:
            index: 图像索引（支持负索引）
        
        Returns:
            加载的图像数组 (H, W, C)
        
        Raises:
            IndexError: 索引超出范围
            FileNotFoundError: 图像文件不存在或无法加载
        """
        # 处理负索引
        if index < 0:
            index = len(self) + index
        
        # 边界检查
        if index < 0 or index >= len(self):
            raise IndexError(f"索引 {index} 超出范围 [0, {len(self)})")
        
        return self._load_cached(index)
    
    def load_gray(self, index: int) -> np.ndarray:
        """
        加载指定索引的灰度图像
        
        Args:
            index: 图像索引
        
        Returns:
            灰度图像数组 (H, W)
        """
        img = self.load_image(index)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def resize_image(self, index: int, size: Tuple[int, int]) -> np.ndarray:
        """
        加载并调整指定索引图像的大小
        
        Args:
            index: 图像索引
            size: 目标尺寸 (width, height)
        
        Returns:
            调整大小后的图像
        """
        img = self.load_image(index)
        return cv2.resize(img, size)
    
    def normalize_image(self, index: int) -> np.ndarray:
        """
        加载并归一化指定索引图像到 [0, 1]
        
        Args:
            index: 图像索引
        
        Returns:
            归一化后的图像 (float32)
        """
        img = self.load_image(index)
        return img.astype(np.float32) / 255.0
    
    def equalize_histogram(self, index: int) -> np.ndarray:
        """
        均衡化图像直方图
        
        Args:
            index: 图像索引
        
        Returns:
            直方图均衡化后的灰度图像
        """
        img = self.load_image(index)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray_img)
    
    def load_batch(self, indices: List[int]) -> List[np.ndarray]:
        """
        批量加载多张图像
        
        Args:
            indices: 图像索引列表
        
        Returns:
            图像列表
        """
        return [self.load_image(i) for i in indices]
    
    def load_all(self) -> List[np.ndarray]:
        """
        加载所有图像
        
        Returns:
            所有图像的列表
        
        Warning:
            如果图像数量很多，可能会占用大量内存
        """
        return [self.load_image(i) for i in range(len(self))]
    
    def get_image_info(self, index: int) -> dict:
        """
        获取图像信息（不加载图像）
        
        Args:
            index: 图像索引
        
        Returns:
            包含图像信息的字典
        """
        if index < 0:
            index = len(self) + index
        
        if index < 0 or index >= len(self):
            raise IndexError(f"索引超出范围")
        
        img_path = self.image_paths[index]
        return {
            'index': index,
            'path': str(img_path),
            'name': img_path.name,
            'exists': img_path.exists(),
            'size_bytes': img_path.stat().st_size if img_path.exists() else 0
        }
    
    def get_image_shape(self, index: int) -> Tuple[int, int, int]:
        """
        获取图像尺寸
        
        Args:
            index: 图像索引
        
        Returns:
            图像形状 (height, width, channels)
        """
        img = self.load_image(index)
        return img.shape
    
    def __len__(self) -> int:
        """
        返回图像数量
        
        Returns:
            图像总数
        """
        return len(self.image_paths)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        通过索引或切片获取图像
        
        Args:
            index: 整数索引或切片对象
        
        Returns:
            单张图像（索引）或图像列表（切片）
        
        Examples:
            >>> loader[0]      # 第一张图片
            >>> loader[-1]     # 最后一张图片
            >>> loader[0:3]    # 前3张图片
            >>> loader[::2]    # 每隔一张
        """
        # 处理切片
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self.load_image(i) for i in indices]
        
        # 处理单个索引
        return self.load_image(index)
    
    def __iter__(self):
        """
        支持迭代
        
        Yields:
            逐个图像
        
        Examples:
            >>> for img in loader:
            ...     process(img)
        """
        for i in range(len(self)):
            yield self.load_image(i)
    
    def __repr__(self) -> str:
        """字符串表示"""
