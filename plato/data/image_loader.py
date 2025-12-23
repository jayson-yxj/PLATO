import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

class ImageLoader:
    '''图像加载器'''
    def __init__(self,data_dir:str):
        self.data_dir = Path(data_dir)
        self.image_paths = self._scan_images()
    
    def _scan_images(self) -> List[Path]:
        '''扫描目录里的所有图像文件'''
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(self.data_dir.glob(ext))
        return sorted(image_paths)
    
    def load_image(self, index:int) -> np.ndarray:
        '''加载指定索引的图像'''
        img_path = self.image_paths[index]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {img_path}")
        return img
    
    def load_gray(self, index:int) -> np.ndarray:
        '''加载指定索引的灰度图像'''
        img = self.load_image(index)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img
    
    def resize_image(self, index:int, size:Tuple[int,int]) -> np.ndarray:
        '''加载并调整指定索引图像的大小'''
        img = self.load_image(index)
        resized_img = cv2.resize(img, size)
        return resized_img
    
    def normalize_image(self, index:int) -> np.ndarray:
        '''加载并归一化指定索引图像'''
        img = self.load_image(index)
        normalized_img = img.astype(np.float32) / 255.0
        return normalized_img
    
    def equalize_histogram(self, index:int) -> np.ndarray:
        '''均衡化图像直方图'''
        img = self.load_image(index)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized_img = cv2.equalizeHist(gray_img)
        return equalized_img

    def __len__(self) -> int:
        '''图像数量'''
        return len(self.image_paths)
    
    def __getitem__(self, index:int) -> np.ndarray:
        '''通过索引获取图像'''
        return self.load_image(index)