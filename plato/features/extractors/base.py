'''特征提取基类'''
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple,Optional

class FeatureExtractor(ABC):
    '''特征提取基类'''
    @abstractmethod
    def extract(self, image:np.ndarray) -> Tuple:
        """
        提取特征
        
        Args:
            image: 输入图像
        
        Returns:
            特征数据（具体格式由子类定义）
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        获取特征提取器名称
        
        Returns:
            名称字符串
        """
        pass

    def __repr__(self):
        return f'{self.get_name()} Feature Extractor'
    

class PointFeatureExtractor(FeatureExtractor):
    '''点特征提取基类'''
    @abstractmethod
    def extract(self, image:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取点特征
        
        Args:
            image: 输入图像
        
        Returns:
            keypoints: 特征点坐标数组 (N, 2)
            descriptors: 特征描述子数组 (N, D)
        """
        pass


class LineFeatureExtractor(FeatureExtractor):
    '''线特征提取基类'''
    @abstractmethod
    def extract(self, image:np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        提取线特征
        
        Args:
            image: 输入图像
        
        Returns:
            lines: 线段数组 (M, 4) 每行表示 (x1, y1, x2, y2)
            descriptors: 线段描述子数组 (M, D)
        """
        pass
