'''SuperPoint特征提取器'''
import sys
import os

# 添加 SuperGlue 路径
superglue_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../third_party/SuperGluePretrainedNetwork'
)
sys.path.insert(0, superglue_path)

from .base import PointFeatureExtractor
from models.matching import Matching
from models.utils import frame2tensor
import torch
import cv2
import numpy as np

class SuperPointDetector(PointFeatureExtractor):
    '''SuperPoint特征提取器'''
    
    def __init__(self,max_keypoints:int=2000,
                 conf_thresh:float=0.015):
        """
        初始化SuperPoint提取器
        
        Args:
            max_keypoints: 最大特征点数量
            conf_thresh: 置信度阈值
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_keypoints = max_keypoints
        self.conf_thresh = conf_thresh

        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': conf_thresh,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }

        self.matcher = Matching(config).to(self.device)

    def extract(self,image:np.ndarray):
        """
        提取SuperPoint特征
        
        Args:
            image: 输入图像 (H, W, 3) BGR格式
        
        Returns:
            keypoints: 关键点坐标 (N, 2)
            descriptors: 描述子 (N, 256)
        """

        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 转为tensor
        gray_tensor = frame2tensor(gray, self.device)

        # 特征提取
        with torch.no_grad(): # 关闭梯度计算
            pred = self.matcher.superpoint({'image': gray_tensor}) # 前向传播

            keypoints =pred['keypoints'][0].cpu().numpy() # (N, 2)
            descriptors = pred['descriptors'][0].cpu().numpy().T # (N, 256)

            return keypoints, descriptors
        
    def get_name(self):
        """
        获取提取器名称
        
        Returns:
            名称字符串
        """
        return "SuperPoint"
