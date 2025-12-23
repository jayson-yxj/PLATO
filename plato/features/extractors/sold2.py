'''SOLD2线特征提取器'''
import torch
import numpy as np
import cv2
from typing import Tuple, Optional

from .base import LineFeatureExtractor

try:
    from kornia.feature import SOLD2 as KorniaSOLD2
    SOLD2_AVAILABLE = True
except ImportError:
    SOLD2_AVAILABLE = False
    print("警告: kornia.feature.SOLD2 不可用，请安装: pip install kornia")


class SOLD2Detector(LineFeatureExtractor):
    '''SOLD2线特征提取器'''
    
    def __init__(self, device: str = 'cuda', min_length: float = 10.0):
        """
        初始化SOLD2线特征提取器
        
        Args:
            device: 'cuda' 或 'cpu'
            min_length: 最小线段长度阈值（像素）
        
        Raises:
            ImportError: 如果SOLD2不可用
        """
        if not SOLD2_AVAILABLE:
            raise ImportError(
                "SOLD2不可用，请安装kornia: pip install kornia\n"
                "或使用传统LSD检测器"
            )
        
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.min_length = min_length
        
        # 初始化SOLD2模型
        self.detector = KorniaSOLD2(pretrained=True).eval().to(self.device)
        
        print(f"SOLD2初始化完成 (device: {self.device}, min_length: {min_length})")
    
    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取SOLD2线特征
        
        Args:
            image: 输入图像 (H, W, 3) BGR格式 或 (H, W) 灰度图
        
        Returns:
            lines: 线段数组 (N, 4)，格式为 [x1, y1, x2, y2]
            descriptors: 线段描述子 (N, 128)
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # 归一化到 [0, 1]
        gray_norm = gray.astype(np.float32) / 255.0
        
        # 转换为torch tensor (1, 1, H, W)
        gray_tensor = torch.tensor(
            gray_norm, 
            dtype=torch.float32, 
            device=self.device
        )
        gray_tensor = gray_tensor.unsqueeze(0).unsqueeze(0)
        
        # 检测线段
        with torch.no_grad():
            outputs = self.detector(gray_tensor)
        
        # 提取线段 - outputs['line_segments']是一个列表
        line_segments_list = outputs['line_segments'][0]  # 第一个batch
        
        # 提取密集描述子 - shape: (1, 128, H/4, W/4)
        dense_desc = outputs['dense_desc']
        
        # 转换格式并提取描述子
        if len(line_segments_list) > 0:
            # 转换为numpy数组
            if isinstance(line_segments_list, torch.Tensor):
                line_segments = line_segments_list.cpu().numpy()
            else:
                line_segments = np.array(line_segments_list)
            
            # ⚠️ 重要：SOLD2输出的坐标格式是 (y, x) 而不是 (x, y)
            # 需要交换坐标轴以匹配OpenCV的 (x, y) 格式
            # 原始格式: [[y1, x1], [y2, x2]]
            # 目标格式: [x1, y1, x2, y2]
            lines = np.zeros((len(line_segments), 4), dtype=np.float32)
            lines[:, 0] = line_segments[:, 0, 1]  # x1 (从第二列取)
            lines[:, 1] = line_segments[:, 0, 0]  # y1 (从第一列取)
            lines[:, 2] = line_segments[:, 1, 1]  # x2 (从第二列取)
            lines[:, 3] = line_segments[:, 1, 0]  # y2 (从第一列取)
            
            # 从密集描述子中采样线段描述子
            descriptors = self._extract_line_descriptors(lines, dense_desc, h, w)
            
            # 过滤太短的线段
            lengths = np.sqrt(
                (lines[:, 2] - lines[:, 0])**2 + 
                (lines[:, 3] - lines[:, 1])**2
            )
            valid_mask = lengths > self.min_length
            lines = lines[valid_mask]
            descriptors = descriptors[valid_mask]
        else:
            lines = np.array([]).reshape(0, 4)
            descriptors = np.array([]).reshape(0, 128)
        
        return lines, descriptors
    
    def _extract_line_descriptors(
        self, 
        lines: np.ndarray, 
        dense_desc: torch.Tensor, 
        img_h: int, 
        img_w: int
    ) -> np.ndarray:
        """
        从密集描述子中提取线段描述子
        
        Args:
            lines: (N, 4) 线段坐标 [x1, y1, x2, y2]
            dense_desc: (1, 128, H', W') 密集描述子
            img_h, img_w: 原始图像尺寸
        
        Returns:
            descriptors: (N, 128) 线段描述子
        """
        desc_h, desc_w = dense_desc.shape[2], dense_desc.shape[3]
        scale_h = desc_h / img_h
        scale_w = desc_w / img_w
        
        descriptors = []
        for line in lines:
            x1, y1, x2, y2 = line
            
            # 计算线段中点
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # 映射到描述子空间
            desc_x = int(mid_x * scale_w)
            desc_y = int(mid_y * scale_h)
            
            # 边界检查
            desc_x = max(0, min(desc_w - 1, desc_x))
            desc_y = max(0, min(desc_h - 1, desc_y))
            
            # 提取描述子
            desc = dense_desc[0, :, desc_y, desc_x].cpu().numpy()
            descriptors.append(desc)
        
        return np.array(descriptors) if descriptors else np.array([]).reshape(0, 128)
    
    def get_name(self) -> str:
        """
        获取提取器名称
        
        Returns:
            名称字符串
        """
        return "SOLD2"
    
    def set_min_length(self, min_length: float):
        """
        设置最小线段长度阈值
        
        Args:
            min_length: 最小长度（像素）
        """
        self.min_length = min_length
        print(f"SOLD2最小线段长度已更新: {min_length}")


class LSDDetector(LineFeatureExtractor):
    '''传统LSD线段检测器（备用方案）'''
    
    def __init__(
        self, 
        scale: float = 0.8,
        sigma_scale: float = 0.6,
        quant: float = 2.0,
        ang_th: float = 22.5,
        log_eps: float = 0,
        density_th: float = 0.7,
        n_bins: int = 1024,
        min_length: float = 20.0
    ):
        """
        初始化LSD线段检测器
        
        Args:
            scale: 图像缩放比例
            sigma_scale: 高斯模糊sigma
            quant: 量化误差
            ang_th: 角度阈值
            log_eps: 检测阈值
            density_th: 密度阈值
            n_bins: bins数量
            min_length: 最小线段长度
        """
        self.min_length = min_length
        
        self.lsd_detector = cv2.createLineSegmentDetector(
            cv2.LSD_REFINE_STD,
            scale=scale,
            sigma_scale=sigma_scale,
            quant=quant,
            ang_th=ang_th,
            log_eps=log_eps,
            density_th=density_th,
            n_bins=n_bins
        )
        
        print(f"LSD初始化完成 (min_length: {min_length})")
    
    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, None]:
        """
        提取LSD线特征
        
        Args:
            image: 输入图像 (H, W, 3) BGR格式 或 (H, W) 灰度图
        
        Returns:
            lines: 线段数组 (N, 4)，格式为 [x1, y1, x2, y2]
            descriptors: None（LSD不提供描述子）
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 检测线段
        lines, width, prec, nfa = self.lsd_detector.detect(gray)
        
        if lines is not None:
            lines = lines.reshape(-1, 4)
            
            # 过滤太短的线段
            lengths = np.sqrt(
                (lines[:, 2] - lines[:, 0])**2 + 
                (lines[:, 3] - lines[:, 1])**2
            )
            lines = lines[lengths > self.min_length]
        else:
            lines = np.array([]).reshape(0, 4)
        
        # LSD不提供描述子，返回None
        return lines, None
    
    def get_name(self) -> str:
        """
        获取提取器名称
        
        Returns:
            名称字符串
        """
        return "LSD"
    
    def set_min_length(self, min_length: float):
        """
        设置最小线段长度阈值
        
        Args:
            min_length: 最小长度（像素）
        """
        self.min_length = min_length
        print(f"LSD最小线段长度已更新: {min_length}")