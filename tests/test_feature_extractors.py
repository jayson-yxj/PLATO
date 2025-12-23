'''特征提取器测试'''
import sys
import os
import unittest
import numpy as np
import cv2
import torch

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from plato.features.extractors.superpoint import SuperPointDetector
from plato.features.extractors.sold2 import SOLD2Detector, LSDDetector


class TestSuperPointDetector(unittest.TestCase):
    '''SuperPoint特征提取器测试'''
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        print("\n" + "="*70)
        print("测试 SuperPoint 特征提取器")
        print("="*70)
        
        # 创建测试图像
        cls.test_image = cls._create_test_image()
        
        # 初始化检测器
        try:
            cls.detector = SuperPointDetector(
                max_keypoints=1000,
                conf_thresh=0.015
            )
            cls.detector_available = True
            print("✓ SuperPoint检测器初始化成功")
        except Exception as e:
            cls.detector_available = False
            print(f"✗ SuperPoint检测器初始化失败: {e}")
    
    @staticmethod
    def _create_test_image(size=(640, 480)):
        """创建测试图像"""
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 128
        
        # 添加一些特征点（角点）
        # 绘制矩形
        cv2.rectangle(img, (100, 100), (300, 200), (255, 255, 255), 2)
        cv2.rectangle(img, (350, 150), (500, 300), (200, 200, 200), 2)
        
        # 绘制圆形
        cv2.circle(img, (200, 350), 50, (255, 255, 255), 2)
        
        # 添加一些随机点
        for _ in range(20):
            x = np.random.randint(50, size[0]-50)
            y = np.random.randint(50, size[1]-50)
            cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
        
        return img
    
    def test_01_initialization(self):
        """测试初始化"""
        if not self.detector_available:
            self.skipTest("SuperPoint不可用")
        
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.max_keypoints, 1000)
        self.assertEqual(self.detector.conf_thresh, 0.015)
        print("✓ 初始化参数正确")
    
    def test_02_extract_features(self):
        """测试特征提取"""
        if not self.detector_available:
            self.skipTest("SuperPoint不可用")
        
        keypoints, descriptors = self.detector.extract(self.test_image)
        
        # 检查输出格式
        self.assertIsInstance(keypoints, np.ndarray)
        self.assertIsInstance(descriptors, np.ndarray)
        
        # 检查形状
        self.assertEqual(len(keypoints.shape), 2)
        self.assertEqual(keypoints.shape[1], 2)  # (N, 2)
        
        self.assertEqual(len(descriptors.shape), 2)
        self.assertEqual(descriptors.shape[1], 256)  # (N, 256)
        
        # 检查数量一致
        self.assertEqual(keypoints.shape[0], descriptors.shape[0])
        
        print(f"✓ 提取了 {len(keypoints)} 个特征点")
        print(f"  - 关键点形状: {keypoints.shape}")
        print(f"  - 描述子形状: {descriptors.shape}")
    
    def test_03_grayscale_input(self):
        """测试灰度图输入"""
        if not self.detector_available:
            self.skipTest("SuperPoint不可用")
        
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.extract(gray_image)
        
        self.assertGreater(len(keypoints), 0)
        print(f"✓ 灰度图输入正常，提取了 {len(keypoints)} 个特征点")
    
    def test_04_get_name(self):
        """测试获取名称"""
        if not self.detector_available:
            self.skipTest("SuperPoint不可用")
        
        name = self.detector.get_name()
        self.assertEqual(name, "SuperPoint")
        print(f"✓ 检测器名称: {name}")
    
    def test_05_repr(self):
        """测试字符串表示"""
        if not self.detector_available:
            self.skipTest("SuperPoint不可用")
        
        repr_str = repr(self.detector)
        self.assertIn("SuperPoint", repr_str)
        print(f"✓ 字符串表示: {repr_str}")


class TestSOLD2Detector(unittest.TestCase):
    '''SOLD2线特征提取器测试'''
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        print("\n" + "="*70)
        print("测试 SOLD2 线特征提取器")
        print("="*70)
        
        # 创建测试图像
        cls.test_image = cls._create_test_image_with_lines()
        
        # 初始化检测器
        try:
            cls.detector = SOLD2Detector(
                device='cuda' if torch.cuda.is_available() else 'cpu',
                min_length=10.0
            )
            cls.detector_available = True
            print("✓ SOLD2检测器初始化成功")
        except Exception as e:
            cls.detector_available = False
            print(f"✗ SOLD2检测器初始化失败: {e}")
    
    @staticmethod
    def _create_test_image_with_lines(size=(640, 480)):
        """创建包含线段的测试图像"""
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 50
        
        # 绘制水平线
        cv2.line(img, (50, 100), (590, 100), (255, 255, 255), 2)
        cv2.line(img, (50, 200), (590, 200), (200, 200, 200), 2)
        
        # 绘制垂直线
        cv2.line(img, (150, 50), (150, 430), (255, 255, 255), 2)
        cv2.line(img, (400, 50), (400, 430), (200, 200, 200), 2)
        
        # 绘制对角线
        cv2.line(img, (100, 250), (500, 400), (255, 255, 255), 2)
        cv2.line(img, (500, 250), (100, 400), (200, 200, 200), 2)
        
        # 绘制矩形（产生更多线段）
        cv2.rectangle(img, (200, 300), (440, 450), (255, 255, 255), 2)
        
        return img
    
    def test_01_initialization(self):
        """测试初始化"""
        if not self.detector_available:
            self.skipTest("SOLD2不可用")
        
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.min_length, 10.0)
        print("✓ 初始化参数正确")
    
    def test_02_extract_lines(self):
        """测试线段提取"""
        if not self.detector_available:
            self.skipTest("SOLD2不可用")
        
        lines, descriptors = self.detector.extract(self.test_image)
        
        # 检查输出格式
        self.assertIsInstance(lines, np.ndarray)
        self.assertIsInstance(descriptors, np.ndarray)
        
        # 检查形状
        if len(lines) > 0:
            self.assertEqual(len(lines.shape), 2)
            self.assertEqual(lines.shape[1], 4)  # (N, 4) [x1, y1, x2, y2]
            
            self.assertEqual(len(descriptors.shape), 2)
            self.assertEqual(descriptors.shape[1], 128)  # (N, 128)
            
            # 检查数量一致
            self.assertEqual(lines.shape[0], descriptors.shape[0])
            
            print(f"✓ 提取了 {len(lines)} 条线段")
            print(f"  - 线段形状: {lines.shape}")
            print(f"  - 描述子形状: {descriptors.shape}")
        else:
            print("⚠ 未检测到线段")
    
    def test_03_line_coordinates(self):
        """测试线段坐标有效性"""
        if not self.detector_available:
            self.skipTest("SOLD2不可用")
        
        lines, _ = self.detector.extract(self.test_image)
        
        if len(lines) > 0:
            # 检查坐标范围
            h, w = self.test_image.shape[:2]
            self.assertTrue(np.all(lines[:, 0] >= 0))  # x1 >= 0
            self.assertTrue(np.all(lines[:, 1] >= 0))  # y1 >= 0
            self.assertTrue(np.all(lines[:, 2] < w))   # x2 < width
            self.assertTrue(np.all(lines[:, 3] < h))   # y2 < height
            
            # 检查线段长度
            lengths = np.sqrt(
                (lines[:, 2] - lines[:, 0])**2 + 
                (lines[:, 3] - lines[:, 1])**2
            )
            self.assertTrue(np.all(lengths >= self.detector.min_length))
            
            print(f"✓ 线段坐标有效")
            print(f"  - 平均长度: {np.mean(lengths):.2f} 像素")
            print(f"  - 最短: {np.min(lengths):.2f}, 最长: {np.max(lengths):.2f}")
    
    def test_04_grayscale_input(self):
        """测试灰度图输入"""
        if not self.detector_available:
            self.skipTest("SOLD2不可用")
        
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        lines, descriptors = self.detector.extract(gray_image)
        
        self.assertIsInstance(lines, np.ndarray)
        print(f"✓ 灰度图输入正常，提取了 {len(lines)} 条线段")
    
    def test_05_set_min_length(self):
        """测试设置最小长度"""
        if not self.detector_available:
            self.skipTest("SOLD2不可用")
        
        self.detector.set_min_length(20.0)
        self.assertEqual(self.detector.min_length, 20.0)
        
        lines, _ = self.detector.extract(self.test_image)
        if len(lines) > 0:
            lengths = np.sqrt(
                (lines[:, 2] - lines[:, 0])**2 + 
                (lines[:, 3] - lines[:, 1])**2
            )
            self.assertTrue(np.all(lengths >= 20.0))
        
        # 恢复原值
        self.detector.set_min_length(10.0)
        print("✓ 最小长度设置正常")
    
    def test_06_get_name(self):
        """测试获取名称"""
        if not self.detector_available:
            self.skipTest("SOLD2不可用")
        
        name = self.detector.get_name()
        self.assertEqual(name, "SOLD2")
        print(f"✓ 检测器名称: {name}")


class TestLSDDetector(unittest.TestCase):
    '''LSD线段检测器测试'''
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        print("\n" + "="*70)
        print("测试 LSD 线段检测器")
        print("="*70)
        
        # 创建测试图像
        cls.test_image = cls._create_test_image_with_lines()
        
        # 初始化检测器
        cls.detector = LSDDetector(min_length=20.0)
        print("✓ LSD检测器初始化成功")
    
    @staticmethod
    def _create_test_image_with_lines(size=(640, 480)):
        """创建包含线段的测试图像"""
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 50
        
        # 绘制清晰的线段
        cv2.line(img, (50, 100), (590, 100), (255, 255, 255), 3)
        cv2.line(img, (50, 200), (590, 200), (255, 255, 255), 3)
        cv2.line(img, (150, 50), (150, 430), (255, 255, 255), 3)
        cv2.line(img, (400, 50), (400, 430), (255, 255, 255), 3)
        
        # 矩形
        cv2.rectangle(img, (200, 300), (440, 450), (255, 255, 255), 3)
        
        return img
    
    def test_01_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.min_length, 20.0)
        print("✓ 初始化参数正确")
    
    def test_02_extract_lines(self):
        """测试线段提取"""
        lines, descriptors = self.detector.extract(self.test_image)
        
        # 检查输出格式
        self.assertIsInstance(lines, np.ndarray)
        self.assertIsNone(descriptors)  # LSD不提供描述子
        
        # 检查形状
        if len(lines) > 0:
            self.assertEqual(len(lines.shape), 2)
            self.assertEqual(lines.shape[1], 4)  # (N, 4)
            
            print(f"✓ 提取了 {len(lines)} 条线段")
            print(f"  - 线段形状: {lines.shape}")
            print(f"  - 描述子: None (LSD不提供)")
        else:
            print("⚠ 未检测到线段")
    
    def test_03_line_length(self):
        """测试线段长度"""
        lines, _ = self.detector.extract(self.test_image)
        
        if len(lines) > 0:
            lengths = np.sqrt(
                (lines[:, 2] - lines[:, 0])**2 + 
                (lines[:, 3] - lines[:, 1])**2
            )
            self.assertTrue(np.all(lengths >= self.detector.min_length))
            
            print(f"✓ 所有线段长度 >= {self.detector.min_length}")
            print(f"  - 平均长度: {np.mean(lengths):.2f} 像素")
    
    def test_04_get_name(self):
        """测试获取名称"""
        name = self.detector.get_name()
        self.assertEqual(name, "LSD")
        print(f"✓ 检测器名称: {name}")


class TestFeatureExtractorComparison(unittest.TestCase):
    '''特征提取器对比测试'''
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        print("\n" + "="*70)
        print("特征提取器性能对比")
        print("="*70)
        
        # 创建测试图像
        cls.test_image = cls._create_complex_image()
        
        # 初始化所有检测器
        try:
            cls.superpoint = SuperPointDetector()
            cls.superpoint_available = True
        except:
            cls.superpoint_available = False
        
        try:
            cls.sold2 = SOLD2Detector()
            cls.sold2_available = True
        except:
            cls.sold2_available = False
        
        cls.lsd = LSDDetector()
    
    @staticmethod
    def _create_complex_image(size=(800, 600)):
        """创建复杂测试图像"""
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 100
        
        # 添加各种特征
        # 矩形
        cv2.rectangle(img, (100, 100), (300, 250), (255, 255, 255), 2)
        cv2.rectangle(img, (400, 150), (650, 400), (200, 200, 200), 2)
        
        # 圆形
        cv2.circle(img, (200, 450), 80, (255, 255, 255), 2)
        cv2.circle(img, (600, 450), 60, (200, 200, 200), -1)
        
        # 线段
        for i in range(5):
            y = 50 + i * 100
            cv2.line(img, (50, y), (750, y), (255, 255, 255), 2)
        
        # 随机点
        for _ in range(50):
            x = np.random.randint(50, size[0]-50)
            y = np.random.randint(50, size[1]-50)
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
        
        return img
    
    def test_01_point_feature_comparison(self):
        """对比点特征提取"""
        if not self.superpoint_available:
            self.skipTest("SuperPoint不可用")
        
        import time
        
        print("\n点特征提取对比:")
        
        # SuperPoint
        start = time.time()
        kp_sp, desc_sp = self.superpoint.extract(self.test_image)
        time_sp = time.time() - start
        
        print(f"  SuperPoint:")
        print(f"    - 特征点数: {len(kp_sp)}")
        print(f"    - 描述子维度: {desc_sp.shape[1]}")
        print(f"    - 耗时: {time_sp:.3f}s")
    
    def test_02_line_feature_comparison(self):
        """对比线特征提取"""
        import time
        
        print("\n线特征提取对比:")
        
        # SOLD2
        if self.sold2_available:
            start = time.time()
            lines_sold2, desc_sold2 = self.sold2.extract(self.test_image)
            time_sold2 = time.time() - start
            
            print(f"  SOLD2:")
            print(f"    - 线段数: {len(lines_sold2)}")
            print(f"    - 描述子维度: {desc_sold2.shape[1] if len(desc_sold2) > 0 else 'N/A'}")
            print(f"    - 耗时: {time_sold2:.3f}s")
        
        # LSD
        start = time.time()
        lines_lsd, _ = self.lsd.extract(self.test_image)
        time_lsd = time.time() - start
        
        print(f"  LSD:")
        print(f"    - 线段数: {len(lines_lsd)}")
        print(f"    - 描述子: 无")
        print(f"    - 耗时: {time_lsd:.3f}s")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestSuperPointDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestSOLD2Detector))
    suite.addTests(loader.loadTestsFromTestCase(TestLSDDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtractorComparison))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)