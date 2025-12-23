import time
import cv2
import numpy as np
from pathlib import Path

def benchmark_feature_detection():
    """测试特征检测性能"""
    import sys
    sys.path.insert(0, '/home/sunteng/Desktop/HighTorque_vision/LineSFM')
    from LSD_SFM_BA import SuperPointDetector, SOLD2LineDetector, LSDLineDetector

    img = cv2.imread('images/3.png')
    img = cv2.resize(img, (640, 480))
    results = {}

    # 1、SuperPoint特征检测
    print("测试SuperPoint特征检测")
    try:
        detector = SuperPointDetector()
        start_time = time.time()
        keypoints,descriptors = detector.detect_and_compute(img)
        results['SuperPoint'] = {
            'time' : time.time() - start_time,
            'features' : len(keypoints)
        }
        print(f"  ✓ 检测到 {len(keypoints)} 个特征点")
    except Exception as e:
        print(f"  ✗ SuperPoint检测失败: {e}")

    # 2、SOLD2线特征检测
    print("测试SOLD2线特征检测")
    try:
        line_detector = SOLD2LineDetector()
        start_time = time.time()
        lines,descriptors = line_detector.detect_lines(img)
        results['SOLD2'] = {
            'time' : time.time() - start_time,
            'lines' : len(lines)
        }
        print(f"  ✓ 检测到 {len(lines)} 条线段")
    except Exception as e:
        print(f"  ✗ SOLD2检测失败: {e}")

    # 3、LSD线特征检测
    print("测试LSD线特征检测")
    try:
        lsd_detector = LSDLineDetector()
        start_time = time.time()
        lines_lsd = lsd_detector.detect_lines(img)
        results['LSD'] = {
            'time' : time.time() - start_time,
            'lines' : len(lines_lsd)
        }
        print(f"  ✓ 检测到 {len(lines_lsd)} 条线段")
    except Exception as e:
        print(f"  ✗ LSD检测失败: {e}")

    # 打印结果
    print("\n" + "="*60)
    print("性能测试结果:")
    print("="*60)

    for name, data in results.items():
        feature_count = data.get('features', data.get('lines', 0))
        print(f"{name:15s}: {data['time']*1000:6.1f}ms, "
              f"特征数: {feature_count:4d}")
    print("="*60)
    
    return results

def benchmark_matching():
    '''测试特征匹配性能'''
    import sys
    sys.path.insert(0, '/home/sunteng/Desktop/HighTorque_vision/LineSFM')
    from LSD_SFM_BA import IncrementalSFMPipeline

    img1 = cv2.imread('images/3.png')
    img2 = cv2.imread('images/4.png')
    if img1 is None or img2 is None:
        print("无法加载测试图像")
        return  
    
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))

    # 创建SFM实例
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    sfm = IncrementalSFMPipeline(K)

    print("\n测试特征匹配性能")
    start = time.time()

    # 添加图像并匹配
    sfm.add_image(img1,0)
    sfm.add_image(img2,1)
    point_matches, line_matches = sfm.match_images(0,1)

    match_time = time.time() - start

    print(f"  匹配时间: {match_time*1000:.1f}ms")
    print(f"  点匹配数: {len(point_matches)}")
    print(f"  线匹配数: {len(line_matches)}")

    return {
        'time': match_time,
        'point_matches': len(point_matches),
        'line_matches': len(line_matches)
    }

if __name__ == "__main__":
    print("="*60)
    print("PLATO 性能基准测试")
    print("="*60)
    
    # 特征检测性能
    detection_results = benchmark_feature_detection()
    
    # 匹配性能
    matching_results = benchmark_matching()
    
    print("\n✓ 性能测试完成")