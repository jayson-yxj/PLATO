import pytest
import cv2
from plato.data.image_loader import ImageLoader

def test_image_loader():
    """测试基本加载功能"""
    loader = ImageLoader('data/test_images')
    print("测试图像加载器")
    print("空格切换")
    
    # 测试图片数量
    assert len(loader) > 0, "No images found"
    
    # 测试加载图片
    img = loader[0]
    assert img is not None
    assert len(img.shape) == 3  # H, W, C
    cv2.imshow("Test Image", img)
    cv2.waitKey(0)
    
    # 测试灰度图
    gray = loader.load_gray(0)
    assert len(gray.shape) == 2  # H, W
    cv2.imshow("Gray Image", gray)
    cv2.waitKey(0)

    # 测试调整大小
    resized = loader.resize_image(0, (100, 100))
    assert resized.shape[0] == 100 and resized.shape[1] == 100, "Resize failed"
    cv2.imshow("Resized Image", resized)
    cv2.waitKey(0)

    # 测试归一化
    normalized = loader.normalize_image(0)
    assert normalized.max() <= 1.0 and normalized.min() >= 0.0, "Normalization failed"
    cv2.imshow("Normalized Image", normalized)
    cv2.waitKey(0)

    # 测试直方图均衡化
    equalized = loader.equalize_histogram(0)
    assert len(equalized.shape) == 2, "Histogram equalization failed"
    cv2.imshow("Equalized Image", equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"✓ Loaded {len(loader)} images successfully")

if __name__ == "__main__":
    test_image_loader()