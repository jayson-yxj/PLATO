#!/usr/bin/env python3
"""验证环境配置"""
import sys
import importlib

def check_package(package_name, display_name=None):
    """检查包是否可导入"""
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {display_name:20s} {version}")
        return True
    except ImportError as e:
        print(f"✗ {display_name:20s} 未安装")
        return False

def check_torch():
    """检查PyTorch和CUDA"""
    try:
        import torch
        print(f"✓ PyTorch            {torch.__version__}")
        print(f"  CUDA可用:          {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA版本:          {torch.version.cuda}")
            print(f"  GPU数量:           {torch.cuda.device_count()}")
            print(f"  当前GPU:           {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("✗ PyTorch            未安装")
        return False

def check_third_party():
    """检查第三方模块"""
    import os
    from pathlib import Path
    
    third_party = Path('third_party')
    
    modules = {
        'SuperPoint': third_party / 'SuperPointPretrainedNetwork',
        'SuperGlue': third_party / 'SuperGluePretrainedNetwork',
        'SOLD2': third_party / 'SOLD2',
    }
    
    print("\n第三方模块:")
    for name, path in modules.items():
        if path.exists():
            print(f"✓ {name:20s} {path}")
        else:
            print(f"✗ {name:20s} 未找到")

def main():
    print("=" * 60)
    print("PLATO 环境验证")
    print("=" * 60)
    
    print("\n基础包:")
    packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('cv2', 'OpenCV'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
        ('yaml', 'PyYAML'),
    ]
    
    for pkg, name in packages:
        check_package(pkg, name)
    
    print("\n深度学习:")
    check_torch()
    check_package('torchvision', 'TorchVision')
    
    print("\n可视化:")
    check_package('open3d', 'Open3D')
    
    print("\n测试工具:")
    check_package('pytest', 'pytest')
    
    check_third_party()
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()