'''批量处理工具'''
from typing import List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

class BatchProcessor:
    '''批量处理器'''
    def __init__(self,num_workers):
        """
        初始化批量处理器
        
        Args:
            num_workers: 并行工作线程数
        """
        self.num_workers = num_workers

    def process(self,
                items:List[Any],
                process_fn:Callable,
                desc:str="Processing") -> List:
        """
        顺序处理（带进度条）
        
        Args:
            items: 待处理项目列表
            process_fn: 处理函数
            desc: 进度条描述
        
        Returns:
            处理结果列表
        """

        results = []
        for item in tqdm(items, desc=desc):
            try:
                result = process_fn(item)
                results.append(result)
            except Exception as e:
                print(f"处理项目时出错: {e}")
                results.append(None)
        return results
    
    def process_parallel(self,
                        items: List[Any],
                        process_fn: Callable,
                        desc: str = "Processing") -> List:
        """
        并行处理（带进度条）
        
        Args:
            items: 待处理项目列表
            process_fn: 处理函数
            desc: 进度条描述
        
        Returns:
            处理结果列表
        """
        results = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(process_fn, item): i 
                for i, item in enumerate(items)
            }
            
            # 收集结果（带进度条）
            for future in tqdm(as_completed(future_to_idx), 
                             total=len(items), 
                             desc=desc):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"\n处理项目 {idx} 时出错: {e}")
                    results[idx] = None
        
        return results
    
    def process_images(self,
                      images: List[np.ndarray],
                      process_fn: Callable,
                      desc: str = "Processing images",
                      parallel: bool = True) -> List:
        """
        处理图像列表
        
        Args:
            images: 图像列表
            process_fn: 处理函数
            desc: 进度条描述
            parallel: 是否并行处理
        
        Returns:
            处理结果列表
        """
        if parallel and self.num_workers > 1:
            return self.process_parallel(images, process_fn, desc)
        else:
            return self.process(images, process_fn, desc)
        

# 使用示例
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from plato.data.image_loader import ImageLoader
    from plato.features.extractors.superpoint import SuperPointDetector
    from plato.features.extractors.sold2 import SOLD2Detector, LSDDetector
    
    print("="*60)
    print("批量处理示例")
    print("="*60)
    
    # 加载图像
    loader = ImageLoader('data/test_images')
    num_images = min(5, len(loader))
    images = [loader[i] for i in range(num_images)]
    print(f"\n加载了 {len(images)} 张图像")
    
    # 创建批量处理器
    processor = BatchProcessor(num_workers=2)
    
    # ========== 测试1: 点特征提取 ==========
    print("\n" + "="*60)
    print("测试1: SuperPoint 点特征提取")
    print("="*60)
    
    point_extractor = SuperPointDetector()
    point_results = processor.process_images(
        images,
        point_extractor.extract,
        desc="提取点特征",
        parallel=True
    )
    
    print(f"\n点特征提取完成:")
    for i, result in enumerate(point_results):
        if result is not None:
            kp, desc = result
            print(f"  图像 {i}: {len(kp)} 个特征点, 描述子维度: {desc.shape[1]}")
        else:
            print(f"  图像 {i}: 处理失败")
    
    # ========== 测试2: SOLD2 线特征提取 ==========
    print("\n" + "="*60)
    print("测试2: SOLD2 线特征提取")
    print("="*60)
    
    try:
        sold2_extractor = SOLD2Detector(min_length=10.0)
        sold2_results = processor.process_images(
            images,
            sold2_extractor.extract,
            desc="提取线特征(SOLD2)",
            parallel=True
        )
        
        print(f"\nSOLD2线特征提取完成:")
        for i, result in enumerate(sold2_results):
            if result is not None:
                lines, desc = result
                print(f"  图像 {i}: {len(lines)} 条线段, 描述子维度: {desc.shape[1] if len(desc) > 0 else 'N/A'}")
            else:
                print(f"  图像 {i}: 处理失败")
    except Exception as e:
        print(f"\nSOLD2不可用: {e}")
        print("跳过SOLD2测试")
    
    # ========== 测试3: LSD 线特征提取 ==========
    print("\n" + "="*60)
    print("测试3: LSD 线特征提取")
    print("="*60)
    print("注意: LSD不支持多线程，使用顺序处理")
    
    lsd_extractor = LSDDetector(min_length=20.0)
    lsd_results = processor.process_images(
        images,
        lsd_extractor.extract,
        desc="提取线特征(LSD)",
        parallel=False  # LSD不支持多线程！
    )
    
    print(f"\nLSD线特征提取完成:")
    for i, result in enumerate(lsd_results):
        if result is not None:
            lines, desc = result
            print(f"  图像 {i}: {len(lines)} 条线段, 描述子: {'无' if desc is None else desc.shape[1]}")
        else:
            print(f"  图像 {i}: 处理失败")
    
    # ========== 总结 ==========
    print("\n" + "="*60)
    print("批量处理总结")
    print("="*60)
    print(f"处理图像数: {len(images)}")
    print(f"并行线程数: {processor.num_workers}")
    print(f"\n特征提取器:")
    print(f"  - SuperPoint: 点特征 + 256维描述子")
    print(f"  - SOLD2: 线特征 + 128维描述子")
    print(f"  - LSD: 线特征（无描述子）")