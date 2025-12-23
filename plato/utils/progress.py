'''进度显示工具'''
from tqdm import tqdm

def process_with_progress(item, process_fn,desc="Processing"):
    '''带进度条的处理函数'''
    results = []
    for item in tqdm(item, desc=desc):
        results.append(process_fn(item))
    return results