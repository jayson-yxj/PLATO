from setuptools import setup, find_packages

setup(
    name="plato",
    version="0.1.0",
    description="Point-Line Adaptive Training Optimization for Structure from Motion",
    author="Xiaojie Yang",
    author_email="3630115255@qq.com",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'opencv-python>=4.5.0',
        'torch>=1.9.0',
    ],
    python_requires='>=3.8',
)
