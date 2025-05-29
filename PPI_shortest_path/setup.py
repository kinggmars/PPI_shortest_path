# 包安装配置
from setuptools import setup, find_packages

setup(
    name="protein_network",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # 指定包根目录在 src 下
    install_requires=[],      # 按需添加依赖项
)
