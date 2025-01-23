#!/bin/bash
# 清理之前的构建
rm -rf build
rm -f glu_cuda*.so

# 使用 setup.py 构建
python setup.py build develop

# 将生成的 .so 文件复制到当前目录
cp build/lib*/glu_cuda*.so .