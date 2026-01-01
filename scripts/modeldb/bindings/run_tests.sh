#!/bin/bash

echo "🚀 Flash Decoding Allocated Buffer 测试自动化脚本"
echo "================================================="
export CUDA_VISIBLE_DEVICES=0
# 检查当前目录
if [ ! -f "setup.py" ]; then
    echo "❌ 错误: 请在bindings目录下运行此脚本"
    echo "   当前目录: $(pwd)"
    echo "   期望目录: scripts/modeldb/bindings"
    exit 1
fi

echo "📂 当前目录: $(pwd)"

echo ""
read -p "🤔 是否清理之前的编译文件？(y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 清理之前的编译文件
    echo "🧹 清理旧的编译文件..."
    rm -rf build/ dist/ bindings.egg-info/
    rm -f bindings.*.so

    # 编译模块
    echo "🔨 编译bindings模块..."
    python setup.py install

    if [ $? -ne 0 ]; then
        echo "❌ 编译失败！请检查CUDA环境和依赖项"
        exit 1
    fi

    echo "✅ 编译成功！"
else
    echo "⏭️  跳过清理"
fi



# 运行简单测试
echo ""
echo "🧪 运行简单测试..."
echo "-----------------------"
python test_simple.py

if [ $? -ne 0 ]; then
    echo "❌ 简单测试失败！"
    exit 1
fi

# 询问是否运行时间测试
echo ""
read -p "🤔 是否运行时间测试？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔬 运行时间测试..."
    echo "------------------------"
    python test_time_simple.py
    
    if [ $? -ne 0 ]; then
        echo "❌ 时间测试失败！"
        exit 1
    fi
else
    echo "⏭️  跳过时间测试"
fi

echo ""
echo "🎉 所有测试完成！"
echo "📋 可用的测试脚本:"
echo "   - test_simple.py: 快速功能验证"  
echo "   - test_time_simple.py: 时间测试"
