#!/bin/bash

# 自动同步脚本，用于自动执行git add、git commit和git push

# 检查是否有提交信息参数
if [ $# -eq 0 ]; then
    echo "请提供提交信息，例如: ./auto_sync.sh \"更新代码\""
    exit 1
fi

# 添加所有更改到暂存区
echo "执行 git add ."
git add .

# 提交更改
echo "执行 git commit -m \"$1\""
git commit -m "$1"

# 推送到远程仓库
echo "执行 git push"
git push -u origin main

echo "自动同步完成！"