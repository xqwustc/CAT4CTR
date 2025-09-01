#!/bin/bash

# 1. 打印脚本的PID
echo "Script running with PID: $$"

# 2. 获取用户输入的目标目录
read -p "Please enter the destination directory name: " DEST_DIR

# 检查输入是否为空
if [ -z "$DEST_DIR" ]; then
    echo "Error: Destination directory name cannot be empty."
    exit 1
fi

# 替换成你的 Git 仓库 URL
REPO_URL="https://github.com/xqwustc/CAT4CTR.git"

echo "--- Script Initialized ---"

# 无限循环
while true
do
  # 3. 等待用户按回车键
  read -p "Press [Enter] to forcefully overwrite '$DEST_DIR', or Ctrl+C to exit..."

  # 检查目标目录是否存在
  if [ -d "$DEST_DIR" ]; then
    echo "Directory exists. Forcefully overwriting with the remote branch..."
    
    # 使用子 shell 执行操作，更安全
    # 这会放弃所有本地修改、提交，并删除未跟踪的文件和目录
    (
      cd "$DEST_DIR" && \
      git fetch origin && \
      git reset --hard origin/main
    )
    
    # 检查上一条命令是否成功
    if [ $? -eq 0 ]; then
      echo "✅ Overwrite successful."
    else
      echo "❌ An error occurred during the overwrite process."
    fi

  else
    # 如果目录不存在，就先克隆一次
    echo "Directory not found. Cloning repository for the first time..."
    git clone "$REPO_URL" "$DEST_DIR"
    if [ $? -eq 0 ]; then
      echo "✅ Clone successful."
    else
      echo "❌ An error occurred during clone."
    fi
  fi
  
  echo "----------------------------------------" # 分隔符
done