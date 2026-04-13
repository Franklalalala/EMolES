import os
import shutil
from pathlib import Path

# --- 1. 定义源文件夹和目标文件夹 ---
# 注意：在 Windows 路径中使用 r"..." (原始字符串) 是个好习惯，可以避免反斜杠问题
source_directory = Path(r"E:\thu\hetero_atoms_workspace")
destination_directory = Path(r"E:\github_local\EMolES\src\emoles\build\paper")

# --- 2. 确保目标文件夹存在 ---
# 如果目标文件夹不存在，则创建它 (包括任何必要的父目录)
destination_directory.mkdir(parents=True, exist_ok=True)
print(f"源文件夹: {source_directory}")
print(f"目标文件夹: {destination_directory}")
print("-" * 20)

# 计数器，用于统计复制的文件数量
copied_files_count = 0

# --- 3. 递归遍历源文件夹，查找并复制 PDF 文件 ---
# Path.rglob('*.pdf') 会递归地查找所有匹配 '*.pdf' 的文件
print("开始查找并复制 PDF 文件...")
for pdf_file_path in source_directory.rglob("*.pdf"):
    try:
        # 构建完整的目标文件路径
        # pdf_file_path.name 只获取文件名部分 (例如 'document.pdf')
        destination_file_path = destination_directory / pdf_file_path.name
        
        print(f"正在复制: {pdf_file_path.name}")
        print(f"  从 -> {pdf_file_path}")
        print(f"  到 -> {destination_file_path}")
        
        # 复制文件。shutil.copy2 会复制文件内容和元数据(如修改时间)
        # 它会自动覆盖已存在的文件，这符合您的要求
        shutil.copy2(pdf_file_path, destination_file_path)
        
        copied_files_count += 1
        
    except Exception as e:
        print(f"复制文件 {pdf_file_path.name} 时出错: {e}")

print("-" * 20)
print(f"操作完成！总共复制了 {copied_files_count} 个 PDF 文件。")