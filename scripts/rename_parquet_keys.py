import os
import pandas as pd

# 设置包含 Parquet 文件的文件夹路径
folder_path = "/home/ps/0324/lerobot/BaseM/data2/chunk-000"  # 请将此路径替换为实际文件夹路径
output_folder_path = "/home/ps/0324/lerobot/BaseM/data/chunk-000"  # 请将此路径替换为输出文件夹路径

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 遍历文件夹中的所有 Parquet 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".parquet"):
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # 读取 Parquet 文件
            df = pd.read_parquet(file_path)

            # 重命名列
            if 'actions' in df.columns:
                df.rename(columns={'actions': 'action'}, inplace=True)
                print(f"文件 {file_name} 的列名 'actions' 已更名为 'action'.")
            else:
                print(f"文件 {file_name} 未找到 'actions' 列.")

            # 保存修改后的文件到新的文件夹，文件名不变
            output_file_path = os.path.join(output_folder_path, file_name)
            df.to_parquet(output_file_path, index=False)
            print(f"文件 {file_name} 已保存为 {output_file_path}")
        
        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {e}")