import tensorflow_datasets as tfds





# # 修改为父目录路径
# data_dir = "/root/workspace/data/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_10_no_noops"

# # 加载时自动识别最新版本
# dataset = tfds.load(
#     "libero_10_no_noops", 
#     data_dir=data_dir,
#     split="train"
# )

# 保留原路径结构但调整层级
data_dir = "/root/workspace/data/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"
version = "1.0.0"

dataset = tfds.load(
    f"libero_10_no_noops:{version}", 
    data_dir=data_dir,
    # split="train"
)

