from huggingface_hub import snapshot_download


# 使用cache_dir参数，将模型/数据集保存到指定“本地路径”
res = snapshot_download(repo_id="openvla/modified_libero_rlds", repo_type="dataset",
                  cache_dir="/root/workspace/data",
                  local_dir_use_symlinks=False, resume_download=True,
                  token='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',
                  local_files_only=True  #设置为True则不会下载，只会返回本地数据集的地址
                  )
print(res)