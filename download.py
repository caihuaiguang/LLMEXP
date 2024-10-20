import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # 这个镜像网站可能也可以换掉
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co/'   # 这个镜像网站可能也可以换掉

from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
                  local_dir_use_symlinks=False,
                  local_dir="models/Qwen/Qwen2.5-Math-1.5B-Instruct")
snapshot_download(repo_id="Qwen/Qwen2.5-Math-7B-Instruct",
                  local_dir_use_symlinks=False,
                  local_dir="models/Qwen/Qwen2.5-Math-7B-Instruct")
snapshot_download(repo_id="peiyi9979/mistral-7b-sft",
                  local_dir_use_symlinks=False,
                  local_dir="models/peiyi9979/mistral-7b-sft")
snapshot_download(repo_id="peiyi9979/math-shepherd-mistral-7b-prm",
                  local_dir_use_symlinks=False,
                  local_dir="models/peiyi9979/math-shepherd-mistral-7b-prm")
snapshot_download(repo_id="Qwen/Qwen2.5-Math-RM-72B",
                  local_dir_use_symlinks=False,
                  local_dir="models/Qwen/Qwen2.5-Math-RM-72B")
