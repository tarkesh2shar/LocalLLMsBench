import os
import sys
from huggingface_hub import hf_hub_download

target_dir = os.path.expanduser("~/models/gguf")
os.makedirs(target_dir, exist_ok=True)

repo_id = "peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-GGUF-MTP"
filename = "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ4_XS.gguf"

print(f"Starting download of {filename} from {repo_id} to {target_dir}...")
sys.stdout.flush()

try:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
    )
    print(f"Successfully downloaded {filename} to {path}")
except Exception as e:
    print(f"Download failed with error: {e}")
    sys.exit(1)
