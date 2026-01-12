#!/bin/bash
set -e 
# Install environment with dependencies
conda env create -f env.yml

# Activate environment
conda activate BondFlow

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
                                            --default-timeout=1000 \
                                            --index-url https://download.pytorch.org/whl/cu124

pip install -e . --no-build-isolation

# Install Flash Attention and FAESM manually
pip install flash-attn --no-build-isolation
pip install faesm[flash_attn]

# Install torch-scatter manually
pip install torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

SITE_PKG=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cp "$SITE_PKG/faesm/esm.py" "$SITE_PKG/faesm/esm.py.bak" 2>/dev/null || true
cp ./esm.py "$SITE_PKG/faesm/esm.py"

# 3. 执行 DeepSpeed 修改 (修复 PyTorch 2.x 兼容性问题)
sed -i 's/from torch\._six import inf/from torch import inf/' "$SITE_PKG/deepspeed/runtime/utils.py"
sed -i 's/from torch\._six import inf/from torch import inf/' "$SITE_PKG/deepspeed/runtime/zero/stage_1_and_2.py"
