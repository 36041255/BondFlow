# =====================================================
# 🌌 Nihilism Project Environment Setup Script
# 功能：自动创建环境、安装依赖、配置 PyRosetta 并验证
# 作者：Tommy
# =====================================================
set -e  # 一旦出错立即停止执行
echo " Setting up environment for Nihilism Project..."
# ---------------------------
# 1️⃣ 创建并激活 Conda 环境
# ---------------------------
if ! conda info --envs | grep -q "analysis"; then
    echo "Creating conda environment: analysis"
    conda create -n analysis python=3.10 -y
else
    echo "Environment 'analysis' already exists, skipping creation."
fi
# ✅ 激活环境
# 注意：在非交互式 shell 中使用 conda activate 需先 source
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate analysis
echo " Conda environment 'analysis' activated!"
# ---------------------------
# 2️⃣ 安装 Python 依赖
# ---------------------------
if [ -f requirements.txt ]; then
    echo " Installing dependencies from requirements.txt..."
    pip install -r requirements.txt  # -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo " requirements.txt not found! Skipping..."
fi
# ---------------------------
# 3️⃣ 安装 PyRosetta
# ---------------------------
echo "🧬 Installing PyRosetta via pyrosetta-installer..."
python - <<'EOF'
import pyrosetta_installer
pyrosetta_installer.install_pyrosetta()
EOF
echo "PyRosetta installation completed!"
# ---------------------------
# 4️⃣ 验证 PyRosetta 是否安装成功
# ---------------------------
echo "Verifying PyRosetta installation..."
python - <<'EOF'
try:
    import pyrosetta
    pyrosetta.init("-mute all")
    print("PyRosetta successfully initialized!")
except Exception as e:
    print("PyRosetta installation failed:", e)
EOF
# ---------------------------
# 5️⃣ 环境检查总结
# ---------------------------
echo "------------------------------------------------------"
echo "Environment setup completed successfully!"
echo " Python version: $(python --version)"
echo " Installed packages:"
pip list | grep pyrosetta
echo "------------------------------------------------------"
echo " Environment 'analysis' is fully equipped for Nihilism Project!"