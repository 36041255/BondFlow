import sys
import os

# 将项目根目录添加到 python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from BondFlow.experiment.analysis.energy import batch_energy

if __name__ == "__main__":
    # 定义测试用例目录
    test_case_dir = os.path.join(os.path.dirname(__file__), "test_case")
    output_dir = os.path.join(os.path.dirname(__file__), "test_results")
    
    print(f"开始测试目录: {test_case_dir}")
    
    # 运行批量能量计算
    try:
        df = batch_energy(
            pdb_folder=test_case_dir,
            output_dir=output_dir,
            num_workers=1,  # 测试时使用单进程方便调试
            relax=True,     # 开启 relax
            save_results=True,
            # ===== PNear (环肽构象熵估计) =====
            compute_pnear=True,
            pnear_n=1000,          # 测试用：采样少一点，保证运行快
            pnear_kT=1.0,
            pnear_lambda=1.0,
            pnear_chain="A",     # 复合物环肽配体链默认 A；单体会忽略
            pnear_method="min",  # 测试用：min 比 fastrelax 更快
            pnear_state="unbound",  # unbound=ligand-only（更符合构象熵损失的定义）
            pnear_sampler="crankshaft",
            pnear_debug_constraints=True,  # 打印 PNear 扰动阶段约束是否生效（仅前3个+最后1个）
            pnear_debug_first_n=3,
            save_relaxed_pdb=True,
            relaxed_pdb_dir=output_dir,
            link_constraints=True,
        )
        print("\n测试完成!")
        cols = [
            "PDB",
            "Binding_Energy",
            "Total_Energy",
            "Has_Ligand",
            "Relaxed",
            "PNear",
            "PNear_n",
            "PNear_kT",
            "PNear_lambda",
            "PNear_chain",
            "PNear_method",
            "PNear_state",
            "PNear_sampler",
            "PNear_sampler_used",
            "PNear_fallback_reason",
            "PNear_error",
            "PNear_Emin",
            "PNear_Rmean",
        ]
        cols = [c for c in cols if c in df.columns]
        # 打印所有列，包括省略号部分在内
        import pandas as pd
        with pd.option_context('display.max_columns', None, 'display.width', 2000):
            print(df[cols])
    except Exception as e:
        print(f"测试失败: {e}")
