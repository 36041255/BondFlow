# BondFlow - 蛋白质设计采样工具

BondFlow 是一个基于流匹配(**Flow Matching**)模型的蛋白质设计工具，具备生成多种共价键(**Bond**)连接的能力。支持从先验分布或部分加噪的结构进行蛋白质生成，并提供多种引导（guidance）机制来控制生成过程。

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [使用 sample.py 脚本](#使用-samplepy-脚本)
- [配置文件说明](#配置文件说明)
- [Guidance 使用指南](#guidance-使用指南)
- [示例](#示例)

## 安装

### 环境配置

进入 `env` 目录并运行安装脚本：

```bash
cd env
bash Install.sh
```

安装脚本会自动：
1. 创建 conda 环境（基于 `env.yml`）
2. 安装 PyTorch 及相关依赖
3. 安装项目依赖包
4. 安装 Flash Attention 和 FAESM
5. 安装 torch-scatter

### PyTorch 版本调整

**重要**：根据不同的 CUDA 环境和驱动版本，需要调整 PyTorch 版本。

当前 `Install.sh` 默认安装 PyTorch 2.6.0（CUDA 12.4 版本）。如果您的系统使用不同的 CUDA 版本，请修改 `Install.sh` 中的 PyTorch 安装命令：

```bash
# 对于 CUDA 12.4（默认）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

同时需要相应调整 `torch-scatter` 的安装命令中的 CUDA 版本标识（如 `torch-2.6.0+cu118`、`torch-2.6.0+cu121` 等）。

查看系统 CUDA 版本：
```bash
nvcc --version
# 或
nvidia-smi
```

更多 PyTorch 版本和 CUDA 版本对应关系，请参考 [PyTorch 官方文档](https://pytorch.org/get-started/locally/)。

## 快速开始

### 基本用法

```bash
python BondFlow/sample.py --cfg path/to/config.yaml --device cuda:0
```

### 多GPU并行采样

```bash
python BondFlow/sample.py --cfg path/to/config.yaml --device cuda:0,cuda:1,cuda:2
```

## 使用 sample.py 脚本

### 命令行参数

- `--cfg`: 配置文件路径（YAML格式），必需参数
- `--device`: 设备选择，可选值：
  - `auto`: 自动选择（优先使用CUDA，否则使用CPU）
  - `cpu`: 使用CPU
  - `cuda`: 使用默认GPU（GPU 0）
  - `cuda:0`, `cuda:1`, ...: 使用指定GPU
  - `cuda:0,cuda:1,cuda:2`: 多GPU模式，使用逗号分隔的GPU列表

### 工作模式

脚本支持两种主要采样模式：

1. **完整扩散（Full Diffusion）**: 从先验分布生成全新结构
   - 设置 `design_config.use_partial_diffusion: false`
   - 可以指定 `contigs` 来定义序列片段

2. **部分扩散（Partial Diffusion）**: 基于输入PDB结构进行部分生成
   - 设置 `design_config.use_partial_diffusion: true`
   - 需要提供 `design_config.input_pdb` 路径
   - 使用 `design_config.partial_t` 控制噪声水平

### 输出

- 生成的PDB文件保存在 `inference.output_prefix` 指定的目录
- 文件命名格式：`{output_prefix}_{index}.pdb`，其中 `index` 从0开始递增
- 如果启用 `write_trajectory`，会保存采样轨迹

## 配置文件说明

配置文件采用YAML格式，主要包含以下部分：

### 1. model 配置

```yaml
model:
  type: "apm_backbone"              # 模型类型
  sidechain_model_type: null        # 侧链模型类型（可选）
  model_config_path: null           # 模型配置文件路径（可选）
```

### 2. inference 配置

```yaml
inference:
  num_designs: 32                   # 每个批次的样本数
  num_cycle: 200                    # 总批次数（总样本数 = num_designs × num_cycle）
  output_prefix: /path/to/output    # 输出文件前缀（目录会自动创建）
  write_trajectory: false           # 是否写入采样轨迹
  inference_step: 400               # 采样时间步数
  seed: null                        # 随机种子（null表示随机）
```

### 3. design_config 配置

#### 基本设置

```yaml
design_config:
  use_partial_diffusion: false      # 是否使用部分扩散
  input_pdb: /path/to/input.pdb     # 输入PDB文件路径（.pdb或.cif格式）
  length: null                      # 序列长度（仅在纯先验模式下使用）
  partial_t: null                   # 部分扩散的噪声水平（0-1之间）
```

#### contigs 配置

`contigs` 用于定义序列片段和固定区域。**每个外层列表项对应一个链**，支持多链设计。在YAML中，可以使用两种等价格式：

**格式1：列表格式（多行，推荐用于复杂配置）**
```yaml
contigs:
  - [New_30-30]                     # 链A：生成30个残基的新序列
  - [Chain_B:seq_FIX:str_FIX]       # 链B：固定输入PDB中Chain_B的序列和结构
```

**格式2：双括号格式**
```yaml
contigs: [[New_30-30],[Chain_A:seq_FIX:str_FIX]]  # 链A：新序列30残基；链B：固定Chain_A
```

**更多示例**：

```yaml
# 单链示例
contigs:
  - [New_30-30]                     # 链A：生成30个残基的新序列

# 单链内混合多个片段：一个链可以包含多个片段
contigs:
  - [New_10-10, Chain_A:seq_FIX:str_DNV, New_5-5]  # 链A：新序列10残基 + 固定Chain_A序列 + 新序列5残基

# 多链复杂示例
contigs:
  - [Chain_A:seq_FIX:str_FIX]                    # 链A：固定Chain_A
  - [New_30-40, B100-B120:seq_FIX:str_DNV, New_10-20, B160-B200:seq_DNV:str_FIX, 10-20, AGGGKI:seq_FIX:str_DNV]  # 链B：混合多个片段

# 或者使用双括号格式
contigs: [[Chain_A:seq_FIX:str_FIX], [New_30-40, B100-B120:seq_FIX:str_DNV, New_10-20]]

# 其他多链示例
contigs:
  - [Chain_A:seq_FIX:str_DNV]       # 链A：固定序列，结构从头生成
  - [Chain_A:seq_PNA:str_PNA]       # 链B：序列和结构都添加部分噪声
  - [A100-A200:seq_FIX:str_DNV]     # 链C：固定A链100-200残基的序列
  - [AGGGKI:seq_FIX:str_DNV]        # 链D：自定义序列，结构从头生成
```

**fix_type 说明**：
- `seq_FIX`: 固定序列
- `str_FIX`: 固定结构
- `seq_DNV`: 序列从头生成（De Novo）
- `str_DNV`: 结构从头生成
- `seq_PNA`: 序列部分噪声添加（需要设置 `partial_t`）
- `str_PNA`: 结构部分噪声添加

#### bond_condition 配置

定义键合约束条件，用于**固定该区域原有的键合值**（从输入PDB中读取的键合信息）。**注意**：这里的链ID（A, B, C...）不是输入PDB结构中的原始链ID，而是根据`contigs`配置重排后的链ID。系统会按照`contigs`中定义的顺序自动分配链ID，默认从A开始依次为A, B, C...

```yaml
bond_condition:
  # 格式: "链1|链2" 或 "链1/位置-链2/位置:键类型:FIX/PNA"
  # 链ID对应contigs中的顺序：第一个contig项=链A，第二个=链B，以此类推
  
  # 固定特定位置的键合值
  - "A/start-A/start|A/end-A/end:0:FIX"  # 固定链A的N端和C端之间的键（类型0，固定）
  
  # 固定链内部所有原有的键合值（从PDB读取）
  - "B|B"                                 # 固定链B内部所有原有的键合值
  
  # 固定链之间的所有原有键合值
  - "A|B"                                 # 固定链A和链B之间所有原有的键合值
  
  # 固定多个链之间的所有原有键合值
  - "A|B|C"                               # 固定链A、B、C之间所有原有的键合值
  
  # 固定指定残基范围的键合值
  - "A100-A200|B100-B200"                 # 固定A链100-200和B链100-200之间的原有键合值
  
  # 固定并设置特定键合值
  - "A100-A100|A200-A200:1:FIX"           # 固定A链残基100和200之间的键，值设为1
  - "A/start-A/2|A10-A10:0:PNA"           # 部分噪声添加（PNA），需要设置partial_t
```

**说明**：
- `"链1|链2"`：固定这两个链/区域之间**所有原有的键合值**（从输入PDB读取）
- `:键类型:FIX`：固定并设置键合值为指定值（0或1），FIX表示完全固定
- `:键类型:PNA`：部分噪声添加，需要设置`partial_t`参数
- 如果不指定值（如`"B|B"`），则固定该区域从PDB读取的原始键合值

#### hotspots 配置

指定热点残基位置（用于结合位点设计）。可以指定多个热点，支持不同链：

```yaml
hotspots:
  # 单链多个热点
  - "A/54"    # Chain A, residue 54
  - "A/62"
  - "A/93"
  - "A/96"
  - "A/100"
  
  # 多链热点示例
  - "A/54"    # Chain A, residue 54
  - "A/62"    # Chain A, residue 62
  - "B/20"    # Chain B, residue 20
  - "B/35"    # Chain B, residue 35
  - "C/10"    # Chain C, residue 10
```

**注意**：链ID同样对应`contigs`中重排后的链ID（A, B, C...），不是原始PDB文件的链ID。

### 4. interpolant 配置

控制扩散过程的插值器设置：

```yaml
interpolant:
  min_t: 0.01                       # 最小时间步
  sampling:
    num_timesteps: ${inference.inference_step}  # 采样时间步数（引用inference配置）
    do_sde: false                   # 是否使用SDE
    self_condition: False           # 是否使用自条件
  # 各模态的配置
  rots:
    corrupt: true                   # 是否对旋转添加噪声
    train_schedule: linear          # 训练调度
    sample_schedule: exp             # 采样调度
  trans:
    corrupt: true
    sample_temp: 1.0                # 采样温度
  aatypes:
    corrupt: true
    schedule: linear
    temp: 0.1                       # 温度参数
  ss:
    corrupt: True
    prior_mix_lambda: 1              # 先验混合系数（0=单位阵，1=随机）
```

### 5. preprocess 配置

```yaml
preprocess:
  sigma_perturb: 0                  # 坐标扰动标准差
  rotation_perturb: False           # 是否旋转扰动
  link_config: /path/to/link.csv   # 键合规则配置文件
  allatom_backend: apm              # 全原子后端（apm 或 rfdiff）
```

### 6. guidance 配置

详见 [Guidance 使用指南](#guidance-使用指南) 部分。

## Guidance 使用指南

Guidance 是在采样过程中对模型输出进行引导的机制，可以在不修改模型参数的情况下影响生成结果。

### Guidance 配置结构

```yaml
guidance:
  list:
    - name: guidance_name           # Guidance类型名称
      # ... 具体参数
    - name: another_guidance
      # ... 具体参数
```

### 可用的 Guidance 类型

#### 1. LogitsBiasGuidance

对氨基酸logits添加偏置，用于偏好或抑制特定氨基酸：

```yaml
- name: logits_bias
  bias: [0.0, 0.0, 0.0, 0.0, -10.0, ...]  # 20个氨基酸的偏置值
                                        # 顺序: ALA,ARG,ASN,ASP,CYS,GLN,GLU,GLY,HIS,ILE,
                                        #       LEU,LYS,MET,PHE,PRO,SER,THR,TRP,TYR,VAL
  weight: 20.0                          # 偏置权重
  schedule: linear                     # 时间调度: linear/quadratic/cosine/exp/inverse
  power: 1.0                           # exp调度时的指数
  positions: [3, 9, 13]               # 应用位置（0-based索引，可选）
  positions_mode: include              # include: 仅这些位置; exclude: 除这些位置外
                                        # 默认（不指定positions）: 全部位置
  ignore_seq_mask: false               # 是否忽略seq_mask
```

**示例**：抑制特定位置之外的CYS

```yaml
- name: logits_bias
  bias: [0.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # CYS偏置-10（抑制）
  weight: 20.0
  schedule: linear
  positions: [3, 9, 13, 21, 26, 28]   # 这些位置允许CYS
  positions_mode: exclude              # exclude模式：除了这些位置，其他位置都抑制CYS
```

**说明**：上述配置的效果是：在位置3, 9, 13, 21, 26, 28**之外**的所有位置抑制CYS（因为这些位置需要形成二硫键，所以允许CYS）。

#### 2. TransAnchorGuidance

将预测的平移向量向锚点（如trans_1）拉近：

```yaml
- name: trans_anchor
  weight: 1.0                          # 权重
  schedule: linear                     # 时间调度
  power: 1.0                           # exp调度时的指数
  anchor_key: trans_1                  # 锚点键名
```

#### 3. SingleBondGuidance

强制形成单个键（用于环化等场景）：

```yaml
- name: single_bond
  mode: at_least_one                   # at_least_one 或 exactly_one
  threshold: 0.5                       # 判断"键"的最小值
  target_value: 1.0                    # 目标键值
  self_weight: 1.0                     # 对角线权重（0表示清除对角线）
  weight: 1.0                          # 全局权重
  schedule: linear                     # 时间调度
  power: 1.0
```

#### 4. SoftBondCountGuidance

通过软计数约束键的数量：

```yaml
- name: soft_bond_count
  mode: exact_N                         # exact_N / exact_N_topk / at_least_N
  target_N: 1                          # 目标键数量
  alpha: 20.0                           # sigmoid锐度
  tau: 0.5                              # 计数阈值
  weight: 0.1                           # 步长（原eta）
  n_steps: 1                            # 内部梯度下降步数
  sinkhorn_iters: 30                    # Sinkhorn迭代次数
  eps: 1e-8                             # 数值稳定性参数
  top_k_soft: 0                         # 如果>0，只计算top-k个最大概率值
```

#### 5. TypeAwareSoftBondCountGuidance

**类型感知的键计数引导**，支持不同键类型（二硫键、异肽键、内酯键等）：

```yaml
- name: type_soft_bond_count
  link_csv_path: ${preprocess.link_config}  # 键合规则文件路径
  bond_step: 0.1                       # 键矩阵更新步长（null表示硬固定）
  seq_step: 0.05                       # 序列logits更新步长（null表示直接赋值）
  n_steps: 1                           # 内部优化步数
  schedule: linear                      # 时间调度
  power: 1.0                           # exp调度时的指数
  stage: pre_model                     # 应用阶段: pre_model / post_step / both
  tau: 0.5                             # Sinkhorn温度参数
  types:                               # 键类型配置列表
    - name: disulfide                  # 二硫键
      mode: exact_N                    # 模式: exact_N / exact_N_topk / at_least_N / fixed_pairs / only_fixed_pairs
      target_N: 1.0                    # 目标键数量
      weight: 1.0                      # 该类型权重
      loss_type: kl                    # 损失类型: kl / mse
      top_k_soft: 0                    # top-k软计数（可选）
      other_weight: 1.0                 # exact_N_topk模式中其他位置的权重
      suppress_other_aa: false         # 是否在非topK位置抑制能形成该键的残基类型
      suppress_other_aa_weight: 1.0    # 抑制权重
      pairs:                            # fixed_pairs/only_fixed_pairs模式下的固定对
        - [3, 21]                      # 残基3和21形成键（0-based索引）
        - [9, 26]
    
    - name: isopeptide                 # 异肽键
      mode: exact_N
      target_N: 0.0                    # 目标：0个异肽键（抑制）
      weight: 5.0
    
    - name: lactone                    # 内酯键
      mode: exact_N
      target_N: 0.0
      weight: 5.0
    
    - name: covalent                   # 泛共价键（所有侧链-侧链规则）
      mode: fixed_pairs
      target_N: 1.0
      weight: 1.0
      pairs:
        - [0, 15]                      # 残基0和15形成共价键
```

**模式说明**：
- `exact_N`: 鼓励精确N个键（使用KL或MSE损失）
- `exact_N_topk`: 通过抬高前N对、压低其他位置实现精确N个键
- `at_least_N`: 鼓励至少N个键
- `fixed_pairs`: 偏好指定位置对，但不禁止其他位置
- `only_fixed_pairs`: 强制键只在指定位置对出现，其他位置强烈抑制

**示例**：设计含3个二硫键的环肽

```yaml
- name: type_soft_bond_count
  link_csv_path: ${preprocess.link_config}
  bond_step: null                      # 硬固定键矩阵
  seq_step: null                       # 直接赋值序列
  stage: post_step
  types:
    - name: disulfide
      mode: only_fixed_pairs
      target_N: 3
      weight: 1.0
      pairs:
        - [3, 21]
        - [9, 26]
        - [13, 28]
```

#### 6. ClashGuidance

减少结构冲突（使用OpenFoldClashLoss）：

```yaml
- name: clash
  link_csv_path: ${preprocess.link_config}
  start_t: 0.5                         # 开始应用的时间点（t >= start_t时生效）
  weight: 5                            # 权重
  n_steps: 2                           # 优化步数
  stage: post_step                     # 应用阶段: pre_model / post_step / both
```

### Guidance 时间调度

所有支持时间调度的guidance都支持以下调度方式：

- `linear`: 线性调度，权重 = base × (1 - t)
- `quadratic`: 二次调度，权重 = base × (1 - t)²
- `cosine`: 余弦调度，权重 = base × (1 - cos(π × (1 - t)))/2
- `exp`: 指数调度，权重 = base × (1 - t)^power
- `inverse`: 反比例调度，权重 = base / (1 - t + ε)

其中 `t` 是当前时间步（0到1之间），`base` 是基础权重。

### Guidance 应用阶段

某些guidance支持在不同阶段应用：

- `pre_model`: 在模型前向传播后、插值器之前应用
- `post_step`: 在插值器采样步骤后应用
- `both`: 在两个阶段都应用

## 示例

### 示例1：从头设计环肽

```yaml
model:
  type: "apm_backbone"

inference:
  num_designs: 32
  num_cycle: 10
  output_prefix: ./outputs/cyclic_peptide
  write_trajectory: false
  inference_step: 400

design_config:
  use_partial_diffusion: false
  length: 30
  bond_condition: ["A/start-A/end:0:FIX"]  # N端和C端形成键

guidance:
  list:
    - name: type_soft_bond_count
      link_csv_path: ./config/link.csv
      bond_step: null
      seq_step: null
      stage: post_step
      types:
        - name: disulfide
          mode: only_fixed_pairs
          target_N: 1
          weight: 1.0
          pairs:
            - [5, 25]
```

运行：
```bash
python BondFlow/sample.py --cfg config.yaml --device cuda:0
```

### 示例2：基于模板的部分设计

```yaml
model:
  type: "apm_backbone"

inference:
  num_designs: 16
  num_cycle: 50
  output_prefix: ./outputs/partial_design
  inference_step: 400

design_config:
  use_partial_diffusion: true
  input_pdb: ./templates/template.pdb
  partial_t: 0.3
  contigs: [[Chain_A:seq_FIX:str_FIX],[New_10-20]]
    # 链A：固定核心区域
    # 链B：设计新区域

guidance:
  list:
    - name: logits_bias
      bias: [0.0, 0.0, 0.0, 0.0, 5.0, ...]  # 偏好CYS
      weight: 10.0
      schedule: linear
```

### 示例3：多链设计

设计包含多个链的复合物，链之间可以形成键合：

```yaml
model:
  type: "apm_backbone"

inference:
  num_designs: 32
  num_cycle: 50
  output_prefix: ./outputs/multichain
  inference_step: 400

design_config:
  use_partial_diffusion: false
  contigs: [[New_50-50],[Chain_B:seq_FIX:str_FIX],[New_30-30]]
    # 链A：生成50个残基
    # 链B：固定输入PDB中Chain_B的序列和结构
    # 链C：生成30个残基
  bond_condition:
    - "A|B"                           # 链A和链B之间允许键合
    - "A|C"                           # 链A和链C之间允许键合
    - "B|C"                           # 链B和链C之间允许键合
  hotspots:
    - "A/20"                          # 链A的残基20
    - "A/35"                          # 链A的残基35
    - "B/10"                          # 链B的残基10
    - "C/15"                          # 链C的残基15

guidance:
  list:
    - name: type_soft_bond_count
      link_csv_path: ./config/link.csv
      stage: post_step
      types:
        - name: disulfide
          mode: exact_N
          target_N: 2
          weight: 1.0
```

**说明**：
- `contigs`中的三个列表项分别对应链A、链B、链C
- `bond_condition`中的链ID（A, B, C）对应`contigs`的顺序
- `hotspots`中的链ID同样对应重排后的链ID

### 示例4：多GPU并行采样

```yaml
inference:
  num_designs: 32
  num_cycle: 1000  # 总共32000个样本
  output_prefix: ./outputs/large_scale
```

运行：
```bash
python BondFlow/sample.py --cfg config.yaml --device cuda:0,cuda:1,cuda:2,cuda:3
```

系统会自动将1000个批次分配到4个GPU上并行处理。

## 注意事项

1. **内存管理**: 大批次采样会占用大量GPU内存，建议根据GPU容量调整 `num_designs`
2. **键合规则**: 确保 `link_config` 文件路径正确，该文件定义了各种键类型的规则
3. **索引基准**: Guidance中的位置索引默认是0-based，注意与PDB文件中的1-based索引区分
4. **部分扩散**: 使用部分扩散时，`partial_t` 值越大，添加的噪声越多
5. **Guidance权重**: 过大的guidance权重可能导致采样不稳定，建议从小值开始调试

## 故障排除

### 常见错误

1. **CUDA out of memory**: 减小 `num_designs` 或使用更多GPU
2. **配置文件解析错误**: 检查YAML语法，注意缩进和引号
3. **PDB文件读取失败**: 检查文件路径和格式（支持.pdb和.cif）
4. **Guidance未生效**: 检查guidance配置是否正确，查看日志输出

### 调试建议

- 启用 `write_trajectory: true` 查看采样过程
- 使用较小的 `num_designs` 和 `num_cycle` 进行测试
- 检查guidance的打印输出（如能量值、梯度等）
- 使用 `--device cpu` 进行CPU模式测试（较慢但便于调试）

## 许可证

[根据项目实际情况填写]

## 引用

[根据项目实际情况填写]
