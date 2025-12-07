# ProtoLife – 数字原始生命实验平台

ProtoLife 旨在验证在有限二维世界中构建具备代谢、感知、行动能力的数字生命体并让其通过强化学习 / 演化形成自主策略的可行性。本仓库提供环境、策略、遗传、通信、记录等模块的初步框架，便于快速迭代实验。

## 核心设计要点
- **纯规则驱动的物理/生理法则**：环境负责移动、碰撞、能量代谢、食物/毒素交互、死亡与再生等硬规则，不写死智能策略。
- **多阶段实验开关**：通过配置切换生存、繁衍、战斗、通信、环境改造等机制，便于逐步演进实验复杂度。
- **压缩日志与回放**：地图编码为单字节单元，可在运行时或离线重放实验过程，支持地图存档导入导出。
- **GPU 优先的向量化并行**：建议在单卡 V100 上并行多个环境与个体，减少 Python 循环。

## 目录结构
```
config/
  default.yaml                # 通用默认配置，可作为基底
  phase0_survival.yaml        # 生存阶段示例（可复制到 model/<name>/<name>.yaml 使用）
model/
  demo_model/                 # 单个模型/实验的归档目录
    demo_model.yaml           # 仅此一份主配置（建议由 config/ 复制后修改）
    checkpoint/               # 训练产生的完整 checkpoint（full_step_xxx.pt）
    log/                      # 以时间戳命名的 map.log / agents.jsonl 回放日志
    config/                   # 训练时自动保存的 merge 后配置快照
protolife/
  env.py                      # 网格环境与规则实现
  encoding.py                 # 地图压缩编码
  agents.py                   # 个体状态结构与批量管理
  policy.py                   # 策略与价值网络
  rewards.py                  # 行为奖励配置
  genetics.py                 # 繁衍与变异逻辑
  communication.py            # 消息接口
  logger.py                   # 实验记录
  replay.py                   # 回放工具
  config_loader.py            # YAML 配置管理
  utils/
    cuda_utils.py
    seed_utils.py
    schedulers.py
scripts/
  train_phase0.py
  map_editor.py               # 命令行地图编辑器
maps/
  default_map.hex             # 简单示例地图（8×8 全空）
```

## 快速开始（可直接运行的最小 Demo）
> 当前仓库提供的是“可跑通流程”的最小原型：环境、策略与配置均为占位实现，但可以直接执行一轮前向推理与环境步进，用于验证依赖安装和配置是否正确。

1. **准备 Python 环境**
   - 建议 Python 3.9+，可用 `python -m venv .venv && source .venv/bin/activate` 创建虚拟环境。
   - 安装依赖（CPU 版本示例）：`pip install torch pyyaml`。
   - 若需要 GPU/CUDA，请根据显卡与 CUDA 版本替换为官方给出的 `pip install torch==<ver>+cu118 -f https://download.pytorch.org/whl/torch_stable.html`。

2. **运行基本 Demo**
   tip:请在项目根目录运行
   ```bash
   # 将示例配置复制到自己的模型目录（只需一次）
   mkdir -p model/demo && cp config/phase0_survival.yaml model/demo/demo.yaml

   # 使用该目录进行训练/回放，所有日志与权重会放在 model/demo 下
   python -m scripts.train_phase0 --config model/demo/demo.yaml --model-dir model/demo
   ```
   预期输出：
   - 打印观测张量的形状（map 与 agents）
   - 打印策略网络的 logits 形状
   - 打印一次环境步进后的平均奖励

3. **配置文件讲解与自定义**
   所有 YAML 位于 `config/`，推荐以 `config/default.yaml` 为基础：
   - `world`：地图尺寸、随机地图中食物/毒素密度、随机种子、外部地图文件（示例：`height: 64`, `width: 64`, `map_file: maps/default_map.hex`）。如果提供了 `map_file` 且文件存在，会优先加载该地图，否则回退到随机/空白地图并按 `food_density`/`toxin_density` 撒点资源。
     Windows 路径请使用正斜杠（例如 `G:/maps/demo.hex`）或在 YAML 中写成双反斜杠（`G:\\maps\\demo.hex`），避免未转义的反斜杠导致解析错误。
   - `agents`：每个环境的个体数量、初始能量等（示例：`per_env: 4`）。
   - `model`：`observation_radius` 控制感知范围（环境会裁剪周围 `(2r+1)^2` 的网格并拆成多通道输入），`hidden` 为 MLP 隐藏层规模。
   - `training`：并行环境数、回合步数、保存间隔等训练相关参数（示例：`num_envs: 8`, `save_interval: 100`, `checkpoint_dir: checkpoints/phase0`）。
     可用 `entropy_coef` 调整策略熵正则（默认提供非零值防止动作过早塌缩），`action_noise` 下的 `gaussian_std` 与 `epsilon` 分别控制 logits 高斯噪声和 epsilon-greedy 随机探索。
     若需要调试模型的动作采样，可开启 `print_actions: true` 查看每步动作编号与含义。
   - `action_rewards`：行为基础奖励，可为正/负（示例：`MOVE: 0.01`, `ATTACK: -0.01`）。
   - `rewards`：生存/进食奖励与食物感知奖励开关。
     - `enable_proximity_reward`：是否根据感知范围内最近食物给予距离衰减奖励（默认开启）。
     - `see_food_reward` / `stand_on_food_reward`：看到食物/踩在食物格时的基础奖励（默认 `0.005` / `0.02`）。
     - `vision_decay_mode` 与 `vision_decay_coefficient`：控制距离衰减；`linear` 模式在半径处衰减到 `max(0, 1 - coeff)`，`log` 模式使用 `1 / (1 + coeff * log(1+d))` 更平滑。
       例如感知半径为 3、`coeff=1.0` 时，线性衰减在距离 3 处约为 `0`，而对数衰减在距离 3 处仍保留约三分之一的奖励。
   - 阶段开关：各阶段配置文件通过布尔开关控制功能模块，例如 `use_reproduction`、`use_combat`、`use_communication`、`use_terraforming`。

   **如何自定义**：
    - 复制默认配置：`cp config/default.yaml config/my_exp.yaml`。
    - 按需修改上述字段；未修改的字段会沿用默认值。
    - 运行时指定：`python -m scripts.train_phase0 --config config/my_exp.yaml`。

4. **常见问题排查**
   - `ModuleNotFoundError: No module named 'torch'`：确认已在当前虚拟环境中执行 `pip install torch pyyaml`。
   - CUDA 未被使用：检查 `torch.cuda.is_available()` 是否为 True，如否则会回退到 CPU。

5. **自定义地图加载与编辑**
   - 在配置 `world.map_file` 中指定十六进制地图文件；确保 `height/width` 与文件中单元数量一致（每个单元两个十六进制字符，长度 = `2*H*W`）。
   - 使用命令行编辑器快速制作地图：
     ```bash
     # 从空白 16x16 地图开始编辑，保存到 maps/custom_xxx.hex
     python -m scripts.map_editor --width 16 --height 16

     # 基于已有地图微调
     python -m scripts.map_editor --input maps/default_map.hex --width 8 --height 8 --output maps/my_map.hex

     # 打开 matplotlib 图形界面，点击涂抹/按 n 切换笔刷类型
     python -m scripts.map_editor --width 16 --height 16 --gui
     ```

6. **模型与环境 checkpoint / 断点续推**
   - 推荐每个模型/实验单独建目录：`model/<model_name>/`（如上文 Demo）。目录下需要有一份 `<model_name>.yaml` 作为训练/回放唯一入口。
   - 训练脚本新增参数：
     * `--save-interval`：每隔多少步保存模型与完整存档（默认读取配置中的 `training.save_interval`）。
     * `--checkpoint-dir`：保存目录（默认指向 `model/<name>/checkpoint`）。
     * `--model-dir` / `--model-name`：统一管理模型目录，便于续训或共享。
     * `--resume-from`：从完整 checkpoint 继续推演（恢复地图、agent 状态、优化器与步数）。
     * `--load-model`：仅加载模型权重，在新地图或新实验上测试。
   - 示例：
     ```bash
     # 带定期保存（日志与 checkpoint 全部写到 model/demo）
     python -m scripts.train_phase0 --config model/demo/demo.yaml --model-dir model/demo --save-interval 50

     # 从最新 checkpoint 直接续训（自动定位 model/demo/checkpoint 下最新 full_step_*.pt）
     python -m scripts.train_phase0 --config model/demo/demo.yaml --model-dir model/demo

     # 显式指定某个存档继续
     python -m scripts.train_phase0 --config model/demo/demo.yaml --resume-from model/demo/checkpoint/full_step_200.pt
     ```

   - checkpoint 内容包括：当前地图、全部 agent 状态、策略网络参数、优化器参数与当前步数，可直接用于“继续推演”或模型回滚。

7. **实时渲染、日志与回放**
   - `logging.realtime_render: true` 时，环境会用 matplotlib 绘制首个并行环境的地图与 agent 位置（参考 `map_editor` 的配色，需安装 matplotlib）。
    - `logging.save_dir` 与 `logging.snapshot_interval` 控制回放日志写入；运行结束后可用 `protolife.replay.playback(log_dir, height, width)` 或脚本 `python -m scripts.visualize_replay --log-dir <dir> --height <H> --width <W>` 在 Python 交互式环境中快速查看轨迹。
    - 每次运行会按照时间戳生成 `<tag>_map.log` 与 `<tag>_agents.jsonl`（第一行包含元数据）：`map.log` 记录压缩地图快照，`agents.jsonl` 记录每个 step 的个体状态摘要。需要回放时可直接将 `--log-dir` 指向 `model/<name>/log`，脚本会自动匹配最新的成对日志。
    - 回放：
      1. 进入对应的模型目录（例如 `model/demo/`），确认 `<model_name>.yaml`、`checkpoint/`、`log/` 存在。
      2. 回放：在'项目根目录'执行
         ```bash
         python -m scripts.visualize_replay --log-target model/test 
         ```

         或在代码中调用 `protolife.replay.playback`。

本 README 为概览，详细设计思路请参考源码中的中文注释，后续可在此基础上逐步补全能量代谢、战斗、通信等真实逻辑。
