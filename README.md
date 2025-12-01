# ProtoLife – 数字原始生命实验平台

ProtoLife 旨在验证在有限二维世界中构建具备代谢、感知、行动能力的数字生命体并让其通过强化学习 / 演化形成自主策略的可行性。本仓库提供环境、策略、遗传、通信、记录等模块的初步框架，便于快速迭代实验。

## 核心设计要点
- **纯规则驱动的物理/生理法则**：环境负责移动、碰撞、能量代谢、食物/毒素交互、死亡与再生等硬规则，不写死智能策略。
- **多阶段实验开关**：通过配置切换生存、繁衍、战斗、通信、环境改造等机制，便于逐步演进实验复杂度。
- **压缩日志与回放**：地图编码为单字节单元，可在运行时或离线重放实验过程。
- **GPU 优先的向量化并行**：建议在单卡 V100 上并行多个环境与个体，减少 Python 循环。

## 目录结构
```
config/
  default.yaml              # 通用默认配置
  phase0_survival.yaml      # 生存阶段示例
  phase1_reproduction.yaml  # 繁衍阶段示例
  phase2_combat.yaml        # 战斗阶段示例
  phase3_communication.yaml # 通信阶段示例
  phase4_terraforming.yaml  # 环境改造阶段示例
protolife/
  env.py                    # 网格环境与规则实现
  encoding.py               # 地图压缩编码
  agents.py                 # 个体状态结构与批量管理
  policy.py                 # 策略与价值网络
  rewards.py                # 行为奖励配置
  genetics.py               # 繁衍与变异逻辑
  communication.py          # 消息接口
  logger.py                 # 实验记录
  replay.py                 # 回放工具
  config_loader.py          # YAML 配置管理
  utils/
    cuda_utils.py
    seed_utils.py
    schedulers.py
scripts/
  train_phase0.py
```

## 快速开始
1. 创建并激活 Python 3 环境，安装 PyTorch 等依赖。
2. 根据目标阶段选择或自定义配置文件，示例：`python scripts/train_phase0.py --config config/phase0_survival.yaml`。
3. 在 `config` 中调整地图大小、奖励 shaping、是否启用繁衍/通信/战斗等开关，配合日志与回放观察演化行为。

本 README 为概览，详细设计思路请参考源码中的中文注释。
