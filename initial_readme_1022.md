# 2026-10-22 初始交接：Windows 运行，Mac 审核

这份文件是 **10 月 22 日的项目交接说明**，与根目录的导航文件 `README.md` 分开。后续实验不需要不断另建 README；在这里修正事实，在 `reports/` 提交经过整理的结果即可。

## 1. 我们要解决什么，现在做到哪里

课程 Street Fighter II championship，游戏为 Genesis 版 `StreetFighterIISpecialChampionEdition`。**只做 Ryu vs Ryu，同一份权重兼顾 PvE 1–8 星和 PvP P1/P2。** 模型名称是 Counter-Ryu。

目标是能根据局面和对手行为调整打法，在多种有效打法下保持完整对局胜率，并学会有收益的进退、距离控制和攻防转换。当前模型容易重复升龙和波动拳，尚未证明具有要求的 adaptive/robust 能力。不能用走路频率、动作熵、辅助损失或八星某次获胜替代这个目标。

当前保留模型是 `models/counter_ryu_final/agent.pt`。它是三个已学习神经策略加可学习路由/价值网络的 PPO mixture；不是一个已验证成功的 DFL、World Model 或通用人类战术模型。三个专家的权重嵌在同一个文件中，推理不再依赖 Mac 上的训练文件。迁移时只保留模型定义和权重，去掉优化器与旧训练路径；不是旧优化器的精确续训快照，权重行为用固定输入比较确认。

| 历史候选 | 同一历史开发集 PvP | P1 / P2 | PvE 每星各一局 |
|---|---:|---:|---:|
| **保留的 Counter-Ryu** | **114/160** | **59/80，55/80** | **8/8** |
| 学习动作持续时间的 PPO | 89/160 | 45/80，44/80 | 8/8 |
| 示范与反事实经验 Q 学习候选 | 83/160 | 43/80，40/80 | 8/8 |

**这些是 Mac 上已经完成的开发实验，不是 Windows 新跑结果，也不是对课程其他小组的实战统计。** 开发对手既包括规则控制器，也包括历史神经策略；多次使用的测试不再是干净留出集。完整 160 局的全部历史对手权重没有随最小包迁移，只保留其报告，不能声称此仓库能原样复现那 160 局。

本包提供可重跑的 `configs/focus_pool.json`：6 个困难 PvP 对手，P1/P2 × offset 0/12 共 **24 局**，另加 Ryu CPU 1–8 星各一局。历史参考结果是 **PvP 11/24（P1 6/12、P2 5/12），PvE 8/8**。这是困难开发集，不能和上表直接比较百分比。

最后两条高层 PPO 分支，在约 2.5 万次调用检查点上都为 11/24；普通分支 PvE 7/8，加入反事实预测辅助任务的分支 8/8。离线动作段预测选择器只有 2/24、PvE 7/8。**没有新模型晋级，Mac 的训练已停止。** 失败候选权重、优化器、回放库和大录像未迁移；保留源码是为了审计，不代表推荐续跑。

## 2. 最重要的约束

先读根目录 [AGENTS.md](AGENTS.md)。**严令禁止文件哈希/校验和，也禁止继续堆叠重复 trainer、复制 notebook 逻辑和追加一串临时包装器。** 不能为了验证文件或 ROM 而调用 SHA/MD5、`hashlib`、`Get-FileHash`、`certutil` 等，也不能运行会自动触发它们的课程原版打包/校验入口。

课程原始训练与 Arena notebook 位于 `notebooks/upstream/`，用作只读规则和接口参考。**不要直接 Run All。** 第二个大型公共代码 cell 主要是在生成共用模块，其中也包含本项目禁止执行的校验路径。迁移版已有独立运行模块，不需要再执行它来生成一整套旧 runtime。

最终 Arena 仍使用课程接口；本包的本地对局和原始权重导出不等于官方 Arena 自检。没有绕过禁令完成官方打包验收，也不要冒称完成。

课程方最新书面回复已由用户提供：学生代码允许修改，算法可以使用 PPO 以外的方法；模板的训练、打包和加载包含 PPO 专用检查，换算法时需一并适配，并在课程提供的 Arena 中验证最终提交包。算法许可与模板兼容性应分开讨论，不能把 PPO 默认实现解释成课程只允许 PPO。最终比赛形式按课程正式说明确认，本地四种覆盖条件不代表最终赛制已经确定。上述许可不取消本项目禁止文件哈希的要求；官方最终包验收仍是未完成事项。

## 3. 文件地图：先看入口，不要遍历所有历史尝试

| 位置 | 用途 |
|---|---|
| `AGENTS.md` | 两端 Codex 的强制规则 |
| `prompts/windows_initial_prompt_1022.md` | 可直接贴给 Windows Codex 的首个任务 |
| `notebooks/working/counter_ryu_windows.ipynb` | 薄 notebook：调用检查、评估、录像和可选短训练入口 |
| `notebooks/upstream/` | 两份课程原版 notebook，只读参考 |
| `scripts/check_environment.py` | 环境和硬件信息，不计算文件摘要 |
| `scripts/install_local_rom.py` | 将明确指定的本地原始 ROM 直接复制到已安装 integration |
| `scripts/check_inference.py` | 固定 80 个合法观察的动作、概率、价值比较，无需 ROM |
| `scripts/check_native.py` | 9 个 Ryu 开局与保存/恢复后的短分支重放 |
| `scripts/train_reference_ppo.py` | 一个精简的普通 PPO 续训/冒烟入口，不自动启动或晋级模型 |
| `experiments/league_adaptive/` | 模型加载、原生模拟器、双端口环境、规则对手、整局评估、录像 |
| `experiments/ryu_direct_league/` | 直接策略/mixture 定义、Ryu 环境、独立权重导出 |
| `experiments/ryu_adaptive_curriculum/` | 历史/记忆、循环 PPO、走位因果诊断等实验源码 |
| `experiments/ryu_counterq/` | 反事实经验 Q 学习实验源码，未证明优于参考模型 |
| `experiments/ryu_option_planner/` | 完整动作段预测与高层 PPO 实验源码，未证明提升 |
| `configs/` | 精确困难评估对手、精简训练池、CPU 各星级配置 |
| `models/counter_ryu_final/` | 参考权重、80 个固定输入与原预测、动作/观察定义、版本 |
| `models/opponents/` | 两个冻结的 Ryu 神经反制对手；不包含优化器 |
| `assets/starts/` | 1 个 PvP、8 个 PvE Ryu 原生开局，不含 ROM |
| `reports/migration_1022/` | 经过筛选的历史逐局结果、反事实和消融证据 |
| `course/` | `Project_List_2026.pdf` 和 `Championship_notes.pdf` |
| `dqn_papers/`、`general_papers/` | 相关论文，阅读顺序见第 9 节 |
| `artifacts/`、`local/` | 本机运行输出与本地 ROM 等，不进入 Git |

为减小迁移体积，去掉了旧环境、旧 runtime、大部分失败权重/数据集、缓存、录像和大量一次性脚本。运行模块改为相对仓库路径；规则对手只提取了原来的 `choose` 函数，避免再导入旧框架。**这是代码与基线的最小交接，不是所有旧实验的完整训练快照。**

## 4. Windows 环境：当前先核查原生构建，WSL2 保留为备选

### 当前 Windows 报告与下一步

用户转发的 Windows 诊断报告称：现有 `sye6308` 中除模拟器外的依赖已安装；80 个固定输入和逐次推理检查通过，动作差异为 0。此结果支持模型在这些观察上的迁移一致性，完整对局尚未验证。Windows 的原始日志和 `summary.json` 尚未提交 Git，本段是对转发报告的记录，不是 Mac 已读取原始日志的审核结论。

首次构建错误是 `CMAKE_MAKE_PROGRAM is not set`，即找不到配套 Make 程序，不能据此声称 CMake 不支持 Unix Makefiles 生成器。后续报告还发现 MSVC 拒绝 GCC 风格的 `-Wextra`，以及核心/LuaJIT 子项目继续依赖 GNU Make。固定版 [CMake 源码](https://github.com/Farama-Foundation/stable-retro/blob/v1.0.1/CMakeLists.txt) 中确有这两类依赖；仅把主生成器换成 Ninja 不能解决整个构建。当前停止继续修 MSVC 路线。

**当前不安装 WSL2/Ubuntu，不修改系统组件，不自动重启。** 下一步先把现有 Windows 报告提交到任务分支，然后给 Windows 原生 GNU 工具链一次最多 45 分钟的可行性验证。优先核查现有工具，确需增加时仅使用当前用户目录下的隔离工具与会话环境变量，不改全局 PATH；需要管理员权限或系统组件时停止并报告。

MSYS2 提供 Windows 上的构建工具，不能直接把它自带的 Python 当作已经装好 PyTorch 的 Conda Python。[MSYS2 官方说明](https://www.msys2.org/docs/python/)明确指出其 Python 的 C 扩展与普通 Windows CPython 不兼容；[UCRT64](https://www.msys2.org/docs/environments/) 是其当前推荐环境。因此，只将 GNU 工具链用于构建，目标保持现有 `sye6308` 的 Windows Python 3.12。第一阶段门槛是扩展能在该环境中加载，而不是在另一个 Python 中能 import。不能切换到 MSYS2 Python 并重装整套依赖来绕过这个门槛。

只有扩展加载成功，才继续检查 Genesis 核心、ROM、9 个开局恢复和完整 24+8 局基线。45 分钟内未打通就保存精确阻碍，不接着改第三方核心或重写打包系统；WSL2 仍只是待用户决定的备选。具体指令见现有 Windows prompt，不新增一套 trainer 或重复运行入口。

### WSL2 安装资料（仅在用户以后明确选择后执行）

Stable-Retro 官方文档提供 [Windows WSL2 运行说明](https://stable-retro.farama.org/linux_installation/)。采用这条备选时，使用 WSL2 Ubuntu 24.04、Python 3.12，并在 Linux 文件系统中克隆仓库。以下是备选资料，**不是现在的执行指令**。安装需用户明确授权；不得自动重启。

用户授权后，如尚未安装 WSL，可在**管理员 PowerShell**执行，再由用户决定重启与初始化 Ubuntu 的时间：

```powershell
wsl --install -d Ubuntu-24.04
```

仅对采用 WSL2 的路线，接下来全部在 **Ubuntu/WSL 终端**执行，不要把 Linux 命令贴进 PowerShell。GitHub 私有仓库需要该 Windows/WSL 会话已有的正常 Git 登录权限，不要把 token、密码或 SSH 私钥发到聊天或提交仓库。

```bash
sudo apt update
sudo apt install -y git python3.12-venv libgl1 libglib2.0-0 ffmpeg
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/oran-chengfan/SYE6308project-streetfiighter.git
cd SYE6308project-streetfiighter
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
export PYTHONUTF8=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

如果使用 Conda，在 **WSL 内的 Linux Conda** 创建 `sye6308`、Python 3.12，再执行相同的 pip 安装；Windows 的 Conda 环境不能直接作为 WSL 的 Linux 环境使用。不要复制 Mac 环境目录。固定依赖版本来自 Mac 已测配置，而不是“越新越好”。

第一轮先使用 CPU，以减少排查变量。后续如确认有 NVIDIA GPU，可按 [PyTorch 2.5.1 官方安装说明](https://pytorch.org/get-started/previous-versions/#v251)选择兼容的 CUDA wheel，但 **安装 CUDA 版不等于当前所有研究脚本自动使用 GPU**：部分脚本显式使用 CPU、NumPy 和模拟器子进程。先量速度瓶颈和固定输入差异，再决定是否改设备；不要同时修改算法和平台设置。

## 5. ROM 与最初的三项检查

游戏 ROM 没有上传。将 Mac 上已有的、课程匹配的原始文件复制到 Windows 仓库的 `local/StreetFighterIISpecialChampionEdition-Genesis.md`；如以后采用 WSL2，再确保该文件能被 Linux 环境读取。Mac 原文件是原工作目录下 `runtime/mac_ppo/resources/StreetFighterIISpecialChampionEdition-Genesis.md`，并已在 Mac 的 `github_sync/local/` 准备一份便于传输的副本，大小为 3,145,728 字节。`local/` 被 Git 忽略，**文件并未通过 GitHub 传到 Windows**，需用两台设备之间已有的文件传输方式复制。模拟器构建和 ROM 缺失是两个独立阻碍。**这是游戏二进制，虽然后缀是 `.md`，不是 Markdown 文档；不要转换内容或行尾。** `.state`、zip 和其他游戏版本不能代替。

先做无需 ROM 的检查：

```bash
python -m scripts.check_environment
python -m scripts.check_inference
```

预期固定输入比较为 80 个观察、动作差异 0，概率/价值最大绝对误差不超过 `1e-5`。这是模型功能比较，不是文件摘要验证。如果失败，先检查版本、预处理、权重加载与批量推理，不能忽略差异后直接训练。

再明确指定已有 ROM，直接复制到当前 Python 环境的 game integration：

```bash
python -m scripts.install_local_rom --rom local/StreetFighterIISpecialChampionEdition-Genesis.md
python -m scripts.check_native --out artifacts/windows_native_check.json
```

安装脚本只检查明确路径、后缀和字节数，然后复制，不扫描 ROM 目录或做身份摘要。字节数相同不能证明是相同版本，兼容性由后续原生开局恢复、读取与重复动作结果确认。脚本拒绝静默覆盖已有 ROM；如果环境已经有 ROM，直接做原生检查，遇到问题先报告安装位置和错误。

原生检查覆盖 PvP 和 CPU 1–8 星的 9 个开局，并验证从同一状态恢复后，同样的双方输入产生一致的画面、RAM 数组和读取值。这里 RAM 仅用于环境正确性核查，不给策略作为输入。迁移包不改变角色、星级、规则或双方控制器。

## 6. 复现完整基线、录像和 notebook

在仓库根目录执行，输出目录必须是新的：

```bash
python -m experiments.league_adaptive.evaluate --agent models/counter_ryu_final/agent.pt --pool configs/focus_pool.json --out artifacts/windows_reference_focus --arena-call-sequence
```

这会打完整 24 局 PvP 和 8 局 PvE，并输出逐局 `partial.json`、汇总 `summary.json`。`--arena-call-sequence` 模拟暖机调用及跨局不显式 reset 的推理序列，**仍然不是课程 Arena 自检**。目标先复现历史参考 11/24、8/8；如果不一致，比较具体对手、端口、offset、血量和结束原因，再定位原因，不能把总体相近当作通过。

这里 offset 0/12 表示双方在开局共同经过 0/12 个原生无输入帧，再开始 agent 控制；是本地覆盖条件。**不能据此宣称课程最终对同一小组必打四局。** P1/P2 是控制器编号，开局位于左/右，越过对方后编号不变。PvE 的电脑是游戏原生逻辑，不是我们加载的神经模型。

下面两条命令分别录制 CPU 八星和 P2 面对反制者的完整对局：

```bash
python -m experiments.league_adaptive.record --agent models/counter_ryu_final/agent.pt --cpu-character Ryu --cpu-stars 8 --player 1 --offset 0 --out artifacts/video_pve8 --label Counter-Ryu
python -m experiments.league_adaptive.record --agent models/counter_ryu_final/agent.pt --opponent models/opponents/strong_response.pt --player 2 --offset 12 --out artifacts/video_pvp_p2 --label Counter-Ryu
```

录像有端口、血量、输入等标注，输出的 `decisions.json` 可用于分析实际位移、动作输入和掉血时机。动作模板名不等于游戏一定成功施放了该招。

如需导出可独立加载的 SB3 权重，执行以下入口；它生成 `policy.zip`、独立学生源码和重新加载的预测比较，不调用课程打包/校验工具。导出后可使用 `stable_baselines3.PPO.load(..., device='cpu')` 加载，仍不等于已获官方 Arena 验收：

```bash
python -m experiments.ryu_direct_league.portable_export --agent models/counter_ryu_final/agent.pt --out artifacts/export_reference
```

如果希望逐块运行：

```bash
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name counter-ryu --display-name "Counter-Ryu Python 3.12"
python -m jupyter lab
```

打开 `notebooks/working/counter_ryu_windows.ipynb`，选上面的 kernel。Notebook 的模拟器、完整评估和训练开关默认关闭，阅读说明后逐个开启；它只是调用相同的模块，不另有一套算法。课程原版 notebook 保持只读。

## 7. 理解训练流程，先不要自动续跑

一次策略调用收到四帧图像与坐标，输出 42 个离散输入模板之一。模板通常展开为 6 个原生游戏帧；升龙/波动拳宏已包含方向与按键序列，不需要 agent 自己拼出整个摇杆指令。对方在同一时间也继续行动。环境推进，获得下一观察和真实输赢/血量变化，PPO 再用采样轨迹更新策略与价值网络。**按键请求、招式是否成功、收益是否来自该招是三件事。**

受训策略可表示为依据合法观察历史 (h_t) 选择动作的 πθ(a_t|h_t)。游戏转移由模拟器提供，当前代码没有学出可部署的转移模型。PvE 与 PvP 使用同样的环境推进；区别在于对方动作分别来自原生电脑或另一个控制器。保存/恢复分支还必须保留对手计数、历史、我方真实输入历史和随机状态，否则不是同一局面下的有效比较。

`scripts/train_reference_ppo.py` 是迁移时新增的精简普通 PPO 续训入口：加载当前 mixture 权重，保留专家，训练路由/价值网络；两个 PvP worker 分别 P1/P2，一个 PvE worker覆盖 1–8 星。使用已有环境的胜率对齐奖励（胜 +1，其余完整结果 -1，终局势函数归零，gamma=1）。**它不是复现旧训练轨迹，也不承诺让原模型更 adaptive。** 初次交接任务不启动训练；完成基线和分析后，如需要小规模运行，可使用：

```bash
python -m scripts.train_reference_ppo --steps 3072 --max-seconds 180 --out artifacts/windows_ppo_smoke
```

输出 `initial.pt`、`last.pt`、配置和训练结果，必须另做冻结整局评估；脚本没有自动晋级。创建输出目录中的 `STOP` 文件可请求停止，紧急情况下 Ctrl+C 后也要确认模拟器子进程退出。PPO 按 rollout 成批采样，`--steps` 不是精确墙钟或原生帧预算。

历史 Q/循环 PPO/高层动作段代码仍保留以供阅读，但其中部分命令引用旧训练数据或初始化权重，本包没有带上。**不要把那些旧命令直接当成可续跑的 Windows 流程。** 当前运行入口是本节明确列出的参考 PPO，后续方法先用具体假设和最小改动决定。

## 8. 历史探索中应该吸取什么

保存同一个局面，改变一段动作，再交回同一个冻结策略，可以测到条件性的整局影响。在 36 个困难根局面中，原策略 18 个非胜局有 17 个存在能转为胜利的替代动作段。这证明更好的路径存在，**不证明新网络能正确识别并持续执行这些路径**。

反事实标签往往是“先用动作段，再由旧策略继续”的回报 (Q^μ(s,a))。新策略后续也会不断改变动作，因此它真正需要的 (Q^π(s,a)) 可能不同。一次替换有效，不代表整局反复介入有效；这也是离线选择器严重掉胜率的重要候选解释，尚不是全部原因的唯一证明。

动作持续时间候选虽然走路更多，但某局 232 次走路请求大量发生在墙边；把走路换成无输入仍同血量获胜。另一个 Q 候选赢下原来失败的场景，却约 85% 仍为升龙输入、没有走路。因此不以“输入更丰富”冒充有收益的走位。血量读数更准、更长记忆、更多示范也均未自动提升胜率。

上轮主要问题是同时改动了表示、记忆、动作时间尺度、算法、奖励和训练数据，且部分实验从更弱的策略重新起步，因果归因不清。新一轮应固定可复现基线，区分“有没有有效场景经验”和“模型能不能利用经验”，每次只验证一个主要机制。不得因为 FightLadder、自博弈、DFL 或 World Model 的名称合理，就假定加入它会成功。

## 9. 论文阅读顺序与用途

论文已随仓库提供，**先读当前问题直接相关的部分**，不要先搜更多论文或照搬另一个游戏的代码。FightLadder 的框架与本课程 Genesis/SB3/Arena 不能直接等同，采用思路前要比较其观察、动作、训练对手与评估方式。

| 优先级 | 本地论文 | 先回答的问题 |
|---|---|---|
| 1 | [PPO](dqn_papers/ppo.pdf) | 当前 actor/critic、采样、裁剪更新和熵分别做什么？ |
| 1 | [FightLadder](dqn_papers/adaptive_rl/FightLadder_2024.pdf)；[项目网站](https://sites.google.com/view/fightladder/home)、[源码](https://github.com/wenzhe-li/FightLadder) | 对手种群、自博弈、利用者和评估如何组织？哪些适合本接口？ |
| 1 | [Opponent Modeling in Deep RL](dqn_papers/adaptive_rl/Opponent_Modeling_DRL_2016.pdf) | 如何从真实交手历史表征对手，而不是给对手 ID？ |
| 1 | [DRQN](dqn_papers/drqn.pdf) | 部分可观测性和历史输入如何训练、重置与评估？ |
| 2 | [Fighting Game Self-play](dqn_papers/adaptive_rl/Fighting_Game_Selfplay_2020.pdf) | 自博弈是否学到多样对抗，而不是重复同一种漏洞？ |
| 2 | [Expert Iteration](dqn_papers/adaptive_rl/Expert_Iteration_2017.pdf) | 如何把搜索得到的有效行为转成策略能力？ |
| 2 | [DAgger](dqn_papers/adaptive_rl/DAgger_2011.pdf) | 旧策略数据与新策略访问状态分布不一致怎么办？ |
| 2 | [DQN](dqn_papers/dqn.pdf)、[Double DQN](dqn_papers/double-dqn.pdf)、[Dueling](dqn_papers/dueling-dqn.pdf) | 值函数、bootstrap、过估计与优势分解；不是默认要换算法 |
| 2 | [PER](dqn_papers/1511.05952v4.pdf)、[Rainbow](dqn_papers/1710.02298v1.pdf) | 经验采样和组件消融如何做，避免无依据叠模块？ |
| 背景 | [Decision-Focused Forecasting](general_papers/2405.14719v2.pdf) | 多阶段预测与后续决策的关系；不是现有代码已实现的方法 |
| 背景 | [DFL in MDPs: Occupancy Measures](general_papers/2610.08384v1.pdf) | MDP 决策目标与模型学习如何连接？ |
| 背景 | [Decision-Focused Model-based RL for Reward Transfer](general_papers/sharma24a.pdf) | 世界模型应为哪类决策服务，其假设与本项目是否相符？ |

## 10. GitHub 两端怎样协作

Mac 的同步工作副本为原工作目录下的 `github_sync/`；大实验目录仍留在 Mac，不是仓库的一部分。Windows 克隆自己的副本，运行结果留在各自 `artifacts/`。**两端共享的是代码、配置、精选结果和明确任务，不是整个工作盘。**

先用 `git status` 确认当前目录是 Git 工作副本。目录名带 `-main` 可能来自下载 ZIP，但不能仅凭目录名判断。若当前确是 ZIP 解压目录，应在旁边正常克隆仓库，再复制本机报告和必要修改；不要覆盖原诊断目录，也不要初始化一个无共同历史的仓库强推主干。

Windows 开始一个任务前拉取主干、创建分支：

```bash
git switch main
git pull --ff-only
git switch -c win/baseline-reproduction
```

完成后只选择必要文件加入 Git，例如源码变更和一个小型汇总。不要 `git add -f artifacts/`，不要把训练全部权重/视频上传；必要录像先本地分析，再按需分享。不要提交 token、环境或 ROM。使用：

```bash
git add scripts/check_environment.py reports/windows_baseline_1022/summary.json
git commit -m "Record Windows baseline reproduction and environment"
git push -u origin win/baseline-reproduction
```

上面的文件名是示例，只加入实际存在且本次需要的文件。Mac 用 `git fetch origin`、切换对应远程分支进行审核；需要反馈时在 `review/<task>` 分支提交对现有 prompt/配置的修订。初期由 Mac 审核后合并主干，Windows 再拉取，避免两个会话同时改同一分支。这里不要求创建大量 PR、Issue 或 Markdown。

Windows 首次交付应包括：系统/依赖与设备、80 输入检查、9 开局恢复检查、32 场基线逐局差异、代码流程解释和下一步单一假设。遇到差异先修环境/接口，不用加训练掩盖；没有差异再讨论如何改善策略。**不要把“安装成功”或“短训跑完”汇报成模型目标已完成。**
