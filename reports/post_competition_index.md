# Counter-Ryu 探索与复跑索引

| 阶段 | 得到了什么 | 权重、配置与证据 |
| --- | --- | --- |
| 课程 PPO | 建立课程观察、动作、奖励与对战接口。历史 Windows 基线复跑覆盖 24 场 PvP 和 8 场 CPU。 | [课程课堂模板（只读）](../notebooks/upstream/sye6308-sf2-classroom-2026.ipynb) · [Windows 环境交接](../initial_readme_1022.md) |
| F | 循环 PPO 与连续经验整合。资格赛 1136 分、四场 KO、CPU 96/96、已知 PvP 204/204。用于比赛备用和同条件比较。 | [训练谱系](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/ryu-targeted-delivery) · [备用 ZIP](../models/competition_delivery/group10_Ryu.zip) · [逐局及交付报告](../reports/competition_delivery/summary.json) |
| 走位教师与学生 | 先验证训练用教师，再检验学生能否从标准开局独立进退。监督修复八星 Ryu 和 P1 远程败局，带成功整局回放的 PPO 继续保有修复。后来研究模型到 1141 分，同时已知 PvP 降至 157/204。 | [教师与学生实验](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/teacher-learnability/reports/teacher_learnability) · [胜局保持与续训](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/footwork-strength-integration/reports/footwork_strength_integration) · [1141 推理入口](../models/retained_to_f/runtime) · [逐局结果](retained_to_f/games.json) |
| 冻结 B | 主监督分支 1136 分、四场 KO、CPU 83/96、已知 PvP 185/204；原冻结面板 8/10（P1 3/5、P2 5/5）。保留为 1243 分模型的直接基线。 | [冻结权重与加载入口](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/ryu-competition-final/models/ryu_competition_final) · [配置、逐局与录像](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/ryu-competition-final/reports/ryu_competition_final) |
| 1243 分比赛模型 | 在冻结 B 后研究 P1 抗压与保持，最后模型资格赛四场 KO、已知 PvP 196/204。冻结测试为 18/20，P1 8/10、P2 10/10。同一面板 B 为 15/20（6/10、9/10），F 为 15/20（7/10、8/10）。 | [Windows 配置](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/ryu-p1-pressure/configs/ryu_p1_pressure.json) · [模型与本地加载入口](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/ryu-p1-pressure/models/ryu_p1_pressure) · [逐局与成对录像](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/ryu-p1-pressure/reports/ryu_p1_pressure) |

GitHub 保存 1243 分模型经 Mac R2 四局自检的外层包和对应回执。课程 Python 3.13 环境请下载 [Kaggle 外层原件](../models/championship_1243/kaggle/Group_10.zip)，对应 [TA Notebook 回执](championship_1243/kaggle/ta_arena_receipt.json) 及 [环境与四局记录](championship_1243/kaggle/ta_notebook_verification.json)。Windows 的 `models/ryu_p1_pressure/policy.zip` 供本地加载。 [R2 外层包](../models/championship_1243/Group_10.zip) · [R2 回执](championship_1243/receipt.json) · [3,696 次动作核对](championship_1243/action_replay.json)

## 有用的发现及反例

| 问题 | 留下的判断 | 配置与原始证据 |
| --- | --- | --- |
| DDQN 与角色的影响 | 反事实 DDQN 对普通 DDQN 的 PvP 结果在 seed 52031 为 5/24 对 1/24，seed 52037 为 12/24 对 17/24。Guile 的持续回放减轻退化，后续检查仍分别丢失先前新增的 5/5 和 3/3 个能力条件。角色与协议的对照支持本项目后续选择 Ryu 循环 PPO 加监督整合，归因限于已观察的协议与训练预算。 | [反事实配置、原生分支及逐局](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/counterfactual-dqn) · [DDQN 稳定化原始记录](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/ddqn-stabilization/reports/ddqn_stabilization) · [角色交叉比较](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/character-algorithm-breakthrough/reports/character_algorithm_breakthrough) |
| 外部对手与示范 | FightLadder 的原生历史与控制周期分开适配。PSRO 右端口实测有主动前进 16 和后退 6 像素。局部移动的诱空收益当时未验证，两个 seed 的 PPO 外部采样终点效果方向不同。 | [原始来源与配置](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/external-opponents/reports/external_opponents/sources.json) · [完整对局与适配验证](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/external-opponents/reports/external_opponents) |
| 反事实走位 | 1141 权重的 Ken 近身 P2 局中，主动调位和蹲防均为 176–0 KO，等时等待为 147–0 KO。主动调位比等待少受伤，蹲防在该历史下同样有效。46 个交回 F 的对照里，14 个出现主动地面移动，1 个达到预设整局收益。记录保留教师续控和接管边界。 | [同状态逐分支结果](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/retained-to-f/reports/retained_to_f/evidence.json) · [完整近身录像](retained_to_f/videos/p2_near_ken.mp4) · [远程失败录像](retained_to_f/videos/p2_ken_failure.mp4) · [交回 F 的对照](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/f-selective-footwork/reports/f_selective_footwork) |
| 走位保持与旧能力损失 | 退让低踢的主体更新终点，两 seed 均保住三个训练条件。一个新 P1 时序在两 seed 出现候选交换，另一个只在 seed 76137 重复。代表已知 PvP 161/204，P2 从 F 的 102 降到 76；部分去掉移动的对照仍有收益。 | [训练配置](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/footwork-retention/configs/footwork_retention.json) · [双 seed 学习记录、移动对照和录像](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/footwork-retention/reports/footwork_retention) |
| 连续学习与干扰 | 73131 新增的六个 A 胜局在学习 B 后保住六个，学习 C 后只剩两个。73137 未复现该积累。学生状态聚合在另一个实验的首 seed 减轻退化，次 seed 仅多一场 PvP 胜局，CPU 和最弱端口下降。 | [A→B→C 配置和完整记录](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/continual-improvement/reports/continual_improvement) · [闭环匹配试验与逐局证据](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/closed-loop-tactics/reports/closed_loop_tactics) |

外部对手权重按来源下载。保留上游许可和原始端口配置。 [FightLadder MIT](../licenses/FightLadder.txt) · [street-fighter-ai Apache 2.0](../licenses/street-fighter-ai.txt)

## 复跑与 Windows 保留范围

- 日常复跑从 [1117 课程 Notebook](../notebooks/working/sye6308_group10_1117.ipynb) 开始。默认 F，`MODEL_CHOICE='research'` 切换 1141 分模型。在第 6 节选择资格赛，第 8 节换边。单个指定对局可使用 `experiments.league_adaptive.record` 的模型、对手池、对手 ID、端口和偏移参数。 [共享录像入口](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/ryu-p1-pressure/experiments/league_adaptive/record.py)
- 最终训练的 10 个文件、冻结评估证据、402 条经验及其来源索引、原始对手与原生开局、两批恢复路径均保存在 Windows 原工作树。旧报告引用的失败分支可用于结果复核，保留推理权重和对手输入的样本可重跑。退役中间步骤的优化器能否恢复以具体路径记录为准。
- Windows 从 `win/ryu-p1-pressure` 的完整实验目录调用本归档分支的 [恢复命令](../scripts/rebuild_p1_archive.py)。`--mode resume` 在独立目录执行最终阶段的下一次真实更新，`--mode rebuild --bank <bank>` 重建派生库并试读固定批次。旧清理映射的 114 场涉及八个等价输入别名。未抽测的对局只登记重跑输入。 [本轮实测、依赖和代表复盘](post_competition_checks.json)

```powershell
git fetch origin refs/heads/win/post-competition-archive:refs/remotes/origin/win/post-competition-archive
git worktree add --detach artifacts/archive_recipe origin/win/post-competition-archive
$archiveTool = 'artifacts/archive_recipe/scripts/rebuild_p1_archive.py'
python -B $archiveTool --workspace . --out artifacts/archive_resume --mode resume
python -B $archiveTool --workspace . --out artifacts/archive_coverage --mode rebuild --bank p1_rehearsal_coverage
```
