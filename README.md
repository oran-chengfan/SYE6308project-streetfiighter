# Counter-Ryu · SYE6308

## 比赛交付与复跑

| 版本 | 归档用途 | 入口 |
| --- | --- | --- |
| **1243 分 Ryu** | 比赛模型。GitHub 保存经 Mac Public Practice Arena R2 四局自检的外层包 | [R2 Group_10.zip](models/championship_1243/Group_10.zip) · [R2 回执](reports/championship_1243/receipt.json) |
| **F，1136 分** | 比赛备用、历史强度基线、Notebook 默认模型 | [本地推理 zip](models/competition_delivery/group10_Ryu.zip) · [交付结果](reports/competition_delivery/summary.json) |
| **冻结 B，1136 分** | 1243 分模型的直接比较基线 | [冻结 B 的权重和实验结果](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/ryu-competition-final) |
| **1141 分走位模型** | 研究主动调位的收益、反例和整局代价 | [推理目录](models/retained_to_f/runtime) · [逐局结果](reports/retained_to_f/games.json) |

1243 分模型四场资格赛均 KO。春丽二星和 Ryu 二、四、八星分别剩 176、176、84、103 血。冻结换边 PvP 为 18/20（P1 8/10，P2 10/10）。Mac 的 R2 记录核对了打包前后 3,696 次动作选择及四场终局。Kaggle／TA 最终外层原件及对应回执待 Mac 补交。 [动作核对](reports/championship_1243/action_replay.json) · [Windows 选型报告](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/ryu-p1-pressure/reports/ryu_p1_pressure/summary.json)

复跑优先使用已配置的 1117 Notebook。赛后索引收录课程 PPO、F、走位师生、B 和 1243 的谱系，也列出了失败实验与 Windows 独有的恢复输入。 [运行 Notebook](notebooks/working/sye6308_group10_1117.ipynb) · [探索与复跑索引](reports/post_competition_index.md) · [赛后验证记录](reports/post_competition_checks.json)

## Notebook 演示与续训

1117 Notebook 默认演示历史基线 **F**。`MODEL_CHOICE='research'` 选择 **1141 分走位研究模型**，用于复看走位与对照实验。比赛使用上方的 1243 分提交包。

打开 [课程结构 Notebook](notebooks/working/sye6308_group10_1117.ipynb)，选择已经配置的 `sye6308` 内核后“全部运行”。它保留上游课程的 53 个单元格结构，默认加载 F，运行Ryu 四星、八星两场 PvE 和换边两场 PvP，显示完整录像并保留终局后一秒。第6节切换 `PVE_EVALUATION_SCOPE='qualification'` 可运行公告四场并在全部KO时显示得分。1117版与1110版默认权重一致，更新的是入口与说明。默认不训练；开启第 5 节或第 9.4 节可执行真实循环 PPO 续训。新结果写入 `artifacts/`，不会覆盖 F。

- [F 单 zip](models/competition_delivery/group10_Ryu.zip)：可在本项目运行环境中独立加载。
- [F 原始权重](models/ryu_targeted_delivery/feedback_candidate.pt)及 [Notebook 推理目录](models/ryu_targeted_delivery/feedback_runtime)。
- [资格赛交付记录](reports/competition_delivery/release.json)、[本地加载检查](reports/competition_delivery/compatibility.json)、[完整实验汇总](reports/competition_delivery/summary.json)。
- 首次配置参考 [Windows 环境交接](initial_readme_1022.md)；它记录的是十月历史基线，当前入口与成绩以本页为准。已有环境无需重装。

## 历史基线 F

| 条件 | F 已记录结果 |
|---|---:|
| 12 个 CPU 角色 × 8 星，标准开局 | 96/96 KO |
| 已知 PvP 回归条件 | 204/204，P1/P2 各 102/102 |
| 公告四场资格赛，当前 P1、offset0 开局 | 4/4 KO，1136 分 |

这些是固定条件的结果，不是对未知同学模型的胜率保证。204 场主要由脚本对手条件构成，并非 204 个独立的强神经对手。F 仍大量使用技能输入，稳定主动走位、跨打法适应与持续能力积累尚未得到充分证明。

四场资格赛录像：[春丽2星](reports/competition_delivery/F_qualification_ChunLi_2.mp4)、[Ryu2星](reports/competition_delivery/F_qualification_Ryu_2.mp4)、[Ryu4星](reports/competition_delivery/F_qualification_Ryu_4.mp4)、[Ryu8星](reports/competition_delivery/F_qualification_Ryu_8.mp4)。自身剩血依次为154、176、84、18；四场均在计时结束前 KO。

F 的便携 ZIP 用于本地推理。课程 R2 外层包见页首。

## 1141 分走位研究模型与完整对局

在 1117 Notebook 第 1 节首个代码格把 `MODEL_CHOICE = 'F'` 改为 `MODEL_CHOICE = 'research'`，然后从头全部运行。第 6 节运行所选模型的 PvE，第 8 节让同一模型作为 P1 和 P2 对战固定神经对手。研究导出使用同目录的策略包和动作、观察定义。[研究推理目录](models/retained_to_f/runtime)

| Windows 已记录条件 | 历史基线 F | 1141 分走位学生 |
|---|---:|---:|
| 公告四场资格赛 | 1136 分，4 KO | 1141 分，4 KO |
| 全角色与星级 CPU | 96/96 KO | 61/96 KO |
| 已知 PvP（P1、P2） | 204/204（102、102） | 157/204（82、75） |

两份模型的资格赛差距来自春丽二星，学生剩 159 血，F 剩 154 血。新近身 Ken 的 P2 局约 8.3 秒时，学生主动调位 15 像素，最终满血 KO。同状态改为等时等待后剩 147 血，改为蹲防也能满血 KO。另一场 P2 远程 Ken 对局中学生落败。两段完整录像均来自表中同一份学生权重。[近身胜局](reports/retained_to_f/videos/p2_near_ken.mp4) · [远程败局](reports/retained_to_f/videos/p2_ken_failure.mp4) · [逐局成绩](reports/retained_to_f/games.json) · [研究报告](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/retained-to-f/reports/retained_to_f/summary.json)

**学生如何学会走位。** 两个训练 seed 的学生在有限的近身条件中能自主进退、衔接攻防，短程 PPO 续训后仍保有这些行为。随后一个学生通过胜局回放保住近身战术，修复八星 Ryu 和 P1 远程两场败局；该阶段的 `retained_ppo.pt` 是表中走位学生的训练起点。[教师学习报告](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/teacher-learnability/reports/teacher_learnability/summary.json) · [学生自主整局](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/teacher-learnability/reports/teacher_learnability/videos/research_new_contact_P2.mp4) · [起点与续训记录](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/footwork-strength-integration/reports/footwork_strength_integration/summary.json)

**F 能否直接接续学生的走位。** 46 个同状态对照中，14 个出现主动地面移动，只有一处达到预设整局收益门槛，等时蹲防也取得相同剩血和 KO。这轮保留了移动、等待、防守和交回 F 后的终局结果，没有另训学生。[对照研究](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/f-selective-footwork/reports/f_selective_footwork/summary.json) · [逐分支记录](https://github.com/oran-chengfan/SYE6308project-streetfiighter/blob/win/f-selective-footwork/reports/f_selective_footwork/paired_games.json)

## 主分支与研究分支

主分支保存比赛交付、教学 Notebook 和精选研究证据。训练过程、对手权重和历史候选保存在各自的研究分支。F 的开发过程与当时的资格赛记录可在原分支查阅。[F 的训练](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/ryu-targeted-delivery) · [F 的交付记录](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/competition-delivery)

Notebook 的历史轨迹整合复现开关需要原研究环境的经验数据，精简主分支不包含这些缓存；普通 PPO 续训不需要它们。动作名称表示发送的按键序列，不等同于游戏中实际成功发动的招式。

Windows 执行实验，Mac 审核和整合。遵守 [AGENTS.md](AGENTS.md)：禁止文件哈希、禁止堆叠重复训练器，prompt 通过聊天转交。课程原始 ROM 按现有授权保留为二进制文件，不转换行尾；`notebooks/upstream/` 保持只读，不直接 Run All。
