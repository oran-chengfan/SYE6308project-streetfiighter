# Counter-Ryu · SYE6308

本主分支整理截至 2026-11-16 已验证的 **F / feedback_candidate** 交付。己方固定 Ryu，同一份权重用于 PvE 与 PvP、P1 与 P2。方法是循环 PPO 基础加经过完整对局验证的连续轨迹监督整合；不能把全部提升归给 PPO 算法名称。

## 运行与交付

打开 [课程结构 Notebook](notebooks/working/sye6308_group10_1110.ipynb)，选择已经配置的 `sye6308` 内核后“全部运行”。它保留上游课程的 53 个单元格结构，默认加载 F，运行两场八星 PvE 和换边两场 PvP，显示完整录像并保留终局后一秒。默认不训练；开启第 5 节或第 9.4 节可执行真实循环 PPO 续训。新结果写入 `artifacts/`，不会覆盖 F。

- [F 单 zip](models/competition_delivery/group10_Ryu.zip)：可在本项目运行环境中独立加载。
- [F 原始权重](models/ryu_targeted_delivery/feedback_candidate.pt)及 [Notebook 推理目录](models/ryu_targeted_delivery/feedback_runtime)。
- [资格赛交付记录](reports/competition_delivery/release.json)、[本地加载检查](reports/competition_delivery/compatibility.json)、[完整实验汇总](reports/competition_delivery/summary.json)。
- 首次配置参考 [Windows 环境交接](initial_readme_1022.md)；它记录的是十月历史基线，当前入口与成绩以本页为准。已有环境无需重装。

## 已取得的成果与边界

| 条件 | F 已记录结果 |
|---|---:|
| 12 个 CPU 角色 × 8 星，标准开局 | 96/96 KO |
| 已知 PvP 回归条件 | 204/204，P1/P2 各 102/102 |
| 公告四场资格赛，当前 P1、offset0 开局 | 4/4 KO，1136 分 |

这些是固定条件的结果，不是对未知同学模型的胜率保证。204 场主要由脚本对手条件构成，并非 204 个独立的强神经对手。F 仍大量使用技能输入，稳定主动走位、跨打法适应与持续能力积累尚未得到充分证明。后续研究没有得到值得替换 F 的模型，不把不同候选的局部成功相加。

四场资格赛录像：[春丽2星](reports/competition_delivery/F_qualification_ChunLi_2.mp4)、[Ryu2星](reports/competition_delivery/F_qualification_Ryu_2.mp4)、[Ryu4星](reports/competition_delivery/F_qualification_Ryu_4.mp4)、[Ryu8星](reports/competition_delivery/F_qualification_Ryu_8.mp4)。自身剩血依次为154、176、84、18；四场均在计时结束前 KO。

**官方 Arena 尚未通过。** 此单 zip 已通过本地加载、连续推理和完整对局检查，但不满足当前存档 Public Practice R2 的外层提交容器接口；最终比赛加载接口仍须按赛方实际版本确认。不要把本地可运行写成官方已验收。

## 主分支与研究分支

本次是从已审核代码和 `win/competition-delivery` 中选择交付内容整合到 main，不是把所有实验分支整体合并。保留原课程文件、论文、环境补丁、必要模型和简洁结果；失败候选、训练缓存、大量中间录像保留在相应研究分支或 Windows 本地。报告中的历史候选路径可能仅存在于原研究分支。

F 的来源见 [win/ryu-targeted-delivery](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/ryu-targeted-delivery)；比赛交付详见 [win/competition-delivery](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/competition-delivery)。后续走位研究见 [win/conditional-footwork](https://github.com/oran-chengfan/SYE6308project-streetfiighter/tree/win/conditional-footwork)，未晋级。

Notebook 的历史轨迹整合复现开关需要原研究环境的经验数据，精简主分支不包含这些缓存；普通 PPO 续训不需要它们。动作名称表示发送的按键序列，不等同于游戏中实际成功发动的招式。

Windows 执行实验，Mac 审核和整合。遵守 [AGENTS.md](AGENTS.md)：禁止文件哈希、禁止堆叠重复训练器，prompt 通过聊天转交。课程原始 ROM 按现有授权保留为二进制文件，不转换行尾；`notebooks/upstream/` 保持只读，不直接 Run All。
