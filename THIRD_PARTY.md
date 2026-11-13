# 方法与公开输入来源

本仓库是方法 3 的独立整理版本，保留本项目训练的逻辑回归组合器。245 个基模型预测来自公开 Kaggle 工作，未在本项目重新训练。

- [Twin Stacks + Family Regularization — goodpjw2008](https://www.kaggle.com/code/goodpjw2008/s6e10-twin-stacks-family-reg-lb-0-96187)：245 成员与家族正则配方，研究时采用版本 8。
- [228-member Meta-Stack — goodpjw2008](https://www.kaggle.com/code/goodpjw2008/s6e10-228-member-meta-stack-lb-0-96184)：公开成员整理和正则化组合参考。
- [Secrets to Satisfaction? It's Stacking — yeashusemwal](https://www.kaggle.com/code/yeashusemwal/secrets-to-satisfaction-it-s-stacking)：多家族 OOF 组合思路。
- [Ablation Members — kagankoral](https://www.kaggle.com/datasets/kagankoral/s6e10-ablation-members-oof)：17 个新增成员，固定 version 1；来源元数据许可为 CC BY-SA 4.0。

全部 16 个输入来源及原作者账号、数据集版本或 Notebook 输出 session、必要文件名见 [来源清单](metadata/public_sources.json)。其中 245 个成员的对应关系见 [成员清单](metadata/members.json)。

原始调查 teacher、航程画像等研究只在历史报告中保留说明，不是此仓库最终模型的额外输入。原始调查来源：[Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)。

公开输入各自受来源页面的许可和访问条件约束，来源许可未知时不推定其许可。本仓库不再分发原始数据、公开 OOF/test 数组或作者 Notebook 全文，也不以统一许可证覆盖第三方材料。
