# Satisfaction_Predict

Kaggle **Predicting Airline Satisfaction**：方法 3 的独立代码与工作报告。

| 评估 | 结果 |
|---|---:|
| Private ROC-AUC | **0.96238** |
| Private 排名 | **106 / 373** |
| Public ROC-AUC | **0.96186** |
| 五折元模型 CV，seed=7 | 0.96222639 |

公开榜与私有榜成绩来自用户实际提交反馈，private 结果确认于 **2026-11-13**。结果绑定原提交的 SHA-256，见 [结果记录](metadata/results.json)。

## 方法

使用 245 个公开基模型的 OOF/test 预测训练一个带 L2 正则的逻辑回归组合器：

1. 按竞赛 id 和固定成员顺序整理预测；无 id 的数组沿用作者声明顺序。
2. 241 列概率截至 [10⁻¹⁵, 1−10⁻¹⁵]，转 logit 后截至 [−30, 30]；4 列原始分数保留。
3. 家族缩放：157 个 GBDT ×1，58 个 MLP ×1，16 个 TabPFN ×3，14 个 TFM/TabM ×1.5。
4. 拟合 LogisticRegression：C=0.003、L2、lbfgs、tol=1e-4、max_iter=5000。
5. 使用全量拟合的一个元模型输出满意度连续预测。

家族缩放改变原始空间系数的正则惩罚，并非固定融合权重。基模型预测和家族正则配方来自公开工作；本项目训练的是元模型，没有从头训练 245 个基模型。元 CV 使用固定公开 OOF，不能视为全部基模型重新训练的严格嵌套验证。

详细数据分析、实际缺失处理、特征工程消融和选模依据见 [完整工作报告](reports/FINAL_METHOD_REPORT.md)。报告中的 53 特征本地探索是研究历史，没有进入本方法。

## 目录

~~~text
family_stack.py       方法 3 的 CV、全量训练、保存模型推理
prepare_inputs.py     从公开成员文件直接构造 245 列矩阵
download_inputs.py    下载固定版本/输出 session 的必要公开输入
common.py            哈希、JSON、竞赛数据与 Compute Node 检查
run.sh               Slurm 入口，1 CPU / 12 GB / 2 小时，无 GPU
requirements.lock    原环境中与本方法有关的依赖版本
model/               原始方法 3 的已训练组合器（约 9 KB）
metadata/            成员顺序、公开来源、协议、哈希、实测结果
reports/             8 份非 submission 报告及相关配图
~~~

代码中没有 V1、V2 或其他 V3 候选的训练、融合分支。报告保留研究历史，因此仍会提到这些对照方法，但不携带它们的代码。仓库不包含 CSV 文件、竞赛数据、OOF/test 大数组、训练日志或 submission 指南。

## 运行

沿用已有项目虚拟环境；原模型环境为 Python 3.9.19、scikit-learn 1.6.1。不要重建原环境。设置其 Python 路径：

~~~bash
export AIRLINE_PYTHON=/path/to/existing/project/.venv/bin/python
# 仅在原环境需要补齐依赖时使用：
uv pip install --python "$AIRLINE_PYTHON" -r requirements.lock
~~~

将竞赛 train.csv、test.csv、sample_submission.csv 放入本目录的 data/，或者通过 --data-dir 指定已有目录。原始文件哈希必须与 metadata/data_audit.json 一致。

### 从公开成员准备输入

~~~bash
"$AIRLINE_PYTHON" download_inputs.py
sbatch run.sh prepare --data-dir /path/to/competition/data
~~~

成员来源固定在 [public_sources.json](metadata/public_sources.json)。下载器只获取方法 3 所需文件，不获取其他方案的最终预测。公开 Notebook 输出使用固定 session；如果作者更新或删除了原输出，程序会报错，需要取得对应历史输入，不会静默替换。竞赛数据须自行按 Kaggle 访问条件下载。

准备脚本按 [成员清单](metadata/members.json) 组装输入，检查各列长度、有限性、范围及 OOF AUC，最后核对两个完整矩阵的 SHA-256。标签不用于测试特征构造。生成：

- inputs/oof_logits.npy：699,635 × 245，float64；
- inputs/test_logits.npy：299,844 × 245，float64。

若已有原项目的方法 3 矩阵，可直接使用 --input-dir 指向矩阵目录，无需重新下载和组装。

### 使用原模型重建预测

~~~bash
sbatch run.sh predict \
  --data-dir /path/to/competition/data \
  --input-dir /path/to/method3/matrices
~~~

默认加载 model/model_family.joblib。输出到 outputs/submission_family.csv，核对模型、输入及生成 CSV 的原始 SHA-256。输出 CSV 被 Git 忽略。

### 重新训练与验证

~~~bash
sbatch run.sh cv --data-dir /path/to/competition/data --input-dir /path/to/method3/matrices
# 前一个作业完成后：
sbatch run.sh train --data-dir /path/to/competition/data --input-dir /path/to/method3/matrices
# 可选：原实验的更严格求解精度检查，仅针对同一方法
sbatch run.sh cv --seeds 7 --tol 1e-6 \
  --data-dir /path/to/competition/data --input-dir /path/to/method3/matrices
~~~

CV 默认三个种子 7、13、101，每个种子五折；为控制资源，独立版按折顺序运行，每次一个 BLAS 线程。全量训练固定 tol=1e-4，将新模型放入 outputs/model_family.joblib，不覆盖仓库内原始模型。重训后的序列化文件哈希可能不同；原始实测分数对应仓库中保留的原模型与登记提交。

全部输入准备、训练与推理都通过 Slurm Compute Node 执行。run.sh 使用 MGPU-TC2 / normal，可按服务器环境调整资源；同一用户总资源与运行作业数应遵守所在集群限制。

## 整理与验证范围

此次整理保留原模型字节及完整成员顺序，提取出单一方法的独立入口；检查了 Python 语法、Shell 语法、文件范围、报告链接、成员来源覆盖和原模型 SHA-256。原项目虚拟环境和原始数据在此次整理环境中不可用，因此没有宣称新入口已完成端到端数值重跑；历史训练与验证结果见 metadata/results.json，当前静态检查范围见 metadata/package_validation.json。

第三方输入不随仓库再分发。方法来源、原作者与许可信息见 [THIRD_PARTY.md](THIRD_PARTY.md)；文档中的历史版本命令仅作记录，实际运行使用本 README。
