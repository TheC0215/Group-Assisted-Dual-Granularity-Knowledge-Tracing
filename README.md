# GADG-KT

GADG-KT：融合全局知识状态与双粒度时序状态的知识追踪模型。

## 模型简介

- **题目编码**：利用注意力机制聚合题目关联的多个知识点（MC），并结合题目专属难度特征（OF），构建综合题目表征，有效解决“一题多知识点”与“同知识点题目难度差异”问题；
- **全局知识状态**：构建交互-知识点层次结构，采用“Epoch 级节点慢更新 + Batch 级参数快适配”的解耦设计，将群体交互状态聚合为全局知识状态；
- **粗粒度**：采用两个并行 LSTM 维护时序状态。细粒度层面采用稀疏更新策略（仅更新当前题目涉及的知识点），粗粒度层面维护题目级宏观认知；
- **细粒度**：在预测阶段，对细粒度知识点状态与粗粒度题目状态分别引入时间间隔遗忘机制，模拟记忆随时间衰减的过程；
- **融合**：通过标量门控机制自适应融合遗忘后的个体双粒度状态与全局知识状态，经前馈神经网络输出最终作答概率。

## 环境要求

- Python 3.8+，PyTorch 1.12+（CUDA 可选）
- `pip install torch pandas numpy scikit-learn tqdm`

## 数据准备

数据目录结构（`data/<数据集>/`）：

| 数据集 | 原始文件 |
|---|---|
| assist09 | `skill_builder_data_corrected_collapsed.csv` |
| assist12 | `2012-2013-data-with-predictions-4-final.csv` |
| NeurIPS 2020 | `train_task_3_4.csv` + `question_metadata_task_3_4.csv` |

运行预处理（按需切换 `preprocessdata.py` 中的 `dataname`）：

```bash
python preprocessdata.py
```

生成 `ques_skill.csv`、`train/test_question.txt`、`train/test_skill.txt`、`train_fold.txt` 等文件。

## 训练与评估

```bash
cd GADG-KT
python main.py --dataset assist09      # 或 assist12 / "NeurIPS 2020"
```

采用 5 折交叉验证，验证集 AUC 早停（patience=10），每折保存最优模型，训练结束后在测试集评估并输出：

- 总体 5 折平均 AUC/ACC
- 逐折 AUC/ACC 与训练耗时

结果文件：`GADG-KT_{数据集}_results.txt`，最优模型：`GADG-KT_{数据集}_{折}_model.pkl`。

## 引用

如使用本代码，请引用相应论文。
