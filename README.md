# GADG-KT

GADG-KT：结合粗粒度知识点图与细粒度时序状态的知识追踪模型。

## 模型简介

- **题目编码**：多面知识点（MC）+ 客观因子（OF），题目与其知识点间通过注意力建立权重联系；
- **粗粒度**：交互-知识点二部图上的两层消息传递，捕获全局知识状态；
- **细粒度**：稀疏知识点 LSTM 更新（只更新当前题涵盖的知识点，无关知识点状态保持不变），并引入时间间隔遗忘机制（距上次作答越久，状态按可学习方式衰减）；
- **融合**：标量门控软融合粗细两种知识状态，AKT 风格预测头输出作答概率。

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
