import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from model import GADGKT
from run import run_epoch
from load_data import load_dataset

# 数据在上级目录 模型/data 下（以本脚本位置为锚点，与运行目录无关）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, '..', 'data')


def read_fold_ids(fold_path):
    with open(fold_path, 'r') as f:
        return [int(x) for x in f.read().split() if x.strip() != '']


def build_int_skill_edges(dataset, pro2skill, device):
    """粗粒度知识点图（交互版）：节点 = 知识点 + 交互，边 = 每个交互 × 其题目的每个知识点。
    与 KT_Dataset 的全局交互编号同源（从同一 dataset 建图，保证 index_add 对齐）。
    返回边三元组 (E_int, E_pro, E_skill)：第 e 条边连接交互 E_int[e] 与知识点 E_skill[e]，E_pro[e] 为该交互的题目。"""
    e_int, e_pro, e_skill = [], [], []
    skill_of = [p.nonzero().flatten().tolist() for p in pro2skill]  # 每题的知识点列表
    for base, seg in zip(dataset.int_base, dataset.problem_list):
        for t, p in enumerate(seg):
            for k in skill_of[p]:
                e_int.append(base + t)
                e_pro.append(p)
                e_skill.append(k)
    E = torch.tensor([e_int, e_pro, e_skill], dtype=torch.long).to(device)
    return E[0], E[1], E[2], dataset.num_int


if __name__ == '__main__':

    mp2path = {
        'assist09': {
            'ques_skill_path': os.path.join(DATA_ROOT, 'assist09', 'ques_skill.csv'),
            'train_path': os.path.join(DATA_ROOT, 'assist09', 'train_question.txt'),
            'test_path': os.path.join(DATA_ROOT, 'assist09', 'test_question.txt'),
            'train_skill_path': os.path.join(DATA_ROOT, 'assist09', 'train_skill.txt'),
            'test_skill_path': os.path.join(DATA_ROOT, 'assist09', 'test_skill.txt'),
            'fold_path': os.path.join(DATA_ROOT, 'assist09', 'train_fold.txt'),
            'skill_max': 999},
        'assist12': {
            'ques_skill_path': os.path.join(DATA_ROOT, 'assist12', 'ques_skill.csv'),
            'train_path': os.path.join(DATA_ROOT, 'assist12', 'train_question.txt'),
            'test_path': os.path.join(DATA_ROOT, 'assist12', 'test_question.txt'),
            'train_skill_path': os.path.join(DATA_ROOT, 'assist12', 'train_skill.txt'),
            'test_skill_path': os.path.join(DATA_ROOT, 'assist12', 'test_skill.txt'),
            'fold_path': os.path.join(DATA_ROOT, 'assist12', 'train_fold.txt'),
            'skill_max': 999},
        'NeurIPS 2020': {
            'ques_skill_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'ques_skill.csv'),
            'train_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'train_question.txt'),
            'test_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'test_question.txt'),
            'train_skill_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'train_skill.txt'),
            'test_skill_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'test_skill.txt'),
            'fold_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'train_fold.txt'),
            'skill_max': 999}
    }

    parser = argparse.ArgumentParser(description='GADG-KT 五折交叉验证')
    parser.add_argument('--dataset', type=str, default='assist09')
    args = parser.parse_args()

    dataset = args.dataset

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    d = 64
    state_d = 64
    p = 0.4
    learning_rate = 0.002
    epochs = 200
    batch_size = 80  # 题目层图已删除，显存大头为交互图（与 batch 无关），assist09/assist12 统一用 80
    min_seq = 3
    max_seq = 200
    grad_clip = 15.0
    patience = 10          # pyKT 风格：验证 AUC 连续 patience 轮不涨即早停
    n_folds = 5

    ques_skill_path = mp2path[dataset]['ques_skill_path']
    train_path = mp2path[dataset]['train_path']
    test_path = mp2path[dataset]['test_path']
    train_skill_path = mp2path[dataset]['train_skill_path']
    test_skill_path = mp2path[dataset]['test_skill_path']
    fold_path = mp2path[dataset]['fold_path']
    skill_max = mp2path[dataset]['skill_max']

    pro_max = 1 + max(pd.read_csv(ques_skill_path).values[:, 0])
    skill_max = 1 + max(pd.read_csv(ques_skill_path).values[:, 1])

    pro2skill = torch.zeros((pro_max, skill_max)).to(device)

    for (x, y) in zip(pd.read_csv(ques_skill_path).values[:, 0], pd.read_csv(ques_skill_path).values[:, 1]):
        pro2skill[x][y] = 1

    ############################ model training ##################################3
    # pyKT 风格五折交叉验证：每折训练用 4 折、验证用 1 折（早停+选模型），
    # 训练结束后用 best checkpoint 在测试集上评一次；报告 5 折 test AUC 平均

    avg_auc = 0
    avg_acc = 0
    fold_aucs = []
    fold_accs = []
    fold_times = []
    total_epochs = 0

    criterion = nn.BCELoss()
    classify = nn.CrossEntropyLoss()

    all_folds = set(range(n_folds))

    for now_step in range(n_folds):

        train_folds = all_folds - {now_step}

        # 图结构只由训练折构建（验证折/测试集不进图）；知识点图与训练集交互编号同源
        train_dataset = load_dataset(pro_max, train_skill_path, train_path, min_seq, max_seq,
                                     folds=train_folds, fold_path=fold_path)
        E_int, E_pro, E_skill, num_int = build_int_skill_edges(train_dataset, pro2skill, device)

        model = GADGKT(skill_max, pro_max, d, p, state_d=state_d, max_seq=max_seq,
                     pro2skill=pro2skill, num_int=num_int,
                     E_int=E_int, E_pro=E_pro, E_skill=E_skill).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)

        best_valid_auc = 0
        best_valid_acc = 0
        bad_cnt = 0

        t0 = time.time()
        for epoch in range(epochs):

            train_loss, train_acc, train_auc = run_epoch(classify, train_skill_path, model, optimizer,
                                                         pro_max, train_path, batch_size,
                                                         True, min_seq, max_seq, criterion, device,
                                                         grad_clip, folds=train_folds, fold_path=fold_path)
            print(
                f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}')

            # epoch 末：用本 epoch 累积的"做完题后"状态刷新节点特征（供下一 epoch 用）
            model.finalize_graph()

            valid_loss, valid_acc, valid_auc = run_epoch(classify, train_skill_path, model, optimizer,
                                                         pro_max, train_path, batch_size, False,
                                                         min_seq, max_seq, criterion, device, grad_clip,
                                                         folds={now_step}, fold_path=fold_path)

            print(
                f'epoch: {epoch}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, valid_auc: {valid_auc:.4f}')

            # 早停 + 模型选择：只看验证集 AUC
            if valid_auc >= best_valid_auc:
                best_valid_auc = valid_auc
                best_valid_acc = valid_acc
                bad_cnt = 0
                torch.save(model.state_dict(), f"./GADG-KT_{dataset}_{now_step}_model.pkl")
            else:
                bad_cnt += 1
                if bad_cnt >= patience:
                    print(f'fold {now_step}: 验证 AUC 连续 {patience} 轮不涨，早停于 epoch {epoch}')
                    break

        fold_time = time.time() - t0
        fold_times.append(fold_time)
        total_epochs += (epoch + 1)
        print(f'fold {now_step}: 训练耗时 {fold_time:.1f} 秒')

        print(f'*******************************************************************************')
        print(f'fold {now_step}: best_valid_auc: {best_valid_auc:.4f}, best_valid_acc: {best_valid_acc:.4f}')

        # 载入 best checkpoint，在测试集上评一次（测试集只用于最终报告）
        model.load_state_dict(torch.load(f"./GADG-KT_{dataset}_{now_step}_model.pkl"))
        test_loss, test_acc, test_auc = run_epoch(classify, test_skill_path, model, optimizer, pro_max,
                                                  test_path, batch_size, False,
                                                  min_seq, max_seq, criterion, device, grad_clip)
        fold_aucs.append(test_auc)
        fold_accs.append(test_acc)
        print(f'fold {now_step}: test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}')
        print(f'*******************************************************************************')

        avg_auc += test_auc
        avg_acc += test_acc

    avg_auc = avg_auc / n_folds
    avg_acc = avg_acc / n_folds

    # ============ 保存结果：总体 AUC/ACC ============
    with open(f'./GADG-KT_{dataset}_results.txt', 'w', encoding='utf-8') as f:
        f.write(f'模型: GADG-KT | 数据集: {dataset}\n')
        f.write(f'总体({n_folds}折平均): AUC={avg_auc:.4f} ACC={avg_acc:.4f}\n')
        f.write('逐折 AUC: ' + ', '.join(f'fold{i}={a:.4f}' for i, a in enumerate(fold_aucs)) + '\n')
        f.write('逐折 ACC: ' + ', '.join(f'fold{i}={a:.4f}' for i, a in enumerate(fold_accs)) + '\n')
        f.write('训练时间(秒/折): ' + ', '.join(
            f'fold{i}={t:.1f}' for i, t in enumerate(fold_times)) + '\n')
        total_time = sum(fold_times)
        f.write(f'总训练时间: {total_time:.1f} 秒 ({total_time / 3600:.2f} 小时)\n')
        if total_epochs > 0:
            f.write(f'平均每 epoch: {total_time / total_epochs:.1f} 秒 (5 折共 {total_epochs} epochs)\n')
    print(f'结果已保存 -> GADG-KT_{dataset}_results.txt')

    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
