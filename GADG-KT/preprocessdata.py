import ast
import os
import sys
import numpy as np
import pandas as pd
import time

# ========== 数据集配置 ==========
# 添加新数据集只需在这里新增一条配置即可
# 交互表没有知识点列时，用 meta_file/meta_problem/meta_skill 指向题目元数据表，
# 知识点将从元数据表读取（每题一个知识点列表，见 "NeurIPS 2020" 配置）
DATASET_CONFIGS = {
    "assist09": {
        "csv_file": "skill_builder_data_corrected_collapsed.csv",
        "encoding": "ISO-8859-1",
        "col_order":   "order_id",       # 排序/去重用的ID列
        "col_user":    "user_id",        # 用户ID
        "col_problem": "problem_id",     # 题目ID
        "col_skill":   "skill_id",       # 技能ID（复合技能用_分隔）
        "col_correct": "correct",        # 答题结果（0/1）
        "col_original": "original",      # 是否原始题目（1=是）
    },
    "assist12": {
        "csv_file": "2012-2013-data-with-predictions-4-final.csv",
        "encoding": "ISO-8859-1",
        "col_order":   "problem_log_id",
        "col_user":    "user_id",
        "col_problem": "problem_id",
        "col_skill":   "skill_id",       # assist12同时有skill和skill_id，用skill_id
        "col_correct": "correct",
        "col_original": "original",
    },
    "NeurIPS 2020": {
        "csv_file": "train_task_3_4.csv",
        "encoding": "utf-8",
        "col_order":   "AnswerId",       # 答题记录唯一递增 id，按它还原交互时序
        "col_user":    "UserId",         # 用户ID
        "col_problem": "QuestionId",     # 题目ID
        "col_skill":   None,             # 交互表无知识点列，知识点来自 meta_file
        "col_correct": "IsCorrect",      # 答题结果（0/1）
        "col_original": None,            # 无脚手架题目列
        # 题目→知识点映射表（交互表没有知识点列的数据集使用）
        "meta_file":    "question_metadata_task_3_4.csv",
        "meta_problem": "QuestionId",
        "meta_skill":   "SubjectId",     # 值为 "[3, 71, 98, 209]" 形式的列表字符串
    },
    # "mydata": {
    #     "csv_file": "my_raw_data.csv",
    #     "encoding": "utf-8",
    #     "col_order":   "order_id",
    #     "col_user":    "user_id",
    #     "col_problem": "problem_id",
    #     "col_skill":   "skill_id",
    #     "col_correct": "correct",
    #     "col_original": "original",
    # },
}

# ========== 选择数据集 ==========
dataname = "assist09"  # ← 改这里切换数据集

# 序列最大长度：与 DGKT-SC/main.py、other model/main_baseline.py 的 max_seq=200 保持一致。
# 预处理时按此长度把超长会话切成多段（pyKT 风格：先保留头部余段，再把尾部按 MAX_SEQ 等分），
# 输出文件不再有超长行，下游加载时也不会因整条会话同时驻留内存而 OOM。
MAX_SEQ = 200

# 学生最少交互数：剔除交互总数少于此值的学生（默认 3，改这里调整；设 0 或 1 表示不过滤）
MIN_INTER = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", dataname)


# ========== 工具函数 ==========
def count(say, df, cfg):
    msg = "%s, 记录数: %d, 学生数: %d, 题目数: %d" % (
        say, len(df),
        len(df[cfg["col_user"]].unique()),
        len(df[cfg["col_problem"]].unique()))
    if cfg["col_skill"] is not None and cfg["col_skill"] in df.columns:
        msg += ", 技能数: %d" % len(df[cfg["col_skill"]].unique())
    print(msg)


def save_graph(tuple_set, file_path, names):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("%s\n" % ','.join(names))
        for tp in tuple_set:
            f.write("%s\n" % ','.join([str(e) for e in tp]))


def segment_seq(seq, max_len=MAX_SEQ):
    """把一条会话切成长度 <= max_len 的段（与 load_data.py 的 KT_Dataset 切分规则一致：
    超出 max_len 时，先保留头部余段（num % max_len），再把尾部按 max_len 等分；
    不超出时原样返回单段）。"""
    num = len(seq)
    if num <= max_len:
        return [seq]
    n_full = num // max_len
    head = num - n_full * max_len
    parts = []
    if head > 0:
        parts.append(seq[:head])
    tail = seq[head:]
    for i in range(n_full):
        parts.append(tail[i * max_len:(i + 1) * max_len])
    return parts


def load_question_skills(cfg):
    """从题目元数据表读取 题目 → 知识点列表 映射（知识点统一为字符串 id）。
    适用于交互表本身没有知识点列、知识点存放在单独元数据表的数据集（如 NeurIPS 2020）。"""
    meta = pd.read_csv(os.path.join(DATA_DIR, cfg['meta_file']))
    q2s = {}
    for q, s in zip(meta[cfg['meta_problem']], meta[cfg['meta_skill']]):
        skills = [str(int(x)) for x in str(s).strip('[]').split(',') if x.strip() != '']
        q2s[q] = skills
    return q2s


# ========== 数据清洗 ==========
def process_csv(csv_path, cfg):
    print('### 1. 清洗数据 ###')
    # 只读取需要的列，大幅降低内存占用（对超大 CSV 尤其重要）
    need_cols = [c for c in [cfg["col_order"], cfg["col_user"], cfg["col_problem"],
                             cfg["col_skill"], cfg["col_correct"], cfg["col_original"],
                             "skill_name"] if c]
    all_cols = pd.read_csv(csv_path, nrows=0, encoding=cfg["encoding"]).columns
    usecols = [c for c in need_cols if c in all_cols]
    df = pd.read_csv(csv_path, low_memory=False, encoding=cfg["encoding"], usecols=usecols)
    count('原始数据', df, cfg)

    order_col = cfg["col_order"]

    # 去重
    df.drop_duplicates(subset=[order_col], keep='first', inplace=True)
    count('去重后', df, cfg)

    # 排序
    df.sort_values(by=[order_col], ascending=True, inplace=True)

    # 移除空技能（交互表有技能列的数据集才做；没有技能列的知识点来自元数据表）
    skill_col = cfg["col_skill"]
    if skill_col is not None and skill_col in df.columns:
        df.dropna(subset=[skill_col], inplace=True)
        count('移除空技能后', df, cfg)

    # 移除非原始题目（scaffolding；没有脚手架题目的数据集跳过）
    original_col = cfg["col_original"]
    if original_col and original_col in df.columns:
        df = df[df[original_col].isin([1])]
        count('移除脚手架题目后', df, cfg)
    elif original_col:
        print(f'⚠ 未找到列 "{original_col}"，跳过脚手架过滤')

    # 剔除交互数过少的学生（MIN_INTER 默认 3，见文件顶部常量）
    if MIN_INTER > 1:
        user_col = cfg["col_user"]
        user_cnt = df[user_col].value_counts()
        keep_users = user_cnt[user_cnt >= MIN_INTER].index
        df = df[df[user_col].isin(keep_users)]
        count('移除低交互学生后', df, cfg)

    return df


# ========== 划分训练/测试集 ==========
def split_train_test_df(df, cfg, train_user_ratio=0.8):
    print('\n### 2. 划分训练/测试集 ###')
    user_col = cfg["col_user"]
    problem_col = cfg["col_problem"]

    all_users = list(df[user_col].unique())
    num_train_user = int(len(all_users) * train_user_ratio)

    train_users = list(np.random.choice(all_users, size=num_train_user, replace=False))
    test_users = list(set(all_users) - set(train_users))

    train_df = df[df[user_col].isin(train_users)]
    test_df = df[df[user_col].isin(test_users)]

    # 移除测试集中训练集没有的题目
    train_questions = list(train_df[problem_col].unique())
    test_df = test_df[test_df[problem_col].isin(train_questions)]

    print(f'训练用户: {len(train_users)}, 测试用户: {len(test_users)}')
    return train_df, test_df


# ========== 编码实体 ==========
def encode_entity(train_df, test_df, cfg, skill_values=None, extra_skills=None):
    print('\n### 3. 编码实体 ###')
    problem_col = cfg["col_problem"]
    skill_col = cfg["col_skill"]
    user_col = cfg["col_user"]

    df = pd.concat([train_df, test_df], ignore_index=True)

    # 编码题目
    problems = df[problem_col].unique()
    question_id_dict = dict(zip(problems, range(len(problems))))
    print('题目数量: %d' % len(problems))

    # 编码技能：优先用传入的技能值（知识点来自元数据表）；否则从交互表技能列取，
    # 并把复合技能（'_' 连接）拆成原子技能一并编码
    if skill_values is None:
        skills = df[skill_col].unique()
        skill_set = set(skills)
        for skill in skills:
            for s in str(skill).split('_'):
                skill_set.add(s)
    else:
        skill_set = set(skill_values)
        if extra_skills:
            skill_set |= set(extra_skills)

    index, skill_id_dict = 0, dict()
    for skill in sorted(skill_set, key=str):
        if '_' not in str(skill):
            skill_id_dict[skill] = index
            index += 1
    for skill in sorted(skill_set, key=str):
        if skill not in skill_id_dict:
            skill_id_dict[skill] = index
            index += 1
    print('技能数量: %d' % len(skill_id_dict))

    # 编码训练用户
    train_users = train_df[user_col].unique()
    train_user_id_dict = dict(zip(train_users, range(len(train_users))))
    print('训练用户数: %d' % len(train_users))

    return question_id_dict, skill_id_dict


# ========== 训练学生划分 5 折（pyKT 风格交叉验证） ==========
def assign_folds(train_df, cfg, n_folds=5, seed=42):
    user_col = cfg["col_user"]
    train_users = train_df[user_col].unique()
    rng = np.random.RandomState(seed)
    fold_ids = rng.randint(0, n_folds, size=len(train_users))
    print('训练学生分 %d 折，每折约 %d 人' % (n_folds, len(train_users) // n_folds))
    return dict(zip(train_users, fold_ids))


# ========== 生成用户序列（原始） ==========
def generate_user_sequence(df, seq_file, cfg, fold_dict=None, fold_file=None, skill_col=None):
    user_col = cfg["col_user"]
    order_col = cfg["col_order"]
    problem_col = cfg["col_problem"]
    if skill_col is None:
        skill_col = cfg["col_skill"]
    correct_col = cfg["col_correct"]

    # 一次性按用户+顺序列排序，分组后组内已有序，无需再逐组排序
    df = df.sort_values(by=[user_col, order_col], ascending=True)
    fold_list = []
    # 逐用户流式写入中间文件，不把全部用户序列同时留在内存里
    with open(os.path.join(DATA_DIR, seq_file), 'w', encoding='utf-8') as f:
        # 注意：groupby 传入标量而不是单元素列表，避免 pandas 弃用警告及兼容性问题
        for user_id, tmp_inter in df.groupby(user_col, sort=False):
            tmp_problems = list(tmp_inter[problem_col])
            tmp_skills = list(tmp_inter[skill_col])
            tmp_ans = list(tmp_inter[correct_col])
            f.write('%s\n' % str([len(tmp_inter)]))
            f.write('%s\n' % str(tmp_skills))
            f.write('%s\n' % str(tmp_problems))
            f.write('%s\n' % str(tmp_ans))
            if fold_dict is not None:
                # 与 encode_user_sequence 的分段一致：该用户被切成几块就写几条折号，
                # 保证 train_fold.txt 与 train_question.txt 逐块对齐（下游按位置过滤）
                n_blocks = len(segment_seq(range(len(tmp_inter))))
                fold_list.extend([fold_dict[user_id]] * n_blocks)
    if fold_dict is not None:
        with open(os.path.join(DATA_DIR, fold_file), 'w', encoding='utf-8') as f:
            for fl in fold_list:
                f.write('%d\n' % fl)


# ========== 编码用户序列 → 最终文件（按 MAX_SEQ 分段） ==========
def encode_user_sequence(train_or_test, question_id_dict, skill_id_dict):
    with open(os.path.join(DATA_DIR, '%s_data.txt' % train_or_test), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 逐用户读入 → 编码 → 分段 → 立即写入：峰值内存只与单个用户相关，与总交互数无关。
    # 输出文件统一 utf-8，避免 Windows 中文系统默认 GBK 编码引发的问题
    with open(os.path.join(DATA_DIR, '%s_question.txt' % train_or_test), 'w', encoding='utf-8') as wq, \
            open(os.path.join(DATA_DIR, '%s_skill.txt' % train_or_test), 'w', encoding='utf-8') as ws:
        index = 0
        while index + 3 < len(lines):
            tmp_skills = [skill_id_dict[ele] for ele in ast.literal_eval(lines[index + 1])]
            tmp_pro = [question_id_dict[ele] for ele in ast.literal_eval(lines[index + 2])]
            tmp_ans = ast.literal_eval(lines[index + 3])
            index += 4

            for seg_skill, seg_pro, seg_ans in zip(segment_seq(tmp_skills),
                                                   segment_seq(tmp_pro),
                                                   segment_seq(tmp_ans)):
                assert len(seg_skill) == len(seg_pro) == len(seg_ans)
                # 写入 question 文件（3 行块：长度 / 题目 / 作答）
                wq.write('%d\n' % len(seg_pro))
                wq.write('%s\n' % ','.join([str(i) for i in seg_pro]))
                wq.write('%s\n' % ','.join([str(int(i)) for i in seg_ans]))
                # 写入 skill 文件（3 行块：长度 / 技能 / 作答）
                ws.write('%d\n' % len(seg_skill))
                ws.write('%s\n' % ','.join([str(i) for i in seg_skill]))
                ws.write('%s\n' % ','.join([str(int(i)) for i in seg_ans]))


# ========== 构建题目-技能映射 ==========
def build_ques_skill(train_df, test_df, question_id_dict, skill_id_dict, cfg, q2s=None):
    print('\n### 4. 构建题目-技能映射 ###')
    problem_col = cfg["col_problem"]

    ques_skill_set = set()
    if q2s is not None:
        # 知识点来自题目元数据表：每题 → 其全部知识点（多知识点即多行）
        for ques, skills in q2s.items():
            if ques not in question_id_dict:
                continue
            quesID = question_id_dict[ques]
            for s in skills:
                ques_skill_set.add((quesID, skill_id_dict[s]))
    else:
        skill_col = cfg["col_skill"]
        df = pd.concat([train_df, test_df], ignore_index=True)
        for ques in question_id_dict.keys():
            quesID = question_id_dict[ques]
            tmp_df = df[df[problem_col] == ques]
            tmp_df_0 = tmp_df.iloc[0]
            tmp_skills = [ele for ele in str(tmp_df_0[skill_col]).split('_')]
            for s in tmp_skills:
                skillID = skill_id_dict[s]
                ques_skill_set.add((quesID, skillID))

    save_graph(ques_skill_set, os.path.join(DATA_DIR, 'ques_skill.csv'), ['ques', 'skill'])
    print('ques_skill 条目数: %d' % len(ques_skill_set))


# ========== 新增输出：知识点映射 / 题目难度 ==========
def save_skill_map(skill_id_dict, df, cfg, file_path, skill_col=None):
    # 知识点编码 id -> 原始 skill_id 字符串 + 知识点名称（skill_name）
    if skill_col is None:
        skill_col = cfg["col_skill"]
    name_map = {}
    if "skill_name" in df.columns:
        for sid, name in zip(df[skill_col], df["skill_name"]):
            key = str(sid)
            if key not in name_map:
                name_map[key] = name
    rows = []
    for skill, sid in sorted(skill_id_dict.items(), key=lambda kv: kv[1]):
        key = str(skill)
        name = name_map.get(key, "")
        if pd.isna(name):
            name = ""
        rows.append((sid, key, name))
    out = pd.DataFrame(rows, columns=["skill_id", "skill", "skill_name"])
    out.to_csv(file_path, index=False, encoding="utf-8")
    print('知识点映射: %d 条 -> %s' % (len(out), file_path))


def save_ques_difficulty(df, question_id_dict, cfg, file_path):
    # 每道题的难度 = 1 - 答对人数比例（难度越大题越难）
    # 注意：调用时请传训练集 df，避免用测试集标签统计（泄漏）
    problem_col = cfg["col_problem"]
    correct_col = cfg["col_correct"]
    rate = df.groupby(problem_col)[correct_col].mean()
    rows = [(question_id_dict[p], round(float(1.0 - rate[p]), 6)) for p in question_id_dict]
    out = pd.DataFrame(rows, columns=["ques", "difficulty"]).sort_values("ques")
    out.to_csv(file_path, index=False, encoding="utf-8")
    print('题目难度: %d 条 -> %s' % (len(out), file_path))


# ========== 主流程 ==========
if __name__ == '__main__':
    # Windows 下从 cmd/重定向运行时 stdout 可能是 GBK，防止打印 emoji 时报错
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    t = time.time()

    # 加载配置
    if dataname not in DATASET_CONFIGS:
        print(f"❌ 未知数据集: {dataname}")
        print(f"   可用: {list(DATASET_CONFIGS.keys())}")
        print(f"   请在 DATASET_CONFIGS 中添加配置")
        exit(1)

    cfg = DATASET_CONFIGS[dataname]
    csv_path = os.path.join(DATA_DIR, cfg["csv_file"])

    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        exit(1)

    print(f"数据集: {dataname}")
    print(f"配置: {cfg}")
    print(f"读取: {csv_path}")
    print(f"输出: {DATA_DIR}\n")

    # 1. 清洗
    DF = process_csv(csv_path, cfg)

    # 知识点来源：交互表无知识点列的数据集（如 NeurIPS 2020）从题目元数据表取
    q2s = None
    if cfg.get("meta_file"):
        q2s = load_question_skills(cfg)
        # 交互级知识点 = 该题知识点组合（复合技能，'_' 连接；单知识点即原子技能）
        compound_of = {q: '_'.join(sorted(skills)) for q, skills in q2s.items()}
        DF['_skill'] = DF[cfg["col_problem"]].map(compound_of)
        n_before = len(DF)
        DF = DF[DF['_skill'].notna()]
        if len(DF) < n_before:
            print('移除元数据中无知识点记录的题目交互: %d 条' % (n_before - len(DF)))
        skill_col = '_skill'
        skill_values = sorted(compound_of.values())
        extra_skills = sorted({s for skills in q2s.values() for s in skills})
    else:
        skill_col = cfg["col_skill"]
        skill_values = None
        extra_skills = None

    # 2. 划分
    np.random.seed(42)
    trainDF, testDF = split_train_test_df(DF, cfg)

    # 3. 编码
    question_id_dict, skill_id_dict = encode_entity(trainDF, testDF, cfg,
                                                    skill_values=skill_values,
                                                    extra_skills=extra_skills)

    # 3.5 训练学生分 5 折（pyKT 风格交叉验证）
    fold_dict = assign_folds(trainDF, cfg)

    # 4. 构建题目-技能映射
    build_ques_skill(trainDF, testDF, question_id_dict, skill_id_dict, cfg, q2s=q2s)

    # 5. 新增输出：知识点映射 + 题目难度（1-答对比例，仅训练集统计）
    print('\n### 5. 生成知识点映射 / 题目难度 ###')
    save_skill_map(skill_id_dict, DF, cfg, os.path.join(DATA_DIR, 'skill_map.csv'),
                   skill_col=skill_col)
    save_ques_difficulty(trainDF, question_id_dict, cfg, os.path.join(DATA_DIR, 'ques_difficulty.csv'))

    # 6. 生成原始用户序列
    print('\n### 6. 生成用户序列 ###')
    generate_user_sequence(trainDF, 'train_data.txt', cfg, fold_dict, 'train_fold.txt', skill_col=skill_col)
    generate_user_sequence(testDF, 'test_data.txt', cfg, skill_col=skill_col)

    # 7. 编码为最终文件（按 MAX_SEQ 分段）
    print('\n### 7. 编码为最终文件（按 %d 分段） ###' % MAX_SEQ)
    encode_user_sequence('train', question_id_dict, skill_id_dict)
    encode_user_sequence('test', question_id_dict, skill_id_dict)

    # 8. 清理中间文件
    for f in ['train_data.txt', 'test_data.txt']:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            os.remove(p)

    print(f'\n✅ 完成! 耗时 {time.time() - t:.0f} 秒')
    print(f'\n生成的文件 ({DATA_DIR}):')
    for f in ['ques_skill.csv', 'skill_map.csv', 'ques_difficulty.csv',
              'train_question.txt', 'test_question.txt',
              'train_skill.txt', 'test_skill.txt', 'train_fold.txt']:
        print(f'  ✔ {f}')
