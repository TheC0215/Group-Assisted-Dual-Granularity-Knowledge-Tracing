import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GADGKT(nn.Module):
    def __init__(self, skill_max, pro_max, embed, p, state_d=64, max_seq=200,
                 pro2skill=None, num_int=None, E_int=None, E_pro=None, E_skill=None,
                 max_interval=200):
        super(GADGKT, self).__init__()

        assert state_d == embed, '当前配置要求 state_d == embed（检索注意力需要同维度）'

        self.skill_max = skill_max
        self.pro_max = pro_max
        self.pro2skill = pro2skill
        # 粗粒度知识点图（交互版）：节点 = 知识点 + 交互，边 = 每个交互 × 其题目的每个知识点
        # 第 e 条边连交互 E_int[e] 与知识点 E_skill[e]，E_pro[e] 为该交互的题目（用于取 qa 边权）
        self.num_int = num_int
        self.E_int = E_int
        self.E_pro = E_pro
        self.E_skill = E_skill

        d = embed

        # ============ 题目编码 Q = MC + OF（不变） ============
        self.pro_embed = nn.Parameter(torch.rand(pro_max, d))
        nn.init.xavier_uniform_(self.pro_embed)

        self.skill_embed = nn.Parameter(torch.rand(skill_max, d))
        nn.init.xavier_uniform_(self.skill_embed)

        self.ans_embed = nn.Embedding(2, d)       # 做对=1，做错=0 的表示
        self.pro_diff = nn.Embedding(pro_max, 1)  # diff_{q_t} 难度标量

        self.pro_linear = nn.Linear(d, d)
        self.skill_linear = nn.Linear(d, d)
        self.pro_change = nn.Linear(d, d)

        # ============ 要求2：细粒度双 LSTM（零初始化） ============
        self.skill_lstm = nn.LSTMCell(2 * d, d)  # 知识点层：所有知识点共用
        self.ques_lstm = nn.LSTMCell(2 * d, d)   # 题目层：per-学生

        # ============ 时间间隔遗忘（检索侧，遗忘结果写回状态矩阵） ============
        # 知识点距上次作答的间隔（首次出现=0，超过 max_interval 截断）→ d 维嵌入，
        # 与状态向量拼接后经 skill_forget 得到遗忘后的知识状态
        self.max_interval = max_interval
        self.interval_embed = nn.Embedding(max_interval + 1, d)
        self.skill_forget = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, d)
        )

        # ============ 粗粒度 GNN（知识点层面：知识点+交互图，2 层 + 残差） ============
        self.skill_gnn_l1 = nn.Linear(d, d)
        self.skill_gnn_l2 = nn.Linear(d, d)
        self.fc_skill = nn.Linear(d, d)

        # ============ 标量门控 ============
        self.gate = nn.Linear(2 * d, 1)

        # ============ 预测头（AKT 风格，3d 输入，不变） ============
        self.predict = nn.Sequential(
            nn.Linear(3 * d, 512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 1)
        )

        self.dropout = nn.Dropout(p=p)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

        # ============ 图缓冲与节点特征（普通属性，不进 state_dict） ============
        self.skill_node_sum = torch.zeros(skill_max, d)   # qa 加权累积的技能状态（技能节点种子特征）
        self.skill_node_w = torch.zeros(skill_max)        # qa 权重和
        self.skill_node_feat = torch.zeros(skill_max, d)  # epoch 级冻结的技能节点特征
        self.int_state_sum = torch.zeros(num_int, d)      # 每个交互的 now_need_state 累积
        self.int_state_cnt = torch.zeros(num_int)
        self.int_node_feat = torch.zeros(num_int, d)      # epoch 级冻结的交互节点特征

    def _to_device(self, device):
        if self.skill_node_sum.device != device:
            self.skill_node_sum = self.skill_node_sum.to(device)
            self.skill_node_w = self.skill_node_w.to(device)
            self.skill_node_feat = self.skill_node_feat.to(device)
            self.int_state_sum = self.int_state_sum.to(device)
            self.int_state_cnt = self.int_state_cnt.to(device)
            self.int_node_feat = self.int_node_feat.to(device)
            self.E_int = self.E_int.to(device)
            self.E_pro = self.E_pro.to(device)
            self.E_skill = self.E_skill.to(device)

    def _run_gnn(self, qa):
        """2 层消息传递（+残差）：输入 epoch 冻结的节点特征，输出当前参数的图嵌入。
        每次前向现算，各 batch 用各自己的计算图（避免共享图二次反向报错）。
        qa 用 detach 值作边权（qa 参数本身仍经题目编码路径训练）。"""
        # 知识点图（交互版）：节点 = 知识点 + 交互（全局知识状态），边权 = 该交互题目对该知识点的 qa 权重
        # 交互节点特征 = epoch 冻结的 now_need_state（该交互预测时的细粒度检索状态）
        w = qa[self.E_pro, self.E_skill]                              # (E,) 边权
        int_h = F.relu(self.skill_gnn_l1(self.int_node_feat))         # (num_int, d) 变换交互状态
        # 知识点 ← 触及它的所有交互（边权按知识点归一化）
        denom_sk = torch.zeros(self.skill_max, device=w.device).index_add_(0, self.E_skill, w)
        ws = w / (denom_sk[self.E_skill] + 1e-8)
        agg = torch.zeros_like(self.skill_node_feat).index_add_(
            0, self.E_skill, ws.unsqueeze(-1) * int_h[self.E_int])
        skill_emb = self.skill_gnn_l2(agg) + self.skill_node_feat  # layer2 + 残差（技能种子特征）
        return skill_emb

    def forward(self, last_problem, last_ans, next_problem, next_ans, int_ids=None):
        device = last_problem.device
        self._to_device(device)

        batch = last_problem.shape[0]
        seq = last_problem.shape[1]
        d = self.pro_embed.shape[1]

        pro2skill = self.pro2skill

        # ============ 题目编码 Q = MC + OF（不变） ============
        skill_mean = torch.matmul(pro2skill, self.skill_embed) / (
                torch.sum(pro2skill, dim=-1, keepdims=True) + 1e-8)  # pro d

        pro_idx = torch.arange(self.pro_max).to(device)
        pro_diff = torch.sigmoid(self.pro_diff(pro_idx))  # pro_max 1

        q_pro = self.pro_linear(self.pro_embed)
        q_skill = self.skill_linear(self.skill_embed)
        attn = torch.matmul(q_pro, q_skill.transpose(-1, -2)) / math.sqrt(q_pro.shape[-1])
        attn = torch.masked_fill(attn, pro2skill == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)  # qa 表 (pro_max, skill_max)
        skill_attn = torch.matmul(attn, self.skill_embed)  # MC_{q_t}

        now_embed = skill_attn + pro_diff * self.pro_change(skill_mean)  # Q = MC + OF
        pro_embed = self.dropout(now_embed)

        # 粗粒度图嵌入（每次前向现算，带梯度流向 GNN 参数）
        skill_gnn_emb = self._run_gnn(attn.detach())

        next_pro_rasch = F.embedding(next_problem, pro_embed)  # batch seq d

        # ============ 细粒度 LSTM 状态（零初始化） ============
        h_skill = torch.zeros(batch, self.skill_max, d).to(device)
        c_skill = torch.zeros_like(h_skill)
        h_ques = torch.zeros(batch, d).to(device)
        c_ques = torch.zeros_like(h_ques)
        # 每个知识点"上一次作答"的位置（-1 表示从未做过，首次出现间隔记 0）
        last_seen = torch.full((batch, self.skill_max), -1, dtype=torch.long, device=device)

        res_p = []
        res_attn = []

        for now_step in range(seq):
            now_pro = next_problem[:, now_step]                            # batch
            now_pro2skill = F.embedding(now_pro, pro2skill).unsqueeze(1)   # batch 1 skill
            now_pro_embed = next_pro_rasch[:, now_step]                    # batch d
            now_ans = next_ans[:, now_step].long()                         # batch
            is_real = (now_pro != 0).unsqueeze(-1).float()                 # padding=0

            # ============ 1) 预测（用更新前的状态，答案 t 不参与） ============
            # 1.1) 当前题相关知识点：按距上次作答的间隔做遗忘，并写回状态矩阵
            rel_mask = now_pro2skill.squeeze(1) > 0            # batch skill（当前题相关技能）
            delta = now_step - last_seen                       # batch skill long
            delta = torch.where(last_seen < 0, torch.zeros_like(delta), delta)  # 首次出现间隔=0
            delta = delta.clamp(0, self.max_interval)
            iv = self.interval_embed(delta)                    # batch skill d
            h_rel = h_skill[rel_mask]                          # (K, d) 相关技能状态
            iv_rel = iv[rel_mask]  # (K, d)
            hq_rel = h_ques.unsqueeze(1).expand(-1, self.skill_max, -1)[rel_mask]  # (K, d) 全局知识状态广播

            h_forg = self.skill_forget(torch.cat([h_rel, iv_rel, hq_rel], dim=-1))  # (K, d)
            h_skill = h_skill.masked_scatter(rel_mask.unsqueeze(-1), h_forg)  # 遗忘写回状态矩阵

            # 1.2) 细粒度检索（照旧）：Q 作 query，遗忘后的 h_skill 作 K/V
            f1 = now_pro_embed.unsqueeze(1)  # batch 1 d
            now_pro_skill_attn = torch.matmul(f1, h_skill.transpose(-1, -2)) / f1.shape[-1]
            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, now_pro2skill == 0, -1e9)
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)  # batch 1 skill
            now_need_state = torch.matmul(now_pro_skill_attn, h_skill).squeeze(1)  # batch d

            # 细粒度状态 f_qt = 知识点检索 + 题目层状态（逐元素相加）
            f_qt = now_need_state + h_ques  # batch d

            # 粗粒度查表（qa 加权聚合该题知识点的知识点图嵌入）
            qa_q = F.embedding(now_pro, attn)  # batch skill（该题对各知识点的权重）
            skill_vec = torch.matmul(qa_q, skill_gnn_emb) / (qa_q.sum(-1, keepdim=True) + 1e-8)  # batch d

            coarse = self.fc_skill(skill_vec)  # batch d

            # 标量门控
            alpha = torch.sigmoid(self.gate(torch.cat([f_qt, coarse], dim=-1)))  # batch 1
            fused = torch.cat([(1 - alpha) * f_qt, alpha * coarse], dim=-1)  # batch 2d
            now_output = torch.sigmoid(self.predict(torch.cat([fused, now_pro_embed], dim=-1)))  # batch 1
            now_output = now_output.squeeze(-1)
            res_p.append(now_output)
            res_attn.append(alpha.squeeze(-1))

            # ============ 2) 更新（喂 t 的作答，只影响 t+1 及以后） ============
            # 稀疏更新：只把当前题涉及的知识点送进 LSTM（遗忘后的状态作为输入），
            # 无关知识点不进 LSTM、状态原样保留；padding 步无涉及技能，自然冻结
            upd_mask = rel_mask & (now_pro != 0).unsqueeze(-1)   # batch skill（与旧版 keep 冻结一致）
            ans_vec = self.ans_embed(now_ans).unsqueeze(1)       # batch 1 d
            sel_c_s = self.skill_embed.unsqueeze(0).expand(batch, -1, -1)[upd_mask]  # (K, d)
            sel_ans = ans_vec.expand(batch, self.skill_max, -1)[upd_mask]            # (K, d)
            sel_in = self.dropout(torch.cat([sel_c_s, sel_ans], dim=-1))             # (K, 2d)
            sel_h = h_skill[upd_mask]     # (K, d) 遗忘后的状态作为 LSTM 隐藏输入
            sel_c = c_skill[upd_mask]
            if sel_in.shape[0] > 0:
                hf, cf = self.skill_lstm(sel_in, (sel_h, sel_c))
                h_skill = h_skill.masked_scatter(upd_mask.unsqueeze(-1), hf)
                c_skill = c_skill.masked_scatter(upd_mask.unsqueeze(-1), cf)
            # 记录"上一次作答"位置（只对真正作答的题目的技能）
            last_seen = torch.where(upd_mask, torch.full_like(last_seen, now_step), last_seen)

            # 题目层输入：[Q; ans_embed]，padding 步 [0;0]
            ques_in = torch.cat([now_pro_embed, self.ans_embed(now_ans)], dim=-1) * is_real
            ques_in = self.dropout(ques_in)
            hq_new, cq_new = self.ques_lstm(ques_in, (h_ques, c_ques))
            h_ques = is_real * hq_new + (1 - is_real) * h_ques
            c_ques = is_real * cq_new + (1 - is_real) * c_ques

            # ============ 3) 累积图缓冲（训练+真实步，做完题后的状态，detach） ============
            if self.training:
                real = (now_pro != 0).float()
                contrib = (qa_q * real.unsqueeze(-1)).detach()  # batch skill
                self.skill_node_sum += torch.einsum('bk,bkd->kd', contrib, h_skill.detach())
                self.skill_node_w += contrib.sum(0)
                # 交互节点：保存做完该题后（t 时刻）的细粒度检索状态——用更新后的 h_skill 重新检索一次
                if int_ids is not None:
                    post_attn = torch.matmul(f1, h_skill.transpose(-1, -2)) / f1.shape[-1]
                    post_attn = torch.masked_fill(post_attn, now_pro2skill == 0, -1e9)
                    post_attn = torch.softmax(post_attn, dim=-1)
                    post_state = torch.matmul(post_attn, h_skill).squeeze(1)  # batch d
                    self.int_state_sum.index_add_(0, int_ids[:, now_step], post_state.detach() * real.unsqueeze(-1))
                    self.int_state_cnt.index_add_(0, int_ids[:, now_step], real)

        P = torch.vstack(res_p).T
        A = torch.vstack(res_attn).T
        return P, A

    def finalize_graph(self):
        """epoch 末：归一化本 epoch 缓冲 → 冻结为节点特征（供下一 epoch 的 _run_gnn 使用）→ 清空缓冲"""
        self.skill_node_feat = self.skill_node_sum / (self.skill_node_w.unsqueeze(-1) + 1e-8)  # skill d
        self.int_node_feat = self.int_state_sum / (self.int_state_cnt.unsqueeze(-1) + 1e-8)    # num_int d

        # 清空缓冲
        self.skill_node_sum.zero_()
        self.skill_node_w.zero_()
        self.int_state_sum.zero_()
        self.int_state_cnt.zero_()
