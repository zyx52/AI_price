"""
GNN 景点/项目客流关联建模

应用场景:
  1. 项目间客流迁移预测(A项目排队长 → B/C项目会如何被连带影响)
  2. 套餐组合推荐(哪些项目+周边组合能最大化单客毛利)
  3. 路径引导优化(为游客推荐最优游园路径,缓解拥堵)

数据结构:
  节点: 景点/项目/餐饮点/商店  (N个)
  边:   连续两个项目的客流迁移关系(带权有向图)
  特征: 每个节点的 [排队时长, 热度, 客群画像, 时间段]

架构:
  - 两层 GCN(图卷积网络)  —— 学习节点表征
  - 节点级预测: 预测每个项目下一时段的客流
  - 图级推荐: 为给定客群推荐最优路径

使用:
  需要 PyTorch + PyTorch Geometric:
    pip install torch torch-geometric

  若未安装,会自动降级为 NetworkX 图结构 + 简单启发式推荐。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("ParkGNN")


# 选择性导入
try:
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]
    from torch_geometric.nn import GCNConv  # type: ignore[import-not-found]
    from torch_geometric.data import Data  # type: ignore[import-not-found]
    _HAS_TORCH_GEO = True
except ImportError:
    _HAS_TORCH_GEO = False
    logger.info("torch-geometric 未安装,GNN将降级为NetworkX启发式版本")

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Attraction:
    """景点/项目节点"""
    id: str
    name: str
    category: str          # 'ride' | 'show' | 'food' | 'shop' | 'indoor_ride'
    capacity_per_hour: int # 每小时承载量
    avg_duration_min: int  # 平均体验时长
    location: Tuple[float, float]  # 园内坐标
    indoor: bool = False   # 是否室内(雨天优选)
    family_friendly: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class RouteRecommendation:
    """路径推荐结果"""
    segment_names: List[str]
    expected_total_wait_min: float
    expected_experience_score: float
    target_segment: str
    reasoning: str


# ============================================================
# GCN 模型(PyTorch Geometric版)
# ============================================================
if _HAS_TORCH_GEO:
    class AttractionGCN(torch.nn.Module):  # type: ignore[name-defined]
        """两层GCN学习节点表征,预测下一时段客流"""

        def __init__(self, in_channels: int, hidden: int = 32, out_channels: int = 1):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)  # type: ignore[name-defined]
            self.conv2 = GCNConv(hidden, hidden)  # type: ignore[name-defined]
            self.head = torch.nn.Linear(hidden, out_channels)  # type: ignore[name-defined]

        def forward(self, x, edge_index, edge_weight=None):  # type: ignore[name-defined]
            h = self.conv1(x, edge_index, edge_weight)  # type: ignore[name-defined]
            h = F.relu(h)  # type: ignore[name-defined]
            h = F.dropout(h, p=0.2, training=self.training)  # type: ignore[name-defined]
            h = self.conv2(h, edge_index, edge_weight)  # type: ignore[name-defined]
            h = F.relu(h)  # type: ignore[name-defined]
            out = self.head(h)  # type: ignore[name-defined]
            return out, h  # 返回预测值 + 节点embedding


# ============================================================
# 主类: 景点图建模
# ============================================================
class ParkAttractionGraph:
    """乐园景点关联图"""

    def __init__(self):
        self.attractions: Dict[str, Attraction] = {}
        self.flow_matrix: Optional[np.ndarray] = None  # 客流迁移概率矩阵 N×N
        self.node_idx: Dict[str, int] = {}
        self.gcn_model = None
        self.node_embeddings: Optional[np.ndarray] = None

    # ---------- 构图 ----------
    def add_attraction(self, att: Attraction):
        self.attractions[att.id] = att

    def build_default_park(self):
        """默认构建一个示例乐园,便于演示"""
        default = [
            Attraction("A1", "过山车·极速", "ride", 1200, 3, (0.2, 0.8), False, False),
            Attraction("A2", "摩天轮", "ride", 800, 15, (0.5, 0.9), False, True),
            Attraction("A3", "旋转木马", "ride", 1500, 5, (0.3, 0.5), False, True),
            Attraction("A4", "海盗船", "ride", 900, 4, (0.7, 0.7), False, False),
            Attraction("B1", "4D影院", "show", 400, 20, (0.5, 0.3), True, True),
            Attraction("B2", "室内演艺秀", "show", 600, 40, (0.4, 0.2), True, True),
            Attraction("C1", "花车巡游", "show", 3000, 30, (0.5, 0.5), False, True),
            Attraction("D1", "主题餐厅", "food", 500, 45, (0.2, 0.3), True, True),
            Attraction("D2", "快餐档", "food", 1500, 15, (0.8, 0.4), True, True),
            Attraction("E1", "纪念品商店", "shop", 1000, 20, (0.5, 0.4), True, True),
            Attraction("F1", "儿童互动区", "indoor_ride", 800, 30, (0.3, 0.7), True, True),
            Attraction("F2", "水世界", "ride", 1200, 60, (0.8, 0.8), False, True),
        ]
        for a in default:
            self.add_attraction(a)
        self._build_flow_matrix_heuristic()
        return self

    def _build_flow_matrix_heuristic(self):
        """
        启发式构建客流迁移矩阵:
          - 距离越近,迁移概率越高
          - 同类型项目之间迁移概率较低(游客不会连续坐两个过山车)
          - 玩完刺激项目后会去餐饮/休闲
        """
        atts = list(self.attractions.values())
        n = len(atts)
        self.node_idx = {a.id: i for i, a in enumerate(atts)}
        M = np.zeros((n, n))

        for i, a in enumerate(atts):
            for j, b in enumerate(atts):
                if i == j:
                    continue
                # 空间距离
                dist = np.hypot(a.location[0] - b.location[0], a.location[1] - b.location[1])
                dist_score = np.exp(-dist * 2.0)
                # 类别兼容性
                if a.category == b.category and a.category == "ride":
                    cat_score = 0.4  # 玩完一个刺激项目不会马上玩另一个
                elif a.category == "ride" and b.category == "food":
                    cat_score = 1.5  # 刺激项目后会吃饭
                elif a.category == "food" and b.category == "ride":
                    cat_score = 1.2
                elif a.category == "ride" and b.category == "show":
                    cat_score = 1.3
                elif b.category == "shop":
                    cat_score = 0.9  # 随时可能去商店
                else:
                    cat_score = 1.0
                M[i, j] = dist_score * cat_score

        # 每行归一化为概率
        row_sum = M.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        self.flow_matrix = M / row_sum

    # ---------- 预测下一时段客流(GNN版) ----------
    def predict_next_flows(
        self,
        current_queues: Dict[str, float],  # 当前各项目排队时长(分钟)
        weather: str = "晴好",
    ) -> Dict[str, float]:
        """
        预测下一时段各项目的客流量变化
        基于客流迁移矩阵 + 天气调节(雨天增加室内权重)
        """
        if self.flow_matrix is None:
            self._build_flow_matrix_heuristic()

        atts = list(self.attractions.values())
        n = len(atts)
        current_vec = np.array([current_queues.get(a.id, 20.0) for a in atts])

        # 天气调节: 雨天 → 室内项目迁移权重提升
        base_flow_matrix = self.flow_matrix if self.flow_matrix is not None else np.zeros((n, n))
        M = base_flow_matrix.copy()
        if weather in ("雨", "暴雨"):
            for j, b in enumerate(atts):
                if b.indoor:
                    M[:, j] *= 1.8  # 雨天室内吸引力增强
        # 重新归一化
        row_sum = M.sum(axis=1, keepdims=True); row_sum[row_sum == 0] = 1
        M = M / row_sum

        # 下一时段客流 = 当前客流 × 迁移矩阵
        next_vec = current_vec @ M
        return {a.id: float(next_vec[i]) for i, a in enumerate(atts)}

    def queue_heat_embedding(
        self,
        current_queues: Optional[Dict[str, float]] = None,
        weather: str = "晴好",
        dim: int = 6,
    ) -> np.ndarray:
        """
        输出给RL使用的园内拥挤度Embedding。
        优先用 GCN node_embeddings + 实时排队热度做加权聚合,否则降级为统计特征向量。
        """
        atts = list(self.attractions.values())
        if not atts:
            return np.zeros(dim, dtype=np.float32)

        queues = current_queues or {a.id: 20.0 for a in atts}
        q_vec = np.array([float(queues.get(a.id, 20.0)) for a in atts], dtype=np.float32)
        q_norm = q_vec / max(float(q_vec.max()), 1.0)

        # 有训练好的节点嵌入时: 用排队热度做加权池化
        if self.node_embeddings is not None and len(self.node_embeddings) == len(atts):
            emb = self.node_embeddings
            weighted = (q_norm[:, None] * emb).sum(axis=0) / max(float(q_norm.sum()), 1e-6)
            vec = weighted.astype(np.float32)
        else:
            next_flows = self.predict_next_flows(queues, weather)
            flow_vec = np.array([float(next_flows.get(a.id, 0.0)) for a in atts], dtype=np.float32)
            indoor_mask = np.array([1.0 if a.indoor else 0.0 for a in atts], dtype=np.float32)

            vec = np.array([
                float(np.mean(q_vec) / 120.0),
                float(np.std(q_vec) / 60.0),
                float(np.percentile(q_vec, 90) / 180.0),
                float(np.mean(flow_vec) / 120.0),
                float(np.sum(flow_vec * indoor_mask) / max(np.sum(flow_vec), 1.0)),
                1.0 if weather in ("雨", "暴雨") else 0.0,
            ], dtype=np.float32)

        if len(vec) >= dim:
            return vec[:dim]
        pad = np.zeros(dim - len(vec), dtype=np.float32)
        return np.concatenate([vec, pad]).astype(np.float32)

    # ---------- GCN 训练(可选) ----------
    def train_gcn(
        self,
        history_snapshots: List[Dict[str, float]],  # 每个时段各节点的客流快照
        epochs: int = 200,
    ):
        """
        用历史时序快照训练GCN预测下一时段客流
        history_snapshots: [{att_id: queue_min}, ...]
        """
        if not _HAS_TORCH_GEO:
            logger.warning("torch-geometric 不可用,跳过GCN训练,使用启发式版本")
            return

        import torch  # type: ignore[import-not-found]

        atts = list(self.attractions.values())
        n = len(atts)
        # 构造边索引(基于迁移矩阵的top-k邻居)
        if self.flow_matrix is None:
            self._build_flow_matrix_heuristic()
        edge_list, edge_w = [], []
        flow_matrix = self.flow_matrix if self.flow_matrix is not None else np.zeros((n, n))
        for i in range(n):
            top3 = np.argsort(flow_matrix[i])[-3:]
            for j in top3:
                if i != j:
                    edge_list.append([i, int(j)])
                    edge_w.append(float(flow_matrix[i, j]))
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # type: ignore[name-defined]
        edge_weight = torch.tensor(edge_w, dtype=torch.float)  # type: ignore[name-defined]

        # 节点特征
        cat_map = {"ride": 0, "show": 1, "food": 2, "shop": 3, "indoor_ride": 4}
        X_static = torch.tensor([  # type: ignore[name-defined]
            [cat_map[a.category], a.capacity_per_hour / 2000.0,
             a.avg_duration_min / 60.0, float(a.indoor), float(a.family_friendly)]
            for a in atts
        ], dtype=torch.float)  # type: ignore[name-defined]

        # 监督数据: (t时刻排队时长) → (t+1时刻排队时长)
        pairs = []
        for t in range(len(history_snapshots) - 1):
            x_t = torch.tensor([[history_snapshots[t].get(a.id, 20.0) / 60.0] for a in atts], dtype=torch.float)  # type: ignore[name-defined]
            y_t = torch.tensor([[history_snapshots[t + 1].get(a.id, 20.0) / 60.0] for a in atts], dtype=torch.float)  # type: ignore[name-defined]
            pairs.append((x_t, y_t))

        if not pairs:
            logger.warning("历史快照不足,无法训练GCN")
            return

        self.gcn_model = AttractionGCN(in_channels=X_static.shape[1] + 1, hidden=32)  # type: ignore[name-defined]
        opt = torch.optim.Adam(self.gcn_model.parameters(), lr=1e-2)  # type: ignore[name-defined]

        logger.info(f"GCN 训练开始 | 节点={n} | 边={edge_index.shape[1]} | 样本={len(pairs)}")
        for epoch in range(epochs):
            total_loss = 0
            for x_t, y_t in pairs:
                feat = torch.cat([X_static, x_t], dim=1)  # type: ignore[name-defined]
                pred, _ = self.gcn_model(feat, edge_index, edge_weight)  # type: ignore[name-defined]
                loss = F.mse_loss(pred, y_t)  # type: ignore[name-defined]
                opt.zero_grad()  # type: ignore[name-defined]
                loss.backward()  # type: ignore[attr-defined]
                opt.step()  # type: ignore[name-defined]
                total_loss += loss.item()  # type: ignore[attr-defined]
            if (epoch + 1) % 50 == 0:
                logger.info(f"  epoch {epoch+1}: loss={total_loss/len(pairs):.4f}")

        # 缓存节点嵌入
        self.gcn_model.eval()
        with torch.no_grad():  # type: ignore[name-defined]
            feat = torch.cat([X_static, pairs[-1][0]], dim=1)  # type: ignore[name-defined]
            _, emb = self.gcn_model(feat, edge_index, edge_weight)  # type: ignore[name-defined]
            self.node_embeddings = emb.numpy()  # type: ignore[attr-defined]
        logger.info("GCN 训练完成,已缓存节点嵌入")

    # ---------- 路径推荐 ----------
    def recommend_route(
        self,
        current_queues: Dict[str, float],
        target_segment: str = "家庭亲子",
        weather: str = "晴好",
        max_attractions: int = 5,
    ) -> RouteRecommendation:
        """
        基于迁移矩阵 + 当前排队状况,为指定客群推荐最优游园路径
        贪心策略: 从入口出发,每步选择 "低排队 + 高兼容" 的下一站
        """
        atts = list(self.attractions.values())

        # 客群偏好过滤
        if target_segment == "家庭亲子":
            candidates = [a for a in atts if a.family_friendly]
        elif target_segment == "年轻客群":
            candidates = [a for a in atts if a.category in ("ride", "show")]
        else:
            candidates = list(atts)

        # 雨天优选室内
        if weather in ("雨", "暴雨"):
            candidates = sorted(candidates, key=lambda a: (not a.indoor, current_queues.get(a.id, 20)))
        else:
            candidates = sorted(candidates, key=lambda a: current_queues.get(a.id, 20))

        # 贪心选路径
        selected = []
        visited = set()
        current = candidates[0] if candidates else None
        total_wait = 0.0

        while current and len(selected) < max_attractions:
            selected.append(current)
            visited.add(current.id)
            total_wait += current_queues.get(current.id, 20.0) + current.avg_duration_min

            # 从迁移矩阵找下一站
            if self.flow_matrix is None:
                break
            cur_idx = self.node_idx[current.id]
            flow_row = self.flow_matrix[cur_idx].copy()
            for vid in visited:
                flow_row[self.node_idx[vid]] = 0
            # 结合排队时长
            scores = np.zeros_like(flow_row)
            for i, a in enumerate(atts):
                if a.id in visited: continue
                if target_segment == "家庭亲子" and not a.family_friendly: continue
                q = current_queues.get(a.id, 20.0)
                scores[i] = flow_row[i] * (1 / (1 + q / 30))
            if scores.max() <= 0:
                break
            next_idx = int(np.argmax(scores))
            current = atts[next_idx]

        exp_score = max(0, 10 - total_wait / 30)  # 简单评分:排队越少体验越高

        return RouteRecommendation(
            segment_names=[a.name for a in selected],
            expected_total_wait_min=total_wait,
            expected_experience_score=round(exp_score, 1),
            target_segment=target_segment,
            reasoning=(f"基于客流迁移矩阵为【{target_segment}】规划,"
                       f"{'优先室内项目防雨' if weather in ('雨','暴雨') else '按排队时长升序'},"
                       f"预计总耗时{total_wait:.0f}分钟"),
        )

    # ---------- 套餐关联推荐 ----------
    def suggest_bundle_attractions(
        self,
        anchor_id: str,
        top_k: int = 3,
    ) -> List[str]:
        """
        给定一个主打项目,推荐搭配的项目/餐饮/商店(用于套餐组合)
        基于迁移矩阵找出 anchor 之后最可能去的 top-k 节点
        """
        if self.flow_matrix is None or anchor_id not in self.node_idx:
            return []
        atts = list(self.attractions.values())
        idx = self.node_idx[anchor_id]
        flows = self.flow_matrix[idx]
        top = np.argsort(flows)[::-1][:top_k + 1]
        return [atts[i].name for i in top if atts[i].id != anchor_id][:top_k]
