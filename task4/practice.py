"""第八章独立算法练习；不连接模型服务或数据库。"""
import math
import re
from dataclasses import dataclass


def cosine(a, b):
    if not a or len(a) != len(b):
        raise ValueError("向量维度必须相同且非空")
    if any(not math.isfinite(x) for x in (*a, *b)):
        raise ValueError("向量必须为有限数值")
    denominator = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / denominator if denominator else 0.0


def semantic_chunks(text, embed, threshold=0.55, max_chars=600):
    """相邻句语义相似度下降时切块，保留原文；超长单句按长度兜底。"""
    if max_chars < 1 or not -1 <= threshold <= 1:
        raise ValueError("分块参数无效")
    sentences = [m.group() for m in re.finditer(r".*?(?:[。！？!?]\s*|\n+|$)", text, re.S)
                 if m.group()]
    units = [s[i:i+max_chars] for s in sentences for i in range(0, len(s), max_chars)]
    if not units:
        return []
    vectors = embed(units)
    if len(vectors) != len(units):
        raise ValueError("编码结果数量不匹配")
    # 验证所有向量，包括只有一个句子的情况。
    for vector in vectors:
        cosine(vector, vector)
    chunks, current = [], units[0]
    for i in range(1, len(units)):
        if len(current) + len(units[i]) > max_chars or cosine(vectors[i-1], vectors[i]) < threshold:
            chunks.append(current)
            current = units[i]
        else:
            current += units[i]
    chunks.append(current)
    return chunks


@dataclass(frozen=True)
class Memory:
    id: str
    importance: float
    accesses: int
    idle_days: float
    pinned: bool = False


def retention_score(memory, half_life=30):
    """越高越应保留；访问次数先归一化，时间采用半衰期衰减。"""
    if (not math.isfinite(memory.importance) or not 0 <= memory.importance <= 1
            or memory.accesses < 0 or not math.isfinite(memory.idle_days)
            or memory.idle_days < 0 or not math.isfinite(half_life) or half_life <= 0):
        raise ValueError("记忆或半衰期参数无效")
    frequency = min(1.0, math.log1p(memory.accesses) / math.log1p(100))
    recency = 2 ** (-memory.idle_days / half_life)
    return 0.5 * memory.importance + 0.2 * frequency + 0.3 * recency


def archive_candidates(memories, threshold=0.35, max_count=10):
    """只提出归档候选，不执行删除；固定记忆不参与自动淘汰。"""
    if not 0 <= threshold <= 1 or max_count < 0:
        raise ValueError("归档参数无效")
    ranked = [(retention_score(m), m.id) for m in memories if not m.pinned]
    return [mid for score, mid in sorted(ranked) if score < threshold][:max_count]
