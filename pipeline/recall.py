"""召回率计算 — 中文汉字 SequenceMatcher，阈值 70%。"""
import difflib
import re
import unicodedata


def _extract_cn(text: str) -> str:
    """仅提取中文字符。"""
    return ''.join(ch for ch in text if '一' <= ch <= '鿿')


def _coverage(needle: str, haystack: str) -> float:
    """needle 中文被 haystack 中文覆盖的比例 (0~1)。"""
    cn_a = _extract_cn(needle)
    cn_b = _extract_cn(haystack)
    if not cn_a:
        return 0.0
    s = difflib.SequenceMatcher(None, cn_a, cn_b)
    return sum(b.size for b in s.get_matching_blocks()) / len(cn_a)


def calc_recall(correct: str, docs: list[dict], threshold: float = 0.7) -> tuple[int, int]:
    """逐 chunk 比对。返回 (召回标记, 命中位置)。"""
    if not correct or not docs:
        return 0, 0
    for i, d in enumerate(docs):
        txt = str(d.get("content_text", "") or "")
        if _coverage(correct, txt) >= threshold:
            return 1, i + 1
    return 0, 0


def calc_recall_merged(correct: str, docs: list[dict], threshold: float = 0.7) -> tuple[int, int]:
    """合并召回 = top-k 中单块覆盖 gold 的最大值。

    不再使用拼接全文的 SequenceMatcher 口径：autojunk 会把拼接长文中
    出现次数 > 总长/100 的高频汉字判为 junk（不参与匹配），导致
    「合并后 coverage 反而低于单块」的伪影（实测 19.1% 实例失真，
    最大 0.84 → 0.32）。max(逐块) 与单块召回口径一致、无虚高。
    """
    if not correct or not docs:
        return 0, 0
    best = 0.0
    for d in docs:
        txt = str(d.get("content_text", "") or "")
        if not txt:
            continue
        cov = _coverage(correct, txt)
        if cov > best:
            best = cov
    if best >= threshold:
        return 1, 0
    return 0, 0


def calc_recall_concat(correct: str, docs: list[dict], threshold: float = 0.7) -> tuple[int, int]:
    """拼接召回 = top-k 全部块拼接后对 gold 的中文覆盖（新增指标，不改动其余口径）。

    针对答案分散在多块的问题（多要点聚合题）。必须显式 autojunk=False：
    difflib 默认的 autojunk 启发式会把拼接长文中出现次数 > 总长/100 的
    高频汉字判为 junk（不参与匹配），导致「拼接后覆盖反而低于单块」的
    伪影。关闭后拼接是单块文本的超集，覆盖单调不减，无虚高。
    """
    if not correct or not docs:
        return 0, 0
    cn_a = _extract_cn(correct)
    if not cn_a:
        return 0, 0
    hay = ''.join(_extract_cn(str(d.get("content_text", "") or "")) for d in docs)
    s = difflib.SequenceMatcher(None, cn_a, hay, autojunk=False)
    cov = sum(b.size for b in s.get_matching_blocks()) / len(cn_a)
    if cov >= threshold:
        return 1, 0
    return 0, 0


def _norm_file_ref(name: str) -> str:
    """参考文件/来源文件名归一化：去扩展名、NFKC 全半角、去空白、剥「0X 第N册」册编号前缀。"""
    s = str(name).strip()
    s = re.sub(r'\.(pdf|md|docx?)$', '', s)
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'\s+', '', s)
    # 剥「0X / 02第二册」类册编号前缀（仅当后面紧跟「第N册」时）
    s = re.sub(r'^0?\d{1,2}(?=第[一二三四五六七八九十]+册)', '', s)
    return s


# 等价文件映射（测试集写法前缀 -> 语料源文件名前缀），按需补充：
# FILE_ALIASES = [("测试集写法前缀", "语料文件名前缀"), ...]
FILE_ALIASES = []


def _file_match(ref: str, src: str) -> bool:
    """文件匹配：精确/双向包含 → ref 以源文件名为前缀（「整书-章节」写法）→ 显式别名映射。

    刻意不用模糊子串比例兜底：「附件1-2…第2册」与「附件1-3…第3册」同构文件名
    会被模糊匹配误判为同一文件；差异无法用字符串消除的仅 05 册一种，写入别名表。
    """
    r, s = _norm_file_ref(ref), _norm_file_ref(src)
    if not r or not s:
        return False
    if r == s or r in s or s in r or r.startswith(s):
        return True
    for ref_pfx, src_pfx in FILE_ALIASES:
        if r.startswith(ref_pfx) and s.startswith(src_pfx):
            return True
    return False


def calc_file_recall(ref_file: str, docs: list[dict], top_k: int) -> tuple[int, int]:
    """文件召回率：Top-K 中是否有来自参考文件的 chunk。返回 (召回标记, 命中位置)。"""
    if not ref_file or not docs:
        return 0, 0
    for i, d in enumerate(docs[:top_k]):
        src = str(d.get("source", "") or "").strip()
        if _file_match(ref_file, src):
            return 1, i + 1
    return 0, 0