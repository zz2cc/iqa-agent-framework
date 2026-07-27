# -*- coding: utf-8 -*-
"""经验库（规则库）：ADD / KEEP / DELETE / MODIFY，tag 定向注入。

移植自 MemRefine experience/library.py 的设计（该项目即为本项目作者开发），
简化点：去重用 token Jaccard（免嵌入模型依赖），规则质量分 = 门控结果。
容量 ≤10；每次调用向单个 Skill 注入同 tag 的 top ≤5 条。
"""
import json
import re

RULE_RE = re.compile(r"^\[([A-Z]+)\]\s*(.+)$")


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


class ExperienceLibrary:
    MAX_SIZE = 10
    INJECT_PER_SKILL = 5
    DEDUP_THRESHOLD = 0.7

    def __init__(self):
        self.rules: list[str] = []       # 纯文本（不含 tag）
        self.tags: list[str] = []        # TECH/AESTH/CONTENT/NATURAL/GLOBAL/GENERAL
        self.scores: list[float] = []    # 门控质量分（+1 过门，每存活一轮 +0.2）
        self.rounds_alive: list[int] = []

    # ---------- 操作 ----------
    def add(self, tagged_rule: str) -> bool:
        m = RULE_RE.match(tagged_rule.strip())
        if not m:
            return False
        tag, text = m.group(1), m.group(2).strip()
        tok = _tokens(text)
        for existing in self.rules:
            if _jaccard(tok, _tokens(existing)) > self.DEDUP_THRESHOLD:
                return False  # 近似重复，拒收
        if len(self.rules) >= self.MAX_SIZE:
            self._evict_weakest()
        self.rules.append(text)
        self.tags.append(tag)
        self.scores.append(1.0)
        self.rounds_alive.append(0)
        return True

    def keep_survivors(self):
        for i in range(len(self.scores)):
            self.scores[i] += 0.2
            self.rounds_alive[i] += 1

    def delete(self, indices: list[int]):
        for i in sorted(set(indices), reverse=True):
            if 0 <= i < len(self.rules):
                del self.rules[i], self.tags[i], self.scores[i], self.rounds_alive[i]

    def modify_merge(self, tagged_rule: str):
        """MODIFY：若与现有规则近似，则替换质量分更低的那条；否则按 ADD 处理。"""
        m = RULE_RE.match(tagged_rule.strip())
        if not m:
            return
        tag, text = m.group(1), m.group(2).strip()
        tok = _tokens(text)
        best_i, best_sim = -1, 0.0
        for i, existing in enumerate(self.rules):
            sim = _jaccard(tok, _tokens(existing))
            if sim > best_sim:
                best_i, best_sim = i, sim
        if best_sim > 0.5 and best_i >= 0:
            self.rules[best_i] = text
            self.tags[best_i] = tag
            self.scores[best_i] = max(self.scores[best_i], 1.0)
        else:
            self.add(tagged_rule)

    def _evict_weakest(self):
        i = min(range(len(self.scores)), key=lambda k: self.scores[k])
        self.delete([i])

    # ---------- 注入 ----------
    def for_skill(self, skill_tag: str) -> list[str]:
        """取某 Skill 应注入的规则（同 tag + GENERAL，按质量分排序取 top N）。"""
        idx = [i for i, t in enumerate(self.tags) if t == skill_tag or t == "GENERAL"]
        idx.sort(key=lambda i: self.scores[i], reverse=True)
        return [self.rules[i] for i in idx[: self.INJECT_PER_SKILL]]

    def as_dict_by_skill(self) -> dict[str, list[str]]:
        """键为 Skill ID（S-TECH 等），供 pipeline/cke 按 Skill 注入。"""
        tag2skill = {"TECH": "S-TECH", "AESTH": "S-AESTH", "CONTENT": "S-CONTENT",
                     "NATURAL": "S-NATURAL", "GLOBAL": "S-GLOBAL"}
        return {tag2skill[tag]: self.for_skill(tag)
                for tag in ["TECH", "AESTH", "CONTENT", "NATURAL", "GLOBAL"]}

    # ---------- 序列化 ----------
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rules": self.rules, "tags": self.tags,
                       "scores": self.scores, "rounds_alive": self.rounds_alive}, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "ExperienceLibrary":
        lib = cls()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        lib.rules, lib.tags, lib.scores, lib.rounds_alive = d["rules"], d["tags"], d["scores"], d["rounds_alive"]
        return lib

    def __len__(self):
        return len(self.rules)

    def __repr__(self):
        return f"ExperienceLibrary({len(self)} rules: " + ", ".join(
            f"[{t}] {r[:40]}..." for t, r in zip(self.tags, self.rules)) + ")"
