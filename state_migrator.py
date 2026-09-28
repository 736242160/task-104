#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
state_migrator.py — 多对象状态组合合法性判定与迁移工具（纯 Python 标准库，单文件）

输入 DSL（行式文本，`#` 之后为注释，空行忽略）：

    object  <名称> : <状态1> <状态2> ... [init <初始状态>]   # 缺省 init 取第一个状态
    rule    legal|illegal : <对象>=<状态> [& <对象>=<状态> ...]
    migrate <对象> <目标状态>
    begin                                # 开启事务块
    commit                               # 提交事务块

组合语义（判定一个完整状态赋值是否合法）：
    * illegal 规则：其全部条件被当前组合满足时，该组合非法；
    * legal 规则：作为“例外/白名单”，优先级高于 illegal —— 若某组合同时命中
      legal 与 illegal 规则，则判定为合法（legal 覆盖 illegal）；
    * 未命中任何 illegal 规则的组合合法。

冲突检测：条件集完全相同而极性相反（legal vs illegal）的规则对，属于规则本身
矛盾，加载时报告；同极性同条件的规则视为重复，给出告警。

回滚范围（设计取舍）：单个 migrate 请求是“先校验、后生效”的原子操作，失败时
状态根本未变，无需回滚；因此回滚只作用于显式事务块 begin..commit —— 块内任一
请求失败，则块内已生效的全部迁移按序撤销（状态恢复、历史条目标记为 rolled-back
保留可追溯），整个事务记为失败。未提交的事务在输入结束时自动回滚。

用法：
    python3 state_migrator.py 输入文件
    python3 state_migrator.py --demo        # 运行内置演示
    cat 输入文件 | python3 state_migrator.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------- 数据模型

@dataclass(frozen=True)
class Rule:
    rid: int                    # 规则编号（按出现顺序）
    polarity: str               # 'legal' | 'illegal'
    conditions: tuple           # 排序后的 ((对象, 状态), ...)，作为规则的唯一键
    lineno: int

    def describe(self) -> str:
        body = " & ".join(f"{obj}={state}" for obj, state in self.conditions)
        return f"规则#{self.rid}[{self.polarity}](第{self.lineno}行): {body}"


@dataclass
class ObjDef:
    name: str
    states: list                # 合法状态集（有序）
    current: str                # 当前状态
    lineno: int


@dataclass
class HistoryEntry:
    seq: int
    obj: str
    from_state: str
    to_state: str
    lineno: int
    status: str = "applied"     # applied | rolled-back


@dataclass
class Problem:
    severity: str               # '错误' | '告警'
    stage: str                  # '解析' | '加载' | '迁移'
    lineno: int                 # 0 表示与具体行无关
    message: str

    def describe(self) -> str:
        where = f"第{self.lineno}行" if self.lineno else "全局"
        return f"[{self.severity}|{self.stage}|{where}] {self.message}"


# ---------------------------------------------------------------- 组合判定

def rule_matches(rule: Rule, config: dict) -> bool:
    """规则的全部条件都被组合 config 满足时，称规则被命中。"""
    return all(config.get(obj) == state for obj, state in rule.conditions)


def evaluate(config: dict, rules: list):
    """判定完整组合是否合法。

    返回 (是否合法, 被违反的 illegal 规则列表)。
    legal 规则是 illegal 的例外：同时命中时 legal 优先。
    """
    illegal_hits = [r for r in rules if r.polarity == "illegal" and rule_matches(r, config)]
    if not illegal_hits:
        return True, []
    legal_hits = [r for r in rules if r.polarity == "legal" and rule_matches(r, config)]
    if legal_hits:
        return True, []
    return False, illegal_hits


# ---------------------------------------------------------------- 引擎

class Engine:
    def __init__(self):
        self.objects = {}       # name -> ObjDef
        self.rules = []         # list[Rule]
        self.requests = []      # list[(kind, args, lineno)]，kind ∈ migrate/begin/commit
        self.problems = []      # list[Problem]
        self.history = []       # list[HistoryEntry]
        self.results = []       # 每个迁移请求的结果行（含 OK/FAIL）
        self.seq = 0

    # -- 问题记录 --
    def report(self, severity, stage, lineno, message):
        self.problems.append(Problem(severity, stage, lineno, message))

    # -- 解析 --
    def parse(self, text: str):
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            head = tokens[0]
            try:
                if head == "object":
                    self._parse_object(tokens, lineno)
                elif head == "rule":
                    self._parse_rule(tokens, lineno)
                elif head == "migrate":
                    if len(tokens) != 3:
                        raise ValueError("migrate 语法: migrate <对象> <目标状态>")
                    self.requests.append(("migrate", (tokens[1], tokens[2]), lineno))
                elif head == "begin":
                    self.requests.append(("begin", None, lineno))
                elif head == "commit":
                    self.requests.append(("commit", None, lineno))
                else:
                    raise ValueError(f"未知指令: {head!r}（可用: object/rule/migrate/begin/commit）")
            except ValueError as exc:
                self.report("错误", "解析", lineno, str(exc))

    def _parse_object(self, tokens, lineno):
        if len(tokens) < 4 or tokens[2] != ":":
            raise ValueError("object 语法: object <名称> : <状态...> [init <初始状态>]")
        name = tokens[1]
        body = tokens[3:]
        init = None
        if "init" in body:
            idx = body.index("init")
            if idx != len(body) - 2:
                raise ValueError("init 之后须且仅须跟一个状态名")
            init = body[-1]
            body = body[:idx]
        if not body:
            raise ValueError(f"对象 {name!r} 至少需要一个状态")
        if len(set(body)) != len(body):
            raise ValueError(f"对象 {name!r} 的状态集存在重复")
        if init is not None and init not in body:
            raise ValueError(f"对象 {name!r} 的初始状态 {init!r} 不在状态集中")
        if name in self.objects:
            raise ValueError(f"对象 {name!r} 重复定义")
        self.objects[name] = ObjDef(name, body, init if init else body[0], lineno)

    def _parse_rule(self, tokens, lineno):
        if len(tokens) < 4 or tokens[2] != ":" or tokens[1] not in ("legal", "illegal"):
            raise ValueError("rule 语法: rule legal|illegal : <对象>=<状态> [& ...]")
        conds = []
        for part in " ".join(tokens[3:]).split("&"):
            part = part.strip()
            if not part or "=" not in part:
                raise ValueError(f"非法条件: {part!r}（应为 对象=状态）")
            obj, _, state = part.partition("=")
            obj, state = obj.strip(), state.strip()
            if not obj or not state:
                raise ValueError(f"非法条件: {part!r}（应为 对象=状态）")
            conds.append((obj, state))
        key = tuple(sorted(set(conds)))
        self.rules.append(Rule(len(self.rules) + 1, tokens[1], key, lineno))

    # -- 加载期校验 --
    def validate(self):
        for rule in self.rules:
            for obj, state in rule.conditions:
                if obj not in self.objects:
                    self.report("错误", "加载", rule.lineno,
                                f"{rule.describe()} 引用了不存在的对象 {obj!r}")
                elif state not in self.objects[obj].states:
                    self.report("错误", "加载", rule.lineno,
                                f"{rule.describe()} 引用了对象 {obj!r} 不存在的状态 {state!r}")
        # 规则冲突 / 重复检测：以条件集为键分组
        by_cond = {}
        for rule in self.rules:
            by_cond.setdefault(rule.conditions, []).append(rule)
        for conds, group in by_cond.items():
            polarities = {r.polarity for r in group}
            if len(polarities) > 1:
                ids = "、".join(f"#{r.rid}(第{r.lineno}行,{r.polarity})" for r in group)
                self.report("错误", "加载", group[0].lineno,
                            f"规则矛盾：条件集相同但极性相反 —— {ids}；"
                            f"按 legal 优先处理，但请修正规则")
            elif len(group) > 1:
                ids = "、".join(f"#{r.rid}(第{r.lineno}行)" for r in group)
                self.report("告警", "加载", group[0].lineno,
                            f"重复规则（同极性同条件）：{ids}")
        # 初始组合合法性
        config = {n: o.current for n, o in self.objects.items()}
        ok, hits = evaluate(config, self.rules)
        if not ok:
            detail = "; ".join(r.describe() for r in hits)
            self.report("告警", "加载", 0, f"初始状态组合本身非法，违反: {detail}")

    # -- 迁移执行 --
    def current_config(self):
        return {n: o.current for n, o in self.objects.items()}

    def try_migrate(self, name, target, lineno):
        """校验并（合法时）立即应用一次迁移。返回 (ok, 描述)。"""
        if name not in self.objects:
            return False, f"对象不存在: {name!r}"
        obj = self.objects[name]
        if target not in obj.states:
            return False, (f"状态不存在: 对象 {name!r} 无状态 {target!r}"
                           f"（可选: {'/'.join(obj.states)}）")
        if obj.current == target:
            return True, f"{name} 已是 {target}，无变化"
        config = self.current_config()
        config[name] = target                      # 目标组合 = 新状态 + 其他对象当前状态
        ok, hits = evaluate(config, self.rules)
        if not ok:
            detail = "; ".join(r.describe() for r in hits)
            return False, f"目标组合非法: {name}->{target} 违反 {detail}"
        self.seq += 1
        self.history.append(HistoryEntry(self.seq, name, obj.current, target, lineno))
        obj.current = target
        return True, f"{name}: {self.history[-1].from_state} -> {target}"

    def run(self):
        in_txn = False
        snapshot = None           # (状态快照, 历史长度)
        for kind, args, lineno in self.requests:
            if kind == "begin":
                if in_txn:
                    self.report("错误", "迁移", lineno, "不支持嵌套事务，忽略内层 begin")
                    continue
                in_txn = True
                snapshot = (self.current_config(), len(self.history))
                self.results.append(f"第{lineno:>3}行  begin            -> 事务开始")
            elif kind == "commit":
                if not in_txn:
                    self.report("错误", "迁移", lineno, "commit 时没有活动事务")
                    self.results.append(f"第{lineno:>3}行  commit           -> FAIL 无活动事务")
                    continue
                in_txn = False
                snapshot = None
                self.results.append(f"第{lineno:>3}行  commit           -> 事务提交")
            else:  # migrate
                name, target = args
                ok, msg = self.try_migrate(name, target, lineno)
                tag = "OK  " if ok else "FAIL"
                self.results.append(f"第{lineno:>3}行  migrate {name} {target:<8} -> {tag} {msg}")
                if not ok:
                    self.report("错误", "迁移", lineno, f"migrate {name} {target}: {msg}")
                    if in_txn:
                        states, hlen = snapshot
                        for n, s in states.items():          # 恢复事务前状态
                            self.objects[n].current = s
                        for entry in self.history[hlen:]:    # 历史保留但标记回滚
                            entry.status = "rolled-back"
                        self.results.append(
                            f"           事务回滚: 撤销 {len(self.history) - hlen} 步已生效迁移")
                        self.report("错误", "迁移", lineno,
                                    "事务内迁移失败，已回滚至 begin 前状态")
                        in_txn = False
                        snapshot = None
        if in_txn:  # 输入结束仍有未提交事务：自动回滚
            states, hlen = snapshot
            for n, s in states.items():
                self.objects[n].current = s
            for entry in self.history[hlen:]:
                entry.status = "rolled-back"
            self.report("错误", "迁移", 0, "输入结束时事务未提交，已自动回滚")
            self.results.append("           事务未提交，自动回滚")

    # -- 输出报告 --
    def render(self) -> str:
        out = []
        out.append("========== 加载报告 ==========")
        for n, o in self.objects.items():
            out.append(f"对象 {n}: 状态集={{{', '.join(o.states)}}} 初始={o.current}")
        for r in self.rules:
            out.append(r.describe())
        load_problems = [p for p in self.problems if p.stage in ("解析", "加载")]
        out.append("加载问题: " + (f"{len(load_problems)} 条（见错误清单）" if load_problems else "无"))

        out.append("")
        out.append("========== 迁移结果 ==========")
        out.extend(self.results if self.results else ["（无迁移请求）"])

        out.append("")
        out.append("========== 错误清单 ==========")
        if self.problems:
            for i, p in enumerate(self.problems, 1):
                out.append(f"P{i}. {p.describe()}")
        else:
            out.append("无")

        out.append("")
        out.append("========== 最终状态 ==========")
        for n, o in self.objects.items():
            out.append(f"{n} = {o.current}")
        ok, hits = evaluate(self.current_config(), self.rules)
        out.append("最终组合合法性: " + ("合法" if ok else
                   "非法，违反 " + "; ".join(r.describe() for r in hits)))

        out.append("")
        out.append("========== 迁移历史 ==========")
        if self.history:
            for h in self.history:
                out.append(f"#{h.seq} 第{h.lineno}行  {h.obj}: {h.from_state} -> {h.to_state}  [{h.status}]")
        else:
            out.append("（无已生效迁移）")
        return "\n".join(out)


# ---------------------------------------------------------------- 演示与入口

DEMO = r"""
# ---------- 对象定义 ----------
object Light  : on off       init off
object Door   : open closed  init closed
object Heater : on off       init off

# ---------- 组合规则 ----------
rule illegal : Light=off & Door=open       # 规则#1: 灯灭时门不可开
rule illegal : Heater=on & Door=closed     # 规则#2
rule legal   : Heater=on & Door=closed     # 规则#3: 与#2 条件相同极性相反 -> 规则矛盾

# ---------- 迁移请求流 ----------
migrate Door  open        # FAIL: Light=off，违反规则#1
migrate Light on          # OK
migrate Door  open        # OK: 前次迁移使组合变为合法（连续迁移）
migrate Light off         # FAIL: 此时 Door=open，违反规则#1
migrate Window open       # FAIL: 对象不存在
migrate Door  shut        # FAIL: 状态不存在
begin
migrate Heater on         # 事务内暂时 OK（规则#2/#3 矛盾，legal 优先）
migrate Light off         # FAIL -> 整个事务回滚，Heater 恢复 off
commit                    # FAIL: 事务已因失败回滚
migrate Door  closed      # OK
migrate Light off         # OK: 门已关，组合合法
"""


def main(argv):
    if len(argv) > 1 and argv[1] == "--demo":
        text = DEMO
    elif len(argv) > 1:
        with open(argv[1], "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()

    eng = Engine()
    eng.parse(text)
    eng.validate()
    eng.run()
    print(eng.render())
    return 1 if any(p.severity == "错误" for p in eng.problems) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
