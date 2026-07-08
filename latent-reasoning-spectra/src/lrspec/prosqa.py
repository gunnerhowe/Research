"""ProsQA records: DAG parsing, ground-truth path, branch labels, linear-chain nulls.

Each ProsQA record (facebookresearch/coconut data/prosqa_*.json) carries:
  question, answer, steps, idx_to_symbol, edges, root, target, neg_target
The reasoning DAG is `edges` over integer node ids; `steps` spell out the ground-truth
path root -> ... -> target as natural-language statements.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Problem:
    idx: int
    question: str
    answer: str
    steps: list[str]
    idx_to_symbol: list[str]
    edges: list[tuple[int, int]]
    root: int
    target: int
    neg_target: int
    # derived
    path: list[int] = field(default_factory=list)  # v_0=root .. v_L=target
    out_degree: dict[int, int] = field(default_factory=dict)

    @property
    def n_hops(self) -> int:
        return len(self.path) - 1

    def branch_label(self, t: int) -> int | None:
        """Branch label for thought step t (1-indexed): 1 iff out-degree(v_{t-1}) > 1.

        Returns None for steps beyond the ground-truth path (padding thoughts).
        """
        if t < 1 or t > self.n_hops:
            return None
        return int(self.out_degree.get(self.path[t - 1], 0) > 1)

    def branch_degree(self, t: int) -> int | None:
        """Graded label: out-degree of v_{t-1}."""
        if t < 1 or t > self.n_hops:
            return None
        return self.out_degree.get(self.path[t - 1], 0)

    @property
    def is_linear_chain(self) -> bool:
        """True iff no step on the ground-truth path had a real choice."""
        return all(
            self.out_degree.get(v, 0) <= 1 for v in self.path[:-1]
        )


_STEP_RE = re.compile(r"^(?:Every )?(\w+) is an? (\w+)\.$")


def _parse_step(step: str) -> tuple[str, str]:
    m = _STEP_RE.match(step.strip())
    if not m:
        raise ValueError(f"unparseable step: {step!r}")
    return m.group(1), m.group(2)


def load_problems(path) -> list[Problem]:
    raw = json.load(open(path, encoding="utf-8"))
    problems = []
    for i, r in enumerate(raw):
        p = Problem(
            idx=i,
            question=r["question"],
            answer=r["answer"],
            steps=r["steps"],
            idx_to_symbol=r["idx_to_symbol"],
            edges=[tuple(e) for e in r["edges"]],
            root=r["root"],
            target=r["target"],
            neg_target=r["neg_target"],
        )
        _derive(p)
        problems.append(p)
    return problems


def _derive(p: Problem) -> None:
    sym_to_idx = {s.lower(): i for i, s in enumerate(p.idx_to_symbol)}
    out: dict[int, int] = {}
    for a, b in p.edges:
        out[a] = out.get(a, 0) + 1
    p.out_degree = out

    # Reconstruct the ground-truth path from the steps.
    # Step 1: "<RootName> is a <concept>."  Steps 2..L: "Every <c1> is a <c2>."
    path = [p.root]
    for s in p.steps:
        subj, obj = _parse_step(s)
        obj_id = sym_to_idx[obj.lower()]
        path.append(obj_id)
    p.path = path

    # sanity: path must start at root, end at target, follow edges
    edge_set = {tuple(e) for e in p.edges}
    assert path[-1] == p.target, f"path end {path[-1]} != target {p.target} (idx {p.idx})"
    for a, b in zip(path[:-1], path[1:]):
        assert (a, b) in edge_set, f"path edge ({a},{b}) not in DAG (idx {p.idx})"


_SENT_SPLIT_RE = re.compile(r"(?<=[.?])\s+")


def prune_to_linear(p: Problem) -> Problem | None:
    """Paired pruned-real null (PLAN.md amendment 2026-07-08).

    Remove from the question exactly the off-path out-edges of ground-truth path nodes,
    so every path node has out-degree 1; keep all other statements, their order, and the
    final question intact.  Returns None if pruning would remove every mention of
    neg_target (instance skipped).
    """
    sents = _SENT_SPLIT_RE.split(p.question.strip())
    q_sent = sents[-1]
    stmts = sents[:-1]
    sym = p.idx_to_symbol

    # statements to drop: root's other memberships; path concept nodes' other out-edges
    drop: set[str] = set()
    for i, v in enumerate(p.path[:-1]):
        nxt = p.path[i + 1]
        if i == 0:
            # person statements: "<Root> is a <X>."
            for a, b in p.edges:
                if a == v and b != nxt:
                    drop.add(f"{sym[v]} is a {sym[b]}.")
        else:
            for a, b in p.edges:
                if a == v and b != nxt:
                    drop.add(f"Every {sym[v]} is a {sym[b]}.")

    kept = [s for s in stmts if s not in drop]
    # guard: neg_target must still be mentioned somewhere
    if not any(f" {sym[p.neg_target]}" in s for s in kept):
        return None

    new_edges = []
    path_set = {v: p.path[i + 1] for i, v in enumerate(p.path[:-1])}
    for a, b in p.edges:
        if a in path_set and b != path_set[a]:
            continue
        new_edges.append((a, b))

    q = " ".join(kept + [q_sent])
    pruned = Problem(
        idx=p.idx,
        question=q,
        answer=p.answer,
        steps=p.steps,
        idx_to_symbol=p.idx_to_symbol,
        edges=new_edges,
        root=p.root,
        target=p.target,
        neg_target=p.neg_target,
    )
    _derive(pruned)
    assert pruned.is_linear_chain, f"pruning failed for idx {p.idx}"
    return pruned


def branch_stats(problems: list[Problem]) -> dict:
    """Summary of branch structure across a problem set."""
    n_steps = 0
    n_branch = 0
    hops = []
    linear = 0
    for p in problems:
        hops.append(p.n_hops)
        if p.is_linear_chain:
            linear += 1
        for t in range(1, p.n_hops + 1):
            n_steps += 1
            n_branch += p.branch_label(t)
    return {
        "n_problems": len(problems),
        "n_path_steps": n_steps,
        "n_branch_steps": n_branch,
        "branch_rate": n_branch / max(n_steps, 1),
        "n_linear_chain": linear,
        "hops_min": min(hops),
        "hops_max": max(hops),
    }
