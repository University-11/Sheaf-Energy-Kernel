"""Small pure-Python reference for sparse Sheaf Energy experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


def soft_threshold(z: float, lam: float) -> float:
    if lam <= 0.0:
        return z
    if z > lam:
        return z - lam
    if z < -lam:
        return z + lam
    return 0.0


@dataclass
class SheafGraph:
    r: int
    num_vertices: int
    class_node: int
    num_classes: int
    wx: list[float]
    c_diag: list[float]
    lambda_l1: float = 0.0
    edge_u: list[int] = field(default_factory=list)
    edge_v: list[int] = field(default_factory=list)
    r_u: list[float] = field(default_factory=list)
    r_v: list[float] = field(default_factory=list)
    adjacency: list[list[tuple[int, int, int]]] = field(default_factory=list)

    def add_edge(self, u: int, v: int, r_u_diag: list[float], r_v_diag: list[float]) -> None:
        if len(r_u_diag) != self.r or len(r_v_diag) != self.r:
            raise ValueError("restriction map diagonals must have length r")
        if not 0 <= u < self.num_vertices or not 0 <= v < self.num_vertices:
            raise ValueError("edge endpoint out of range")
        self.edge_u.append(u)
        self.edge_v.append(v)
        self.r_u.extend(float(x) for x in r_u_diag)
        self.r_v.extend(float(x) for x in r_v_diag)

    def build_adjacency(self) -> None:
        self.adjacency = [[] for _ in range(self.num_vertices)]
        for edge_id, (u, v) in enumerate(zip(self.edge_u, self.edge_v)):
            self.adjacency[u].append((edge_id, v, 0))
            self.adjacency[v].append((edge_id, u, 1))


@dataclass(frozen=True)
class EnergyBreakdown:
    per_vertex: list[float]
    per_edge: list[float]
    task_term: float
    total: float


def sheaf_sweep(graph: SheafGraph, h: list[float], target: list[float] | None, beta: float) -> None:
    if not graph.adjacency:
        graph.build_adjacency()
    r = graph.r
    use_task = target is not None and beta > 0.0

    for v in range(graph.num_vertices):
        for k in range(r):
            a = 1.0
            b = graph.wx[v * r + k]

            for edge_id, other, side in graph.adjacency[v]:
                off = edge_id * r + k
                r_self = graph.r_u[off] if side == 0 else graph.r_v[off]
                r_other = graph.r_v[off] if side == 0 else graph.r_u[off]
                a += r_self * r_self
                b += r_self * r_other * h[other * r + k]

            if use_task and v == graph.class_node:
                c = graph.c_diag[k]
                a += beta * c * c
                b += beta * c * target[k]

            h[v * r + k] = soft_threshold(b / a, 0.5 * max(0.0, graph.lambda_l1) / a)


def run_to_equilibrium(
    graph: SheafGraph,
    h: list[float],
    target: list[float] | None,
    beta: float,
    sweeps: int,
) -> None:
    for _ in range(max(0, sweeps)):
        sheaf_sweep(graph, h, target, beta)


def energy(
    graph: SheafGraph,
    h: list[float],
    target: list[float] | None,
    beta: float,
) -> EnergyBreakdown:
    r = graph.r
    per_vertex = [0.0 for _ in range(graph.num_vertices)]
    per_edge = [0.0 for _ in graph.edge_u]
    total = 0.0

    for v in range(graph.num_vertices):
        s = 0.0
        for k in range(r):
            diff = graph.wx[v * r + k] - h[v * r + k]
            s += diff * diff + max(0.0, graph.lambda_l1) * abs(h[v * r + k])
        per_vertex[v] = s
        total += s

    for edge_id, (u, v) in enumerate(zip(graph.edge_u, graph.edge_v)):
        s = 0.0
        for k in range(r):
            off = edge_id * r + k
            diff = graph.r_u[off] * h[u * r + k] - graph.r_v[off] * h[v * r + k]
            s += diff * diff
        per_edge[edge_id] = s
        total += s

    task_term = 0.0
    if target is not None and beta > 0.0:
        cls = graph.class_node
        for k in range(r):
            diff = graph.c_diag[k] * h[cls * r + k] - target[k]
            task_term += diff * diff
        task_term *= beta
        total += task_term

    return EnergyBreakdown(per_vertex=per_vertex, per_edge=per_edge, task_term=task_term, total=total)


def edge_restriction_grad(graph: SheafGraph, h: list[float], edge_id: int, k: int) -> tuple[float, float]:
    r = graph.r
    off = edge_id * r + k
    u = graph.edge_u[edge_id]
    v = graph.edge_v[edge_id]
    hu = h[u * r + k]
    hv = h[v * r + k]
    residual = graph.r_u[off] * hu - graph.r_v[off] * hv
    return 2.0 * hu * residual, -2.0 * hv * residual


def equiprop_gradient_R(
    graph: SheafGraph,
    h_free: list[float],
    h_nudge: list[float],
    beta: float,
) -> tuple[list[float], list[float]]:
    total = len(graph.r_u)
    grad_u = [0.0 for _ in range(total)]
    grad_v = [0.0 for _ in range(total)]
    if beta <= 0.0:
        return grad_u, grad_v

    for edge_id in range(len(graph.edge_u)):
        for k in range(graph.r):
            gu0, gv0 = edge_restriction_grad(graph, h_free, edge_id, k)
            gu1, gv1 = edge_restriction_grad(graph, h_nudge, edge_id, k)
            off = edge_id * graph.r + k
            grad_u[off] = (gu1 - gu0) / beta
            grad_v[off] = (gv1 - gv0) / beta
    return grad_u, grad_v


def equiprop_update_R(
    graph: SheafGraph,
    h_free: list[float],
    h_nudge: list[float],
    eta: float,
    beta: float,
) -> None:
    grad_u, grad_v = equiprop_gradient_R(graph, h_free, h_nudge, beta)
    for i, grad in enumerate(grad_u):
        graph.r_u[i] -= eta * grad
    for i, grad in enumerate(grad_v):
        graph.r_v[i] -= eta * grad

