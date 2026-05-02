// Sheaf Energy Kernel, C++17.
//
// Prediction is an equilibrium of a sparse constraint system. This header uses
// diagonal restriction maps so every sweep is O(|E| * r) and every parameter
// update is explicit. No raw ownership is used; all memory is managed by STL
// containers.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sheaf_energy {

inline double soft_threshold(double z, double lam) noexcept {
    if (lam <= 0.0) return z;
    if (z >  lam) return z - lam;
    if (z < -lam) return z + lam;
    return 0.0;
}

struct SheafGraph {
    std::size_t r = 0;
    std::size_t num_vertices = 0;
    std::size_t class_node = 0;
    std::size_t num_classes = 0;

    std::vector<double> Wx;      // [num_vertices * r]
    double lambda_l1 = 0.0;

    std::vector<std::size_t> edge_u;
    std::vector<std::size_t> edge_v;
    std::vector<double> R_u;     // [num_edges * r], diagonal R_{e,u}
    std::vector<double> R_v;     // [num_edges * r], diagonal R_{e,v}

    std::vector<std::size_t> ix_start;
    std::vector<std::size_t> incident_edge;
    std::vector<std::size_t> incident_other;
    std::vector<std::uint8_t> incident_side;  // 0: self is u, 1: self is v

    // Diagonal class head: logits[k] = C_diag[k] * h_class[k].
    // The nudged energy uses beta * ||C h_class - target||^2.
    std::vector<double> C_diag;

    std::size_t num_edges() const noexcept { return edge_u.size(); }

    bool edge_valid(std::size_t e) const noexcept {
        return e < edge_u.size() &&
               e < edge_v.size() &&
               edge_u[e] < num_vertices &&
               edge_v[e] < num_vertices;
    }

    bool edge_storage_valid() const noexcept {
        const std::size_t need = edge_u.size() * r;
        return R_u.size() >= need && R_v.size() >= need;
    }

    bool node_storage_valid() const noexcept {
        return r > 0 &&
               num_vertices > 0 &&
               Wx.size() >= num_vertices * r &&
               C_diag.size() >= r &&
               class_node < num_vertices;
    }

    void build_adjacency() {
        ix_start.assign(num_vertices + 1, 0);
        incident_edge.clear();
        incident_other.clear();
        incident_side.clear();
        if (num_vertices == 0) return;

        std::vector<std::size_t> deg(num_vertices, 0);
        for (std::size_t e = 0; e < edge_u.size(); ++e) {
            if (!edge_valid(e)) continue;
            ++deg[edge_u[e]];
            ++deg[edge_v[e]];
        }
        for (std::size_t v = 0; v < num_vertices; ++v) {
            ix_start[v + 1] = ix_start[v] + deg[v];
        }

        const std::size_t total = ix_start[num_vertices];
        incident_edge.assign(total, 0);
        incident_other.assign(total, 0);
        incident_side.assign(total, 0);

        std::vector<std::size_t> cursor = ix_start;
        for (std::size_t e = 0; e < edge_u.size(); ++e) {
            if (!edge_valid(e)) continue;
            const std::size_t u = edge_u[e];
            const std::size_t v = edge_v[e];

            const std::size_t pu = cursor[u]++;
            incident_edge[pu] = e;
            incident_other[pu] = v;
            incident_side[pu] = 0;

            const std::size_t pv = cursor[v]++;
            incident_edge[pv] = e;
            incident_other[pv] = u;
            incident_side[pv] = 1;
        }
    }
};

inline bool graph_ready_for_sweep(const SheafGraph& g,
                                  const std::vector<double>& h) noexcept {
    return g.node_storage_valid() &&
           g.edge_storage_valid() &&
           g.ix_start.size() >= g.num_vertices + 1 &&
           h.size() >= g.num_vertices * g.r;
}

// One Gauss-Seidel coordinate sweep. With diagonal restrictions, each vertex
// update exactly minimizes the current quadratic-plus-L1 objective while
// holding its neighbors fixed:
//   A h^2 - 2 b h + lambda |h|.
inline void sheaf_sweep(const SheafGraph& g,
                        std::vector<double>& h,
                        const double* target,
                        double beta) noexcept {
    if (!graph_ready_for_sweep(g, h)) return;

    const std::size_t r = g.r;
    const double lambda = std::max(0.0, g.lambda_l1);
    const bool use_task = target && beta > 0.0 && std::isfinite(beta);

    for (std::size_t v = 0; v < g.num_vertices; ++v) {
        const std::size_t ix0 = g.ix_start[v];
        const std::size_t ix1 = g.ix_start[v + 1];
        double* hv = h.data() + v * r;
        const double* wx = g.Wx.data() + v * r;

        for (std::size_t k = 0; k < r; ++k) {
            double A = 1.0;
            double b = wx[k];

            for (std::size_t i = ix0; i < ix1; ++i) {
                const std::size_t e = g.incident_edge[i];
                if (!g.edge_valid(e)) continue;
                const std::size_t other = g.incident_other[i];
                const std::uint8_t side = g.incident_side[i];
                const double r_self = (side == 0 ? g.R_u[e * r + k]
                                                 : g.R_v[e * r + k]);
                const double r_other = (side == 0 ? g.R_v[e * r + k]
                                                  : g.R_u[e * r + k]);
                A += r_self * r_self;
                b += r_self * r_other * h[other * r + k];
            }

            if (use_task && v == g.class_node) {
                const double c = g.C_diag[k];
                A += beta * c * c;
                b += beta * c * target[k];
            }

            const double threshold = 0.5 * lambda / A;
            hv[k] = soft_threshold(b / A, threshold);
        }
    }
}

inline void run_to_equilibrium(const SheafGraph& g,
                               std::vector<double>& h,
                               const double* target,
                               double beta,
                               int sweeps) noexcept {
    if (sweeps <= 0) return;
    for (int t = 0; t < sweeps; ++t) sheaf_sweep(g, h, target, beta);
}

struct EnergyBreakdown {
    std::vector<double> per_vertex;
    std::vector<double> per_edge;
    double task_term = 0.0;
    double total = 0.0;
};

inline EnergyBreakdown energy(const SheafGraph& g,
                              const std::vector<double>& h,
                              const double* target,
                              double beta) {
    EnergyBreakdown out;
    out.per_vertex.assign(g.num_vertices, 0.0);
    out.per_edge.assign(g.edge_u.size(), 0.0);
    if (!graph_ready_for_sweep(g, h)) return out;

    const std::size_t r = g.r;
    const double lambda = std::max(0.0, g.lambda_l1);

    for (std::size_t v = 0; v < g.num_vertices; ++v) {
        const double* hv = h.data() + v * r;
        const double* wx = g.Wx.data() + v * r;
        double s = 0.0;
        for (std::size_t k = 0; k < r; ++k) {
            const double d = wx[k] - hv[k];
            s += d * d + lambda * std::abs(hv[k]);
        }
        out.per_vertex[v] = s;
        out.total += s;
    }

    for (std::size_t e = 0; e < g.edge_u.size(); ++e) {
        if (!g.edge_valid(e)) continue;
        const std::size_t u = g.edge_u[e];
        const std::size_t v = g.edge_v[e];
        const double* hu = h.data() + u * r;
        const double* hv = h.data() + v * r;
        double s = 0.0;
        for (std::size_t k = 0; k < r; ++k) {
            const double d = g.R_u[e * r + k] * hu[k] -
                             g.R_v[e * r + k] * hv[k];
            s += d * d;
        }
        out.per_edge[e] = s;
        out.total += s;
    }

    if (target && beta > 0.0 && std::isfinite(beta)) {
        const double* hc = h.data() + g.class_node * r;
        double s = 0.0;
        for (std::size_t k = 0; k < r; ++k) {
            const double d = g.C_diag[k] * hc[k] - target[k];
            s += d * d;
        }
        out.task_term = beta * s;
        out.total += out.task_term;
    }

    return out;
}

struct ViolatedConstraint {
    double energy;
    std::size_t index;
    bool is_vertex;
};

inline std::vector<ViolatedConstraint>
top_violated(const EnergyBreakdown& eb, std::size_t k) {
    std::vector<ViolatedConstraint> all;
    all.reserve(eb.per_vertex.size() + eb.per_edge.size());
    for (std::size_t v = 0; v < eb.per_vertex.size(); ++v) {
        all.push_back({eb.per_vertex[v], v, true});
    }
    for (std::size_t e = 0; e < eb.per_edge.size(); ++e) {
        all.push_back({eb.per_edge[e], e, false});
    }
    k = std::min(k, all.size());
    std::partial_sort(all.begin(), all.begin() + k, all.end(),
                      [](const ViolatedConstraint& a, const ViolatedConstraint& b) {
                          return a.energy > b.energy;
                      });
    all.resize(k);
    return all;
}

inline void edge_restriction_grad(const SheafGraph& g,
                                  const std::vector<double>& h,
                                  std::size_t e,
                                  std::size_t k,
                                  double& grad_u,
                                  double& grad_v) noexcept {
    grad_u = 0.0;
    grad_v = 0.0;
    if (!g.edge_valid(e) || !g.edge_storage_valid() ||
        h.size() < g.num_vertices * g.r) {
        return;
    }

    const std::size_t r = g.r;
    const std::size_t off = e * r + k;
    const std::size_t u = g.edge_u[e];
    const std::size_t v = g.edge_v[e];
    const double hu = h[u * r + k];
    const double hv = h[v * r + k];
    const double residual = g.R_u[off] * hu - g.R_v[off] * hv;
    grad_u =  2.0 * hu * residual;
    grad_v = -2.0 * hv * residual;
}

// Equilibrium propagation gradient estimator:
//   grad J(theta) ~= (grad_theta E(h_beta) - grad_theta E(h_0)) / beta.
inline void equiprop_gradient_R(const SheafGraph& g,
                                const std::vector<double>& h_free,
                                const std::vector<double>& h_nudge,
                                double beta,
                                std::vector<double>& grad_R_u,
                                std::vector<double>& grad_R_v) {
    const std::size_t total = g.edge_u.size() * g.r;
    grad_R_u.assign(total, 0.0);
    grad_R_v.assign(total, 0.0);
    if (!(beta > 0.0) || !std::isfinite(beta) || !g.edge_storage_valid()) return;

    const std::size_t r = g.r;
    const double inv_beta = 1.0 / beta;
    for (std::size_t e = 0; e < g.edge_u.size(); ++e) {
        if (!g.edge_valid(e)) continue;
        for (std::size_t k = 0; k < r; ++k) {
            double gu0 = 0.0, gv0 = 0.0, gu1 = 0.0, gv1 = 0.0;
            edge_restriction_grad(g, h_free, e, k, gu0, gv0);
            edge_restriction_grad(g, h_nudge, e, k, gu1, gv1);
            const std::size_t off = e * r + k;
            grad_R_u[off] = (gu1 - gu0) * inv_beta;
            grad_R_v[off] = (gv1 - gv0) * inv_beta;
        }
    }
}

inline void equiprop_update_R(SheafGraph& g,
                              const std::vector<double>& h_free,
                              const std::vector<double>& h_nudge,
                              double eta,
                              double beta) noexcept {
    if (!(beta > 0.0) || !std::isfinite(beta) || !std::isfinite(eta) ||
        !g.edge_storage_valid()) {
        return;
    }

    const std::size_t r = g.r;
    const double scale = eta / beta;
    for (std::size_t e = 0; e < g.edge_u.size(); ++e) {
        if (!g.edge_valid(e)) continue;
        for (std::size_t k = 0; k < r; ++k) {
            double gu0 = 0.0, gv0 = 0.0, gu1 = 0.0, gv1 = 0.0;
            edge_restriction_grad(g, h_free, e, k, gu0, gv0);
            edge_restriction_grad(g, h_nudge, e, k, gu1, gv1);
            const std::size_t off = e * r + k;
            g.R_u[off] -= scale * (gu1 - gu0);
            g.R_v[off] -= scale * (gv1 - gv0);
        }
    }
}

inline EnergyBreakdown sheaf_train_step(SheafGraph& g,
                                        std::vector<double>& h_free,
                                        std::vector<double>& h_nudge,
                                        const double* target,
                                        double beta,
                                        int free_sweeps,
                                        int nudged_sweeps,
                                        double eta) {
    run_to_equilibrium(g, h_free, target, 0.0, free_sweeps);
    h_nudge = h_free;
    run_to_equilibrium(g, h_nudge, target, beta, nudged_sweeps);
    EnergyBreakdown audit = energy(g, h_free, target, 0.0);
    equiprop_update_R(g, h_free, h_nudge, eta, beta);
    return audit;
}

inline void sheaf_predict(const SheafGraph& g,
                          std::vector<double>& h,
                          int free_sweeps,
                          double* logits_out) noexcept {
    if (!logits_out) return;
    run_to_equilibrium(g, h, nullptr, 0.0, free_sweeps);
    if (!graph_ready_for_sweep(g, h)) {
        for (std::size_t k = 0; k < g.num_classes; ++k) logits_out[k] = 0.0;
        return;
    }

    const double* hc = h.data() + g.class_node * g.r;
    const std::size_t classes = std::min(g.num_classes, g.r);
    for (std::size_t k = 0; k < classes; ++k) {
        logits_out[k] = g.C_diag[k] * hc[k];
    }
}

}  // namespace sheaf_energy
