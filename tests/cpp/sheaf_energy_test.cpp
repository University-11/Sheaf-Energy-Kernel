// Build:
//   g++ -O3 -std=c++17 -Wall -Wextra -I Seaf-Enegry-Kernel/include Seaf-Enegry-Kernel/tests/sheaf_energy_test.cpp -o sheaf_energy_test
//
// Verifies the Sheaf Energy Kernel: graph construction, the coordinate sweep
// is energy-monotone, the free phase converges to a fixed point, the nudged
// phase pulls the class node toward the target, and EquiProp training reduces
// classification energy on a tiny synthetic task. Also verifies the
// auditability surface (top-violated constraints).

#include "sheaf_energy/sheaf_energy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace gm = sheaf_energy;

namespace {

int g_failed = 0;

void check(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failed;
    }
}

void check_close(double got, double want, double tol, const char* msg) {
    if (std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s  got=%.12g want=%.12g (|d|=%.3g, tol=%.3g)\n",
                     msg, got, want, std::abs(got - want), tol);
        ++g_failed;
    }
}

// ---------- Fixtures ----------

// Build a 4-node chain with a class node attached. Stalk dim r, num_classes <= r.
// vertices: 0 - 1 - 2 - 3 (class node)
gm::SheafGraph make_chain(std::size_t r, std::size_t num_classes,
                          double r_init = 1.0) {
    gm::SheafGraph g;
    g.r = r;
    g.num_vertices = 4;
    g.class_node = 3;
    g.num_classes = num_classes;
    g.lambda_l1 = 0.0;
    g.Wx.assign(g.num_vertices * r, 0.0);
    g.C_diag.assign(r, 0.0);
    for (std::size_t k = 0; k < num_classes; ++k) g.C_diag[k] = 1.0;

    auto add_edge = [&](std::size_t u, std::size_t v) {
        g.edge_u.push_back(u);
        g.edge_v.push_back(v);
        for (std::size_t k = 0; k < r; ++k) {
            g.R_u.push_back(r_init);
            g.R_v.push_back(r_init);
        }
    };
    add_edge(0, 1);
    add_edge(1, 2);
    add_edge(2, 3);
    g.build_adjacency();
    return g;
}

// ---------- Tests ----------

void test_soft_threshold() {
    check_close(gm::soft_threshold( 0.3, 0.5), 0.0, 0.0, "soft inside zero");
    check_close(gm::soft_threshold( 1.0, 0.5), 0.5, 0.0, "soft above shrinks");
    check_close(gm::soft_threshold(-1.0, 0.5), -0.5, 0.0, "soft below shrinks");
}

void test_adjacency_csr() {
    auto g = make_chain(/*r=*/3, /*num_classes=*/2);
    // Vertex degrees in a chain 0-1-2-3: 1, 2, 2, 1.
    check((g.ix_start[1] - g.ix_start[0]) == 1, "deg(0) = 1");
    check((g.ix_start[2] - g.ix_start[1]) == 2, "deg(1) = 2");
    check((g.ix_start[3] - g.ix_start[2]) == 2, "deg(2) = 2");
    check((g.ix_start[4] - g.ix_start[3]) == 1, "deg(3) = 1");
    // Sweep totals: every edge appears exactly twice in the incidence list.
    check(g.incident_edge.size() == 2 * g.edge_u.size(),
          "incidence list lists every edge from both ends");
}

void test_free_phase_zero_input_zero_state() {
    // No unary signal, no class target -> equilibrium is identically zero.
    auto g = make_chain(2, 2);
    std::vector<double> h(g.num_vertices * g.r, 0.13);  // arbitrary init
    gm::run_to_equilibrium(g, h, /*target=*/nullptr, /*beta=*/0.0, 32);
    for (double v : h) check_close(v, 0.0, 1e-12, "free phase converges to 0");
}

void test_free_phase_constant_propagation() {
    // Pin vertex 0's unary to a constant; with R = 1 everywhere, the
    // equilibrium should propagate that constant along the chain (with mild
    // attenuation from the unary anchors at hidden nodes Wx=0).
    auto g = make_chain(1, 1);
    g.Wx[0 * 1 + 0] = 1.0;  // h_0 anchored toward 1
    std::vector<double> h(g.num_vertices * g.r, 0.0);
    gm::run_to_equilibrium(g, h, nullptr, 0.0, 64);
    // Each hidden node's anchor pulls toward 0 and edge transports pull
    // toward neighbors. So the chain decays monotonically.
    check(h[0] > h[1] && h[1] > h[2] && h[2] > h[3], "chain decays from anchor");
    check(h[0] > 0.5, "anchored node holds most of its mass");
    check(h[3] > 0.0, "signal still reaches the class node");
}

void test_energy_monotone_under_sweep() {
    // A coordinate sweep against E^0 must never increase the energy.
    auto g = make_chain(2, 2);
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (auto& w : g.Wx) w = U(rng);
    for (auto& r : g.R_u) r = 0.5 + 0.5 * U(rng);  // keep R bounded away from 0
    for (auto& r : g.R_v) r = 0.5 + 0.5 * U(rng);
    std::vector<double> h(g.num_vertices * g.r);
    for (auto& v : h) v = U(rng);

    auto E_prev = gm::energy(g, h, nullptr, 0.0).total;
    for (int t = 0; t < 30; ++t) {
        gm::sheaf_sweep(g, h, nullptr, 0.0);
        const auto E = gm::energy(g, h, nullptr, 0.0).total;
        check(E <= E_prev + 1e-10, "sweep is monotone");
        E_prev = E;
    }
}

void test_nudged_phase_pulls_class_toward_target() {
    auto g = make_chain(2, 2);
    g.Wx[0] = 0.5;
    std::vector<double> h_free(g.num_vertices * g.r, 0.0);
    gm::run_to_equilibrium(g, h_free, nullptr, 0.0, 64);
    const double class0_free = h_free[g.class_node * g.r + 0];

    std::vector<double> h_nudge = h_free;
    double target[2] = {1.0, 0.0};   // push class 0 up
    const double beta = 0.5;
    gm::run_to_equilibrium(g, h_nudge, target, beta, 64);
    const double class0_nudge = h_nudge[g.class_node * g.r + 0];
    check(class0_nudge > class0_free, "nudge raises target class slot");
}

void test_equiprop_R_gradient_matches_finite_difference() {
    auto g = make_chain(/*r=*/2, /*num_classes=*/2, /*r_init=*/0.8);
    for (std::size_t i = 0; i < g.Wx.size(); ++i) {
        g.Wx[i] = 0.1 * static_cast<double>(i + 1);
    }

    std::vector<double> h_free = {
        0.20, -0.10,
        0.05,  0.30,
       -0.15,  0.25,
        0.10, -0.20,
    };
    std::vector<double> h_nudge = {
        0.25, -0.08,
        0.02,  0.35,
       -0.18,  0.21,
        0.14, -0.16,
    };
    const double beta = 0.4;

    std::vector<double> grad_u, grad_v;
    gm::equiprop_gradient_R(g, h_free, h_nudge, beta, grad_u, grad_v);

    const std::size_t off = 1;  // edge 0, coord 1, R_u
    const double eps = 1e-6;
    auto contrastive = [&](const gm::SheafGraph& graph) {
        const double nudged = gm::energy(graph, h_nudge, nullptr, 0.0).total;
        const double free = gm::energy(graph, h_free, nullptr, 0.0).total;
        return (nudged - free) / beta;
    };

    auto gp = g;
    auto gm_graph = g;
    gp.R_u[off] += eps;
    gm_graph.R_u[off] -= eps;
    const double finite_diff = (contrastive(gp) - contrastive(gm_graph)) / (2.0 * eps);

    check_close(grad_u[off], finite_diff, 1e-6,
                "EquiProp R_u gradient matches finite difference");

    const double old = g.R_u[off];
    gm::equiprop_update_R(g, h_free, h_nudge, /*eta=*/0.05, beta);
    const double implied_grad = (old - g.R_u[off]) / 0.05;
    check_close(implied_grad, grad_u[off], 1e-12,
                "EquiProp update applies gradient descent sign");
}

void test_beta_zero_update_is_safe_noop() {
    auto g = make_chain(1, 1, 0.6);
    std::vector<double> h0(g.num_vertices * g.r, 0.1);
    std::vector<double> h1(g.num_vertices * g.r, 0.2);
    const std::vector<double> before = g.R_u;
    gm::equiprop_update_R(g, h0, h1, /*eta=*/1.0, /*beta=*/0.0);
    check(g.R_u == before, "beta=0 update is a no-op");
}

void test_equiprop_reduces_loss_on_toy_task() {
    // Two examples, two classes, stalk r=2, chain depth 4.
    // Example A: input = +1 at vertex 0, target class 0.
    // Example B: input = -1 at vertex 0, target class 1.
    // With learnable diagonal R's, EquiProp should learn restriction maps
    // that route the input sign onto the right class slot.
    const std::size_t r = 2;
    auto g = make_chain(r, /*num_classes=*/2, /*r_init=*/0.7);

    auto run_step = [&](double x0, const double* tgt) {
        std::fill(g.Wx.begin(), g.Wx.end(), 0.0);
        g.Wx[0 * r + 0] = x0;     // input lives in slot 0 of vertex 0
        g.Wx[0 * r + 1] = -x0;    // and antipodally in slot 1, for asymmetry
        std::vector<double> h_free(g.num_vertices * r, 0.0);
        std::vector<double> h_nudge(g.num_vertices * r, 0.0);
        return gm::sheaf_train_step(g, h_free, h_nudge, tgt,
                                    /*beta=*/0.3,
                                    /*free_sweeps=*/24,
                                    /*nudged_sweeps=*/24,
                                    /*eta=*/0.05);
    };

    auto loss_at = [&](double x0, const double* tgt) {
        std::fill(g.Wx.begin(), g.Wx.end(), 0.0);
        g.Wx[0 * r + 0] = x0;
        g.Wx[0 * r + 1] = -x0;
        std::vector<double> h(g.num_vertices * r, 0.0);
        gm::run_to_equilibrium(g, h, nullptr, 0.0, 32);
        const double* hc = h.data() + g.class_node * r;
        double s = 0.0;
        for (std::size_t k = 0; k < r; ++k) {
            const double d = g.C_diag[k] * hc[k] - tgt[k];
            s += d * d;
        }
        return s;
    };

    double tgt_a[2] = {1.0, 0.0};
    double tgt_b[2] = {0.0, 1.0};
    const double loss_before =
        loss_at(+1.0, tgt_a) + loss_at(-1.0, tgt_b);

    for (int epoch = 0; epoch < 80; ++epoch) {
        run_step(+1.0, tgt_a);
        run_step(-1.0, tgt_b);
    }
    const double loss_after =
        loss_at(+1.0, tgt_a) + loss_at(-1.0, tgt_b);

    check(loss_after < loss_before,
          "EquiProp reduces task loss on toy 2-class problem");
    std::printf("  toy task loss: before=%.4f, after=%.4f\n",
                loss_before, loss_after);
}

void test_auditability_top_violated() {
    // Plant a deliberate disagreement on edge (1,2) by mismatching unary
    // anchors at the two endpoints. The most violated constraint must be a
    // vertex unary or the offending edge.
    auto g = make_chain(1, 1);
    g.Wx[0] = 0.0;
    g.Wx[1] = +5.0;   // pulls h_1 toward +5
    g.Wx[2] = -5.0;   // pulls h_2 toward -5
    g.Wx[3] = 0.0;
    std::vector<double> h(g.num_vertices, 0.0);
    gm::run_to_equilibrium(g, h, nullptr, 0.0, 64);

    auto eb = gm::energy(g, h, nullptr, 0.0);
    auto top = gm::top_violated(eb, /*k=*/3);
    check(!top.empty(), "auditor returns at least one constraint");
    bool conflict_edge_or_unary = false;
    for (const auto& c : top) {
        if (!c.is_vertex && (c.index == 1 /* edge (1,2) */)) {
            conflict_edge_or_unary = true; break;
        }
        if (c.is_vertex && (c.index == 1 || c.index == 2)) {
            conflict_edge_or_unary = true; break;
        }
    }
    check(conflict_edge_or_unary,
          "the conflict region surfaces in the top violated constraints");
}

void test_predict_returns_logits() {
    auto g = make_chain(2, 2);
    g.Wx[0] = 0.7;
    std::vector<double> h(g.num_vertices * g.r, 0.0);
    double logits[2] = {0.0, 0.0};
    gm::sheaf_predict(g, h, /*free_sweeps=*/64, logits);
    // C is diagonal identity on the first num_classes slots, so logits = h_cls.
    check_close(logits[0], h[g.class_node * g.r + 0], 1e-12,
                "logit_0 = C_0 * h_cls[0]");
    check_close(logits[1], h[g.class_node * g.r + 1], 1e-12,
                "logit_1 = C_1 * h_cls[1]");
}

void bench_sweep() {
    // Mid-size sparse graph: NxN grid with degree-4 lattice + class node.
    const std::size_t side = 16;
    const std::size_t r = 4;
    gm::SheafGraph g;
    g.r = r;
    g.num_vertices = side * side + 1;
    g.class_node = side * side;
    g.num_classes = r;
    g.lambda_l1 = 0.0;
    g.Wx.assign(g.num_vertices * r, 0.1);
    g.C_diag.assign(r, 1.0);

    auto add_edge = [&](std::size_t u, std::size_t v) {
        g.edge_u.push_back(u); g.edge_v.push_back(v);
        for (std::size_t k = 0; k < r; ++k) {
            g.R_u.push_back(0.6); g.R_v.push_back(0.6);
        }
    };
    for (std::size_t i = 0; i < side; ++i) {
        for (std::size_t j = 0; j < side; ++j) {
            const std::size_t v = i * side + j;
            if (j + 1 < side) add_edge(v, v + 1);
            if (i + 1 < side) add_edge(v, v + side);
            if (j == 0) add_edge(v, g.class_node);
        }
    }
    g.build_adjacency();
    std::vector<double> h(g.num_vertices * r, 0.0);

    const int reps = 4000;
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < reps; ++t) gm::sheaf_sweep(g, h, nullptr, 0.0);
    auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
    std::printf("sheaf_sweep: |V|=%zu, |E|=%zu, r=%zu  ->  %.0f ns/sweep, %.2f ns/(E*r)\n",
                g.num_vertices, g.edge_u.size(), r,
                ns, ns / static_cast<double>(g.edge_u.size() * r));
}

}  // namespace

int main() {
    test_soft_threshold();
    test_adjacency_csr();
    test_free_phase_zero_input_zero_state();
    test_free_phase_constant_propagation();
    test_energy_monotone_under_sweep();
    test_nudged_phase_pulls_class_toward_target();
    test_equiprop_R_gradient_matches_finite_difference();
    test_beta_zero_update_is_safe_noop();
    test_equiprop_reduces_loss_on_toy_task();
    test_auditability_top_violated();
    test_predict_returns_logits();

    if (g_failed != 0) {
        std::fprintf(stderr, "%d test(s) failed.\n", g_failed);
        return EXIT_FAILURE;
    }
    std::printf("All %s tests passed.\n", "sheaf_energy");
    bench_sweep();
    return EXIT_SUCCESS;
}
