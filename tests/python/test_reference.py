import unittest

from sheaf_energy import SheafGraph, energy, equiprop_gradient_R, equiprop_update_R, run_to_equilibrium


def make_chain() -> SheafGraph:
    graph = SheafGraph(
        r=2,
        num_vertices=3,
        class_node=2,
        num_classes=2,
        wx=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        c_diag=[1.0, 1.0],
    )
    graph.add_edge(0, 1, [0.8, 0.8], [0.8, 0.8])
    graph.add_edge(1, 2, [0.8, 0.8], [0.8, 0.8])
    graph.build_adjacency()
    return graph


class SheafReferenceTest(unittest.TestCase):
    def test_equilibrium_propagates_signal(self) -> None:
        graph = make_chain()
        h = [0.0 for _ in range(graph.num_vertices * graph.r)]
        run_to_equilibrium(graph, h, target=None, beta=0.0, sweeps=32)

        self.assertGreater(h[0], h[2])
        self.assertGreater(h[2], h[4])
        self.assertGreater(h[4], 0.0)

    def test_equiprop_gradient_matches_finite_difference(self) -> None:
        graph = make_chain()
        h_free = [0.2, -0.1, 0.05, 0.3, 0.1, -0.2]
        h_nudge = [0.25, -0.08, 0.02, 0.35, 0.14, -0.16]
        beta = 0.4
        grad_u, _ = equiprop_gradient_R(graph, h_free, h_nudge, beta)
        off = 1
        eps = 1e-6

        def contrast() -> float:
            return (energy(graph, h_nudge, None, 0.0).total - energy(graph, h_free, None, 0.0).total) / beta

        old = graph.r_u[off]
        graph.r_u[off] = old + eps
        plus = contrast()
        graph.r_u[off] = old - eps
        minus = contrast()
        graph.r_u[off] = old

        self.assertAlmostEqual(grad_u[off], (plus - minus) / (2.0 * eps), places=6)

    def test_equiprop_update_is_noop_for_beta_zero(self) -> None:
        graph = make_chain()
        before = list(graph.r_u)
        equiprop_update_R(graph, [0.0] * 6, [1.0] * 6, eta=1.0, beta=0.0)
        self.assertEqual(graph.r_u, before)


if __name__ == "__main__":
    unittest.main()

