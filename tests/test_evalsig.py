import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evalsig.intervals import (
    binom_cdf,
    cluster_bootstrap_ci,
    mcnemar_exact,
    newcombe_diff_interval,
    two_proportion_test,
    wilson_interval,
    wilson_interval_roots,
)
from evalsig.variance import variance_components
from evalsig.power import n_for_ci_width, n_paired, n_two_proportions
from evalsig.sequential import LookSchedule, SequentialAB, SequentialProportion
from evalsig.decide import Run, compare, decide, holm, load_runs
from evalsig.cli import main


class TestWilson(unittest.TestCase):
    def test_two_derivations_agree(self):
        # closed form vs direct quadratic roots, an independent derivation
        for s, n in [(0, 1), (1, 1), (0, 50), (50, 50), (8, 10), (27, 60), (90, 100)]:
            for alpha in (0.05, 0.01):
                a = wilson_interval(s, n, alpha)
                b = wilson_interval_roots(s, n, alpha)
                self.assertAlmostEqual(a.low, b.low, places=10, msg=f"{s}/{n}")
                self.assertAlmostEqual(a.high, b.high, places=10, msg=f"{s}/{n}")

    def test_symmetric_case_hand_computed(self):
        # 50/100 at 95%: center exactly 0.5 by symmetry, half = 0.096170
        # (hand derivation: z*sqrt(pq/n + z^2/4n^2)/(1+z^2/n))
        z = 1.959964
        half = z * math.sqrt(0.25 / 100 + z * z / 40000) / (1 + z * z / 100)
        ci = wilson_interval(50, 100)
        self.assertAlmostEqual(ci.center, 0.5, places=12)
        self.assertAlmostEqual(ci.half, half, places=4)

    def test_boundaries(self):
        lo = wilson_interval(0, 50)
        self.assertEqual(lo.low, 0.0)
        self.assertAlmostEqual(lo.high, 0.0714, places=3)
        hi = wilson_interval(50, 50)
        self.assertEqual(hi.high, 1.0)
        self.assertGreater(hi.low, 0.9)

    def test_validation(self):
        with self.assertRaises(ValueError):
            wilson_interval(1, 0)
        with self.assertRaises(ValueError):
            wilson_interval(5, 3)


class TestNewcombeAndZ(unittest.TestCase):
    def test_equal_proportions_contain_zero(self):
        for p, n in [(0.5, 40), (0.1, 30), (0.9, 50)]:
            ci = newcombe_diff_interval(p, n, p, n)
            self.assertLessEqual(ci.low, 0.0)
            self.assertGreaterEqual(ci.high, 0.0)

    def test_stays_in_unit_range(self):
        ci = newcombe_diff_interval(1.0, 30, 0.0, 30)
        self.assertGreaterEqual(ci.low, -1.0)
        self.assertLessEqual(ci.high, 1.0)
        self.assertGreater(ci.high, 0.9)

    def test_identical_samples_no_signal(self):
        res = two_proportion_test(40, 80, 40, 80)
        self.assertEqual(res.z, 0.0)
        self.assertGreater(res.p_value, 0.99)

    def test_known_z_value(self):
        # 60/100 vs 40/100: pooled p=0.5, z = 0.2/sqrt(0.25*0.02) = 2.8284
        res = two_proportion_test(60, 100, 40, 100)
        self.assertAlmostEqual(res.z, 2.82843, places=4)
        self.assertLess(res.p_value, 0.01)


class TestBinom(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(binom_cdf(0, 10, 0.5), 1 / 1024, places=12)
        self.assertAlmostEqual(binom_cdf(1, 10, 0.5), 11 / 1024, places=12)
        self.assertEqual(binom_cdf(10, 10, 0.5), 1.0)

    def test_complement(self):
        for k in (0, 3, 7, 20):
            self.assertAlmostEqual(
                binom_cdf(k, 20, 0.3) + (1 - binom_cdf(k, 20, 0.3)), 1.0, places=12
            )

    def test_extreme_p(self):
        self.assertEqual(binom_cdf(5, 10, 0.0), 1.0)
        self.assertEqual(binom_cdf(4, 10, 1.0), 0.0)


class TestMcNemar(unittest.TestCase):
    def test_hand_computed(self):
        # b=1, c=9: two-sided exact p = 2*(C(10,0)+C(10,1))/2^10 = 22/1024
        mc = mcnemar_exact(1, 9)
        self.assertAlmostEqual(mc.p_value, 22 / 1024, places=12)
        self.assertAlmostEqual(mc.diff, -0.8, places=12)
        self.assertTrue(mc.significant)

    def test_balanced_discordance_no_signal(self):
        mc = mcnemar_exact(5, 5)
        self.assertGreater(mc.p_value, 0.9)
        self.assertAlmostEqual(mc.diff, 0.0)

    def test_degenerate(self):
        mc = mcnemar_exact(0, 0)
        self.assertEqual(mc.p_value, 1.0)


class TestBootstrap(unittest.TestCase):
    def test_deterministic(self):
        clusters = [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
        a = cluster_bootstrap_ci(clusters, seed=7)
        b = cluster_bootstrap_ci(clusters, seed=7)
        self.assertEqual(a, b)

    def test_degenerate_collapses(self):
        ci = cluster_bootstrap_ci([[0.5], [0.5], [0.5]])
        self.assertAlmostEqual(ci.low, 0.5, places=9)
        self.assertAlmostEqual(ci.high, 0.5, places=9)

    def test_brackets_the_statistic(self):
        clusters = [[0.0, 1.0, 1.0], [1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0]]
        ci = cluster_bootstrap_ci(clusters, seed=3)
        self.assertLess(ci.low, 0.5)
        self.assertGreater(ci.high, 0.5)

    def test_resamples_clusters_not_runs(self):
        # 2 tasks x 50 identical runs: run-level bootstrap would give a tiny CI,
        # cluster bootstrap must respect that only 2 independent units exist
        ci = cluster_bootstrap_ci([[1.0] * 50, [0.0] * 50], B=2000, seed=1)
        self.assertGreater(ci.width, 0.3)


class TestVariance(unittest.TestCase):
    def test_hand_computed(self):
        # groups [1,2],[3,4],[5,6]: MSB=8, MSW=0.5, n0=2
        # sigma^2_b = (8-0.5)/2 = 3.75, ICC = 3.75/4.25
        vc = variance_components({"g1": [1, 2], "g2": [3, 4], "g3": [5, 6]})
        self.assertAlmostEqual(vc.between, 3.75, places=10)
        self.assertAlmostEqual(vc.within, 0.5, places=10)
        self.assertAlmostEqual(vc.icc, 3.75 / 4.25, places=10)

    def test_single_group(self):
        vc = variance_components({"only": [1.0, 2.0, 3.0]})
        self.assertEqual(vc.between, 0.0)
        self.assertEqual(vc.icc, 0.0)

    def test_identical_groups(self):
        vc = variance_components({"a": [1.0, 1.0], "b": [1.0, 1.0]})
        self.assertEqual(vc.between, 0.0)


class TestPower(unittest.TestCase):
    def test_two_proportion_hand_computed(self):
        # 0.50 vs 0.55, alpha .05, power .8: n = 1565 per version
        plan = n_two_proportions(0.50, 0.55)
        self.assertEqual(plan.n_per_version, 1565)

    def test_paired_hand_computed(self):
        # discordance 0.3, delta 0.05: (z_a+z_b)^2 * 0.3 / 0.0025 = 942
        plan = n_paired(0.3, 0.05)
        self.assertEqual(plan.n_per_version, 942)

    def test_width_hand_computed(self):
        # width .05 at p=.5: z^2 * .25 / .025^2 = 1537
        plan = n_for_ci_width(0.05, 0.5)
        self.assertEqual(plan.n_per_version, 1537)

    def test_validation(self):
        with self.assertRaises(ValueError):
            n_two_proportions(0.5, 0.5)
        with self.assertRaises(ValueError):
            n_paired(0.0, 0.05)


class TestSequential(unittest.TestCase):
    def test_confirms_clear_difference(self):
        seq = SequentialProportion(reference=0.5)
        v = seq.update(90, 100)
        self.assertEqual(v, "CONFIRMED")
        self.assertEqual(seq.verdict, "CONFIRMED")

    def test_precision_exit(self):
        seq = SequentialProportion(reference=0.5, target_half_width=0.05)
        v = "CONTINUE"
        for n in [25, 50, 100, 200, 400]:
            v = seq.update(round(0.5 * n), n)
            if v != "CONTINUE":
                break
        self.assertEqual(v, "PRECISION-REACHED")
        self.assertLessEqual(seq.ci().half, 0.05)

    def test_bonferroni_alpha(self):
        seq = SequentialProportion(reference=0.5)
        seq.update(20, 25)
        self.assertAlmostEqual(
            seq.report()["test_alpha_per_look"], 0.05 / 10, places=12
        )

    def test_look_schedule(self):
        sched = LookSchedule(n0=25, max_looks=5)
        self.assertEqual(sched.ns(), [25, 50, 100, 200, 400])
        self.assertEqual(sched.look_for(24), -1)
        self.assertEqual(sched.look_for(25), 0)
        self.assertEqual(sched.look_for(99), 1)
        self.assertEqual(sched.look_for(400), 4)

    def test_monotonicity_guard(self):
        seq = SequentialProportion()
        seq.update(10, 20)
        with self.assertRaises(ValueError):
            seq.update(5, 10)

    def test_ab_confirms(self):
        ab = SequentialAB()
        v = ab.update(90, 100, 40, 100)
        self.assertEqual(v, "CONFIRMED")


class TestHolm(unittest.TestCase):
    def test_hand_computed(self):
        adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"][0], 0.03, places=12)
        self.assertAlmostEqual(adj["c"][0], 0.06, places=12)
        self.assertAlmostEqual(adj["b"][0], 0.06, places=12)
        self.assertTrue(adj["a"][1])
        self.assertFalse(adj["b"][1])
        self.assertFalse(adj["c"][1])

    def test_monotonic_enforcement(self):
        adj = holm({"x": 0.01, "y": 0.011})
        self.assertGreaterEqual(adj["y"][0], adj["x"][0])


class TestCompare(unittest.TestCase):
    @staticmethod
    def _runs(pattern, name):
        # pattern: list of (task, outcome)
        return [Run(t, o, {"version": name}) for t, o in pattern]

    def test_paired_strong_difference(self):
        a = self._runs([(f"T{i}", 1.0 if i < 30 else 0.0) for i in range(38)], "A")
        b = self._runs([(f"T{i}", 0.0 if i < 30 else 1.0) for i in range(38)], "B")
        comp = compare(a, b, "A", "B")
        self.assertTrue(comp.paired)
        self.assertEqual(comp.verdict, "A-BETTER")
        self.assertLess(comp.p_value, 0.001)

    def test_unpaired_detection(self):
        a = self._runs([(f"X{i}", 1.0 if i < 60 else 0.0) for i in range(100)], "A")
        b = self._runs([(f"Y{i}", 1.0 if i < 40 else 0.0) for i in range(100)], "B")
        comp = compare(a, b, "A", "B")
        self.assertFalse(comp.paired)
        self.assertEqual(comp.verdict, "A-BETTER")

    def test_inconclusive_small_n(self):
        a = self._runs([(f"T{i}", 1.0) for i in range(5)], "A")
        b = self._runs([(f"T{i}", 0.0) for i in range(5)], "B")
        comp = compare(a, b, "A", "B")
        # 5 tasks: McNemar p = 2/2^5 = 0.0625 > alpha -> not conclusive
        self.assertAlmostEqual(comp.p_value, 2 / 32, places=12)
        self.assertEqual(comp.verdict, "INCONCLUSIVE")

    def test_continuous_outcomes(self):
        a = self._runs([(f"T{i}", 0.8 + 0.01 * i) for i in range(40)], "A")
        b = self._runs([(f"T{i}", 0.2 + 0.01 * i) for i in range(40)], "B")
        comp = compare(a, b, "A", "B")
        self.assertEqual(comp.verdict, "A-BETTER")
        self.assertGreater(comp.ci.low, 0.3)


class TestDecide(unittest.TestCase):
    def _candidates(self):
        def runs(pattern, name):
            return [Run(f"T{i}", o, {"version": name}) for i, o in enumerate(pattern)]

        n = 60
        x = [1.0] * 27 + [0.0] * 33
        y = [1.0] * 9 + [0.0] * 51
        z = [1.0] * 25 + [0.0] * 35
        return {"X": runs(x, "X"), "Y": runs(y, "Y"), "Z": runs(z, "Z")}

    def test_elimination(self):
        rep = decide(self._candidates())
        actions = {name: action for name, _, action in rep.ranking}
        self.assertEqual(actions["X"], "KEEP (baseline)")
        self.assertEqual(actions["Y"], "ELIMINATE (worse)")
        self.assertEqual(actions["Z"], "KEEP (not separable)")
        self.assertLess(rep.candidates["adjusted"]["Y"], 0.001)

    def test_needs_two(self):
        with self.assertRaises(ValueError):
            decide({"only": [Run("T", 1.0, {})]})


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write(self, name, data):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def test_compare_cli_exit_codes(self):
        a = self._write("a.json", [
            {"task": f"T{i}", "success": 1 if i < 60 else 0} for i in range(100)
        ])
        b = self._write("b.json", [
            {"task": f"T{i}", "success": 1 if i < 40 else 0} for i in range(100)
        ])
        rc = main(["compare", a, b])
        self.assertEqual(rc, 0)
        c = self._write("c.json", [
            {"task": f"T{i}", "success": 1 if i < 57 else 0} for i in range(100)
        ])
        rc2 = main(["compare", a, c])
        self.assertEqual(rc2, 2)

    def test_report_cli(self):
        runs = [
            {"task": f"T{i % 40}", "success": 1 if i % 3 else 0, "seed": i % 4}
            for i in range(80)
        ]
        path = self._write("runs.json", runs)
        rc = main(["report", path, "--factor", "seed"])
        self.assertEqual(rc, 0)

    def test_decide_cli(self):
        data = {
            "X": [{"task": f"T{i}", "success": 1 if i < 27 else 0} for i in range(60)],
            "Y": [{"task": f"T{i}", "success": 1 if i < 9 else 0} for i in range(60)],
        }
        path = self._write("cand.json", data)
        rc = main(["decide", path])
        self.assertEqual(rc, 0)

    def test_load_runs_schema(self):
        path = self._write("r.json", [{"id": "T1", "score": 0.75, "engine": "v2"}])
        runs = load_runs(path)
        self.assertEqual(runs[0].task, "T1")
        self.assertEqual(runs[0].outcome, 0.75)
        self.assertEqual(runs[0].factors["engine"], "v2")


if __name__ == "__main__":
    unittest.main()
