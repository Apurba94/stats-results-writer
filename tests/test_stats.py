import math
import os
import shutil
import tempfile
import tkinter as tk
import unittest
from unittest import mock

from stats_results_writer import app as A
from stats_results_writer import appkit as ak
from stats_results_writer import report as rp
from stats_results_writer import stats as st

# Student's sleep data and the PlantGrowth data, with results published in standard references
SLEEP1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
SLEEP2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
CTRL = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]
TRT1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
TRT2 = [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]


class TestDistributions(unittest.TestCase):
    def test_table_values(self):
        self.assertAlmostEqual(st.t_p(2.228139, 10), 0.05, places=5)
        self.assertAlmostEqual(st.t_p(1.959964, 100000), 0.05, places=3)
        self.assertAlmostEqual(st.t_p(12.7062, 1), 0.05, places=4)
        self.assertAlmostEqual(st.f_p(4.256495, 2, 9), 0.05, places=5)
        self.assertAlmostEqual(st.f_p(161.4476, 1, 1), 0.05, places=4)
        self.assertAlmostEqual(st.chi2_p(3.841459, 1), 0.05, places=6)
        self.assertAlmostEqual(st.chi2_p(5.991465, 2), 0.05, places=6)
        self.assertAlmostEqual(st.chi2_p(18.30704, 10), 0.05, places=6)
        self.assertAlmostEqual(st.normal_cdf(1.959964), 0.975, places=5)

    def test_critical_values(self):
        self.assertAlmostEqual(st.t_critical(10), 2.228139, places=5)
        self.assertAlmostEqual(st.t_critical(30), 2.042272, places=5)

    def test_identities(self):
        for t in (0.5, 1.5, 3):
            self.assertAlmostEqual(st.t_cdf(t, 7) + st.t_cdf(-t, 7), 1, places=10)
            self.assertAlmostEqual(st.f_p(t * t, 1, 7), st.t_p(t, 7), places=10)   # t squared is F(1, df)
        self.assertEqual(st.betai(2, 3, 0), 0)
        self.assertEqual(st.betai(2, 3, 1), 1)
        self.assertAlmostEqual(st.betai(2, 3, 0.4), 1 - st.betai(3, 2, 0.6), places=12)


class TestPublishedResults(unittest.TestCase):
    def test_welch_sleep(self):
        r = st.independent_t(SLEEP1, SLEEP2)
        self.assertAlmostEqual(r["t"], -1.8608, places=4)
        self.assertAlmostEqual(r["df"], 17.776, places=3)
        self.assertAlmostEqual(r["p"], 0.07939, places=5)
        self.assertAlmostEqual(r["ci"][0], -3.3654832, places=5)
        self.assertAlmostEqual(r["ci"][1], 0.2054832, places=5)

    def test_student_sleep(self):
        r = st.independent_t(SLEEP1, SLEEP2, welch=False)
        self.assertAlmostEqual(r["t"], -1.8608, places=4)
        self.assertEqual(r["df"], 18)
        self.assertAlmostEqual(r["p"], 0.07919, places=5)

    def test_paired_sleep(self):
        r = st.paired_t(SLEEP1, SLEEP2)
        self.assertAlmostEqual(r["t"], -4.0621, places=4)
        self.assertEqual(r["df"], 9)
        self.assertAlmostEqual(r["p"], 0.002833, places=6)

    def test_anova_plantgrowth(self):
        r = st.anova([CTRL, TRT1, TRT2])
        self.assertAlmostEqual(r["F"], 4.846, places=3)
        self.assertEqual((r["df1"], r["df2"]), (2, 27))
        self.assertAlmostEqual(r["p"], 0.01591, places=5)
        self.assertAlmostEqual(r["ssb"], 3.76634, places=4)
        self.assertAlmostEqual(r["ssw"], 10.49209, places=4)

    def test_anova_two_groups_equals_t_squared(self):
        t = st.independent_t(SLEEP1, SLEEP2, welch=False)
        f = st.anova([SLEEP1, SLEEP2])
        self.assertAlmostEqual(f["F"], t["t"] ** 2, places=8)
        self.assertAlmostEqual(f["p"], t["p"], places=8)

    def test_pearson(self):
        r = st.pearson([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
        self.assertAlmostEqual(r["r"], 0.7745967, places=6)
        t = r["r"] * math.sqrt(3 / (1 - r["r"] ** 2))
        self.assertAlmostEqual(t, 2.12132, places=4)                      # sqrt(4.5)
        self.assertAlmostEqual(r["p"], st.f_p(4.5, 1, 3), places=10)
        self.assertAlmostEqual(r["p"], 0.1240, places=3)

    def test_chi_square(self):
        r = st.chi_square([[10, 20], [20, 10]])
        self.assertAlmostEqual(r["chi2"], 6.666667, places=5)
        self.assertEqual(r["df"], 1)
        self.assertAlmostEqual(r["p"], 0.009823, places=5)
        self.assertAlmostEqual(r["v"], 1 / 3, places=6)
        y = st.chi_square([[10, 20], [20, 10]], correction=True)
        self.assertAlmostEqual(y["chi2"], 5.4, places=5)

    def test_regression_simple(self):
        r = st.regression([2, 4, 5, 4, 5], [[1, 2, 3, 4, 5]])
        self.assertAlmostEqual(r["coefs"][0]["b"], 2.2, places=8)
        self.assertAlmostEqual(r["coefs"][1]["b"], 0.6, places=8)
        self.assertAlmostEqual(r["r2"], 0.6, places=8)
        self.assertAlmostEqual(r["F"], 4.5, places=6)
        self.assertAlmostEqual(r["p"], st.pearson([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])["p"], places=8)

    def test_regression_recovers_coefficients(self):
        x1 = [1, 2, 3, 4, 5, 6, 7, 8]
        x2 = [2, 1, 4, 3, 6, 5, 8, 7]
        y = [3 + 2 * a - b for a, b in zip(x1, x2)]
        r = st.regression(y, [x1, x2], ["a", "b"])
        self.assertAlmostEqual(r["coefs"][1]["b"], 2, places=8)
        self.assertAlmostEqual(r["coefs"][2]["b"], -1, places=8)
        with self.assertRaises(ValueError):
            st.regression(y, [x1, x1])

    def test_nonparametric(self):
        self.assertEqual(st.ranks([10, 20, 20, 30]), [1, 2.5, 2.5, 4])
        self.assertEqual(st.mann_whitney([1, 2, 3], [4, 5, 6])["U"], 0)
        self.assertAlmostEqual(st.spearman([1, 2, 3, 4], [1, 4, 9, 16])["rho"], 1, places=9)
        r = st.mann_whitney(CTRL, TRT2)
        wins = sum((x > y) + 0.5 * (x == y) for x in CTRL for y in TRT2)      # U by direct counting
        self.assertEqual(r["U"], min(wins, 100 - wins))


class TestDescription(unittest.TestCase):
    def test_describe(self):
        d = st.describe([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertEqual(d["mean"], 5)
        self.assertAlmostEqual(d["sd"], 2.13809, places=4)
        self.assertEqual(d["median"], 4.5)
        self.assertEqual(st.describe([3])["mean"], 3)

    def test_guards(self):
        for fn in (lambda: st.one_sample_t([1]), lambda: st.one_sample_t([2, 2]), lambda: st.independent_t([1], [1, 2]), lambda: st.paired_t([1, 2], [1]),
                   lambda: st.anova([[1, 2]]), lambda: st.chi_square([[1, 2]]), lambda: st.chi_square([[0, 0], [1, 2]]), lambda: st.pearson([1, 2], [1, 2])):
            with self.assertRaises(ValueError):
                fn()

    def test_levene(self):
        self.assertLess(st.levene([[1, 2, 3, 4, 5, 6] * 3, [10, 40, 90, -60, 200, 5] * 3])[3], 0.05)


class TestWriting(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(rp.p_text(0.0004), "p < .001")
        self.assertEqual(rp.p_text(0.0123), "p = .012")
        self.assertEqual(rp.no_zero(0.456), ".46")
        self.assertEqual(rp.no_zero(-0.456), "-.46")
        self.assertEqual(rp.df_text(18.0), "18")
        self.assertEqual(rp.df_text(17.776), "17.78")

    def test_sizes(self):
        self.assertEqual([rp.size_d(v) for v in (0.1, 0.3, 0.6, 1.0)], ["negligible", "small", "medium", "large"])
        self.assertEqual(rp.size_r(-0.4), "medium")
        self.assertEqual(rp.size_eta(0.2), "large")

    def test_sentences(self):
        t = rp.apa_independent(st.independent_t(SLEEP1, SLEEP2), "Drug A", "Drug B", "extra sleep")
        for needle in ("Welch t test", "M = 0.75", "SD = 1.79", "not statistically significant", "t(17.78) = -1.86", "p = .079", "95% CI [-3.37, 0.21]"):
            self.assertIn(needle, t)
        self.assertIn("F(2, 27) = 4.85, p = .016", rp.apa_anova(st.anova([CTRL, TRT1, TRT2]), "treatment", "weight"))
        self.assertIn("r(3) = .77, p = .124", rp.apa_pearson(st.pearson([1, 2, 3, 4, 5], [2, 4, 5, 4, 5]), "a", "b"))
        self.assertIn("chi-square(1, N = 60) = 6.67", rp.apa_chi(st.chi_square([[10, 20], [20, 10]])))

    def test_notes(self):
        self.assertTrue(any("expected count" in n for n in rp.notes_chi(st.chi_square([[1, 2], [3, 1]]))))
        self.assertTrue(any("Small samples" in n for n in rp.notes_two_groups([1.0, 2, 3], [2.0, 3, 5])))


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.d = rp.Dataset(rp.SAMPLE_CSV)

    def test_columns(self):
        self.assertTrue(self.d.is_numeric("recall"))
        self.assertFalse(self.d.is_numeric("group"))
        self.assertEqual(len(self.d.groups("recall", "group")["control"]), 8)
        rows, cols, t = self.d.crosstab("group", "passed")
        self.assertEqual((rows, cols), (["control", "treatment"], ["no", "yes"]))
        self.assertEqual(t, [[5, 3], [2, 6]])

    def test_listwise_deletion(self):
        d = rp.Dataset("a,b\n1,2\nx,3\n4,\n5,6\n")
        self.assertEqual(d.pairs(["a", "b"]), [[1.0, 5.0], [2.0, 6.0]])

    def test_bad_tables(self):
        for bad in ("only,a,header\n", "a,a\n1,2\n", ",b\n1,2\n"):
            with self.assertRaises(ValueError):
                rp.Dataset(bad)
        self.assertEqual(rp.Dataset("a;b\n1;2\n").columns, ["a", "b"])
        self.assertEqual(rp.Dataset("a\tb\n1\t2\n").columns, ["a", "b"])

    def test_every_test_runs(self):
        for test, a, b, g in (("Independent t test", "recall", "", "group"), ("Mann-Whitney U", "recall", "", "group"), ("One-way ANOVA", "recall", "", "group"),
                              ("Paired t test", "recall", "mood", ""), ("One-sample t test", "recall", "", ""), ("Chi-square test", "group", "passed", ""),
                              ("Pearson correlation", "sleep", "recall", ""), ("Spearman correlation", "sleep", "recall", "")):
            text, rows, notes = A.run_test(self.d, test, a, b, g, 10)
            self.assertTrue(text, test)
        text, rows, _ = A.run_test(self.d, "Linear regression", "recall", "", "", predictors=["sleep", "mood"])
        self.assertEqual(rows[1][0], "sleep")
        with self.assertRaises(ValueError):
            A.run_test(self.d, "Independent t test", "recall", "", "")
        with self.assertRaises(ValueError):
            A.run_test(self.d, "Independent t test", "recall", "", "id")


class TestGui(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = ak.Store(A.SCHEMA, os.path.join(self.dir, "t.sqlite3"), A.APP.APP_NAME)
        A.sample(self.store)
        try:
            self.app = A.APP(self.store)
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.errors = []
        self.app.report_callback_exception = lambda e, v, t: self.errors.append(repr(v))
        self.app.update()

    def tearDown(self):
        self.store.close()
        self.app.destroy()
        shutil.rmtree(self.dir, ignore_errors=True)
        self.assertEqual(self.errors, [])

    def test_every_screen_in_both_themes(self):
        for theme in ("Dark", "Light"):
            self.app.set_theme(theme)
            for t in self.app.titles():
                self.app.navigate(t)
                self.app.update()

    def test_titles_are_unique(self):
        self.assertEqual(len(set(self.app.titles())), len(self.app.titles()))

    def test_analysis_and_saving(self):
        self.app.navigate("Analyse")
        a = self.app.views["Analyse"]
        self.assertIn("independent-samples", ak.get_text(a.text))
        a.test.set("Linear regression")
        self.assertFalse(a.run())
        self.assertIn("Cannot run", ak.get_text(a.text))
        a.pred.selection_set(0)
        self.assertTrue(a.run())
        self.assertIn("linear regression", ak.get_text(a.text))
        n = self.store.count("result")
        a.save()
        self.assertEqual(self.store.count("result"), n + 1)

    def test_data_loading(self):
        d = self.app.views["Data"]
        with mock.patch("tkinter.messagebox.showerror") as err:
            self.assertFalse(d.add("bad", "one line only"))
        self.assertTrue(err.called)
        self.assertTrue(d.add("mine", "x,y\n1,2\n3,5\n4,4\n"))
        self.assertEqual(d.name.get(), "mine")
        self.assertEqual(len(d.table.tree.get_children()), 2)


if __name__ == "__main__":
    unittest.main()
