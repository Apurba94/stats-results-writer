"""APA-style sentences, effect size wording, assumption checks and dataset handling."""
import csv
import io
import math

from . import stats as st


def num(v, d=2):
    return "%.*f" % (d, v)


def no_zero(v, d=2):
    """APA drops the leading zero for values that cannot exceed 1."""
    s = "%.*f" % (d, abs(v))
    s = s[1:] if s.startswith("0") else s
    return ("-" if v < 0 and float(s or 0) != 0 else "") + s


def p_text(p):
    return "p < .001" if p < 0.001 else "p = %s" % no_zero(p, 3)


def size_d(d):
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def size_r(r):
    r = abs(r)
    return "negligible" if r < 0.1 else "small" if r < 0.3 else "medium" if r < 0.5 else "large"


def size_eta(e):
    return "negligible" if e < 0.01 else "small" if e < 0.06 else "medium" if e < 0.14 else "large"


def sig(p, alpha=0.05):
    return "statistically significant" if p < alpha else "not statistically significant"


def df_text(df):
    return "%d" % df if abs(df - round(df)) < 1e-9 else "%.2f" % df


def apa_independent(r, a="Group 1", b="Group 2", outcome="the outcome"):
    return ("An independent-samples %st test compared %s between %s (M = %s, SD = %s, n = %d) and %s (M = %s, SD = %s, n = %d). The difference was %s, "
            "t(%s) = %s, %s, d = %s, 95%% CI [%s, %s], a %s effect." % (
                "Welch " if r["welch"] else "", outcome, a, num(r["ma"]), num(r["sa"]), r["na"], b, num(r["mb"]), num(r["sb"]), r["nb"], sig(r["p"]), df_text(r["df"]), num(r["t"]),
                p_text(r["p"]), num(r["d"]), num(r["ci"][0]), num(r["ci"][1]), size_d(r["d"])))


def apa_paired(r, a="Condition 1", b="Condition 2"):
    return ("A paired-samples t test compared %s (M = %s, SD = %s) with %s (M = %s, SD = %s). The mean difference of %s was %s, t(%d) = %s, %s, d = %s, 95%% CI [%s, %s], a %s effect." % (
        a, num(r["ma"]), num(r["sa"]), b, num(r["mb"]), num(r["sb"]), num(r["diff"]), sig(r["p"]), r["df"], num(r["t"]), p_text(r["p"]), num(r["d"]), num(r["ci"][0]), num(r["ci"][1]), size_d(r["d"])))


def apa_one(r, name="The variable"):
    return "%s (M = %s, SD = %s) %s from %s, t(%d) = %s, %s, d = %s." % (
        name, num(r["mean"]), num(r["sd"]), "differed significantly" if r["p"] < 0.05 else "did not differ significantly", ak_trim(r["mu"]), r["df"], num(r["t"]), p_text(r["p"]), num(r["d"]))


def ak_trim(v):
    return ("%g" % v)


def apa_anova(r, factor="group", outcome="the outcome"):
    return "A one-way ANOVA showed that the effect of %s on %s was %s, F(%d, %d) = %s, %s, eta squared = %s, a %s effect." % (
        factor, outcome, sig(r["p"]), r["df1"], r["df2"], num(r["F"]), p_text(r["p"]), no_zero(r["eta2"]), size_eta(r["eta2"]))


def apa_chi(r, a="the two variables"):
    return "A chi-square test of independence for %s was %s, chi-square(%d, N = %d) = %s, %s, Cramer's V = %s." % (a, sig(r["p"]), r["df"], r["n"], num(r["chi2"]), p_text(r["p"]), no_zero(r["v"]))


def apa_pearson(r, a="the two variables", b=None):
    what = "%s and %s" % (a, b) if b else a
    kind = "positive" if r["r"] > 0 else "negative"
    return "There was %s %s correlation between %s, r(%d) = %s, %s, 95%% CI [%s, %s]." % (
        "a %s" % size_r(r["r"]) if size_r(r["r"]) != "large" else "a strong", kind, what, r["df"], no_zero(r["r"]), p_text(r["p"]), no_zero(r["ci"][0]), no_zero(r["ci"][1]))


def apa_spearman(r, a, b):
    return "The Spearman rank correlation between %s and %s was %s, rs(%d) = %s, %s." % (a, b, sig(r["p"]).replace("statistically ", ""), r["df"], no_zero(r["rho"]), p_text(r["p"]))


def apa_mw(r, a, b):
    return "A Mann-Whitney U test showed that the difference between %s (n = %d) and %s (n = %d) was %s, U = %s, z = %s, %s, r = %s." % (
        a, r["n1"], b, r["n2"], sig(r["p"]), ("%.1f" % r["U"]).rstrip("0").rstrip("."), num(r["z"]), p_text(r["p"]), no_zero(r["r"]))


def apa_regression(r, outcome="the outcome"):
    head = "A linear regression of %s explained %s of the variance, R squared = %s, adjusted R squared = %s, F(%d, %d) = %s, %s." % (
        outcome, "%.1f%%" % (100 * r["r2"]), no_zero(r["r2"]), no_zero(r["adj"]), r["df1"], r["df2"], num(r["F"]), p_text(r["p"]))
    rows = ["%s predicted the outcome: b = %s, SE = %s, t(%d) = %s, %s." % (c["name"], num(c["b"]), num(c["se"]), r["df2"], num(c["t"]), p_text(c["p"]))
            for c in r["coefs"][1:] if c["p"] < 0.05]
    return " ".join([head] + rows)


def notes_two_groups(a, b):
    out = []
    for name, g in (("first", a), ("second", b)):
        if len(g) >= 8:
            _, p = st.jarque_bera(g)
            if p < 0.05:
                out.append("The %s group looks non-normal. With small groups, consider a Mann-Whitney test." % name)
    if min(len(a), len(b)) < 2:
        return out
    _, _, _, p = st.levene([a, b])
    if p < 0.05:
        out.append("Variances differ (Brown-Forsythe %s), so the Welch version is the right choice." % p_text(p))
    if min(len(a), len(b)) < 20:
        out.append("Small samples: treat this as tentative, and check for outliers.")
    return out


def notes_anova(groups):
    out = []
    _, _, _, p = st.levene(groups)
    if p < 0.05:
        out.append("Variances differ across groups (Brown-Forsythe %s). Interpret F with care." % p_text(p))
    if len({len(g) for g in groups}) > 1:
        out.append("Group sizes are unequal.")
    return out


def notes_chi(r):
    return ["%d of %d cells have an expected count under 5, so the chi-square p value is unreliable. Consider Fisher's exact test." % (r["small"], r["cells"])] if r["small"] else []


class Dataset:
    def __init__(self, text):
        self.columns, self.rows = parse_table(text)

    def col(self, name):
        i = self.columns.index(name)
        return [r[i] if i < len(r) else "" for r in self.rows]

    def numeric(self, name):
        out = []
        for v in self.col(name):
            try:
                out.append(float(v))
            except ValueError:
                pass
        return out

    def is_numeric(self, name):
        vals = [v for v in self.col(name) if v != ""]
        return bool(vals) and len(self.numeric(name)) == len(vals)

    def pairs(self, names):
        """Rows where all the named columns are numeric (listwise deletion)."""
        idx = [self.columns.index(n) for n in names]
        out = []
        for r in self.rows:
            try:
                out.append([float(r[i]) for i in idx])
            except (ValueError, IndexError):
                pass
        return [list(c) for c in zip(*out)] if out else [[] for _ in names]

    def groups(self, value, factor):
        """{level: numeric values of `value`} for a grouping column."""
        i, j = self.columns.index(value), self.columns.index(factor)
        out = {}
        for r in self.rows:
            if j < len(r) and r[j] != "" and i < len(r):
                try:
                    out.setdefault(r[j], []).append(float(r[i]))
                except ValueError:
                    pass
        return out

    def crosstab(self, a, b):
        i, j = self.columns.index(a), self.columns.index(b)
        rows = sorted({r[i] for r in self.rows if i < len(r) and r[i] != ""})
        cols = sorted({r[j] for r in self.rows if j < len(r) and r[j] != ""})
        t = [[0] * len(cols) for _ in rows]
        for r in self.rows:
            if i < len(r) and j < len(r) and r[i] != "" and r[j] != "":
                t[rows.index(r[i])][cols.index(r[j])] += 1
        return rows, cols, t


def parse_table(text):
    first = text.split("\n", 1)[0]
    delim = "\t" if "\t" in first else ";" if first.count(";") > first.count(",") else ","
    rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text.strip()), delimiter=delim) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("the data need a header row and at least one row of values")
    head = rows[0]
    if len(set(head)) != len(head) or any(not h for h in head):
        raise ValueError("every column needs a unique name")
    return head, rows[1:]


SAMPLE_CSV = """id,group,sleep,recall,mood,passed
1,control,6.1,12,5,no
2,control,5.8,11,4,no
3,control,7.0,14,6,yes
4,control,6.4,13,5,no
5,control,5.5,10,3,no
6,control,6.8,15,6,yes
7,control,6.0,12,5,no
8,control,7.2,14,6,yes
9,treatment,7.9,16,7,yes
10,treatment,8.1,17,7,yes
11,treatment,7.4,14,6,yes
12,treatment,8.3,18,8,yes
13,treatment,7.7,15,6,no
14,treatment,8.0,17,7,yes
15,treatment,7.1,13,5,no
16,treatment,8.4,19,8,yes
"""
GUIDE = [("Compare two independent groups, numeric outcome", "Independent t test", "Mann-Whitney if skewed and small"),
         ("Compare the same people twice", "Paired t test", "Check the differences for outliers"),
         ("Compare one group with a known value", "One-sample t test", ""),
         ("Compare three or more groups", "One-way ANOVA", "Then pairwise comparisons"),
         ("Two categorical variables", "Chi-square test", "Needs expected counts of 5 or more"),
         ("Association of two numeric variables", "Pearson correlation", "Spearman if ordinal or non-linear"),
         ("Predict a numeric outcome", "Linear regression", "Check residuals")]
_ = math
