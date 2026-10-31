"""Statistics from first principles, using only the standard library. Distribution functions are computed with
continued fractions for the incomplete beta and gamma functions."""
import math


def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h


def betai(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def gammainc(a, x):
    """Regularised lower incomplete gamma P(a, x)."""
    if x <= 0:
        return 0.0
    if x < a + 1:
        term = total = 1 / a
        n = a
        for _ in range(1000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > tiny else tiny)
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1 - h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def normal_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * betai(df / 2, 0.5, x)
    return 1 - tail if t > 0 else tail


def t_p(t, df):
    """Two-sided p value."""
    return betai(df / 2, 0.5, df / (df + t * t))


def f_p(f, d1, d2):
    """Upper-tail p value."""
    if f <= 0:
        return 1.0
    return betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * f))


def chi2_p(x, df):
    return 1 - gammainc(df / 2, x / 2) if x > 0 else 1.0


def t_critical(df, conf=0.95):
    target = (1 + conf) / 2
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------- description ----------------
def mean(x):
    return sum(x) / len(x)


def variance(x):
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def sd(x):
    return math.sqrt(variance(x))


def quantile(x, q):
    s = sorted(x)
    i = (len(s) - 1) * q
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def describe(x):
    n = len(x)
    out = {"n": n, "mean": mean(x), "min": min(x), "max": max(x), "median": quantile(x, 0.5), "q1": quantile(x, 0.25), "q3": quantile(x, 0.75)}
    if n > 1:
        out["sd"] = sd(x)
        out["se"] = out["sd"] / math.sqrt(n)
        h = t_critical(n - 1) * out["se"]
        out["ci"] = (out["mean"] - h, out["mean"] + h)
    if n > 2 and out.get("sd"):
        m, s = out["mean"], out["sd"]
        out["skew"] = n / ((n - 1) * (n - 2)) * sum(((v - m) / s) ** 3 for v in x)
    return out


def jarque_bera(x):
    """A rough normality check: (statistic, p). Weak in small samples."""
    n = len(x)
    m = mean(x)
    s2 = sum((v - m) ** 2 for v in x) / n
    if s2 == 0:
        return 0.0, 1.0
    s = sum((v - m) ** 3 for v in x) / n / s2 ** 1.5
    k = sum((v - m) ** 4 for v in x) / n / s2 ** 2
    jb = n / 6 * (s * s + (k - 3) ** 2 / 4)
    return jb, chi2_p(jb, 2)


def levene(groups):
    """Brown-Forsythe version: (F, df1, df2, p)."""
    z = [[abs(v - quantile(g, 0.5)) for v in g] for g in groups]
    r = anova(z)
    return r["F"], r["df1"], r["df2"], r["p"]


# ---------------- tests ----------------
def one_sample_t(x, mu=0.0):
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 values")
    s = sd(x)
    if s == 0:
        raise ValueError("all values are identical")
    se = s / math.sqrt(n)
    t = (mean(x) - mu) / se
    df = n - 1
    h = t_critical(df) * se
    return {"t": t, "df": df, "p": t_p(t, df), "d": (mean(x) - mu) / s, "mean": mean(x), "sd": s, "n": n, "mu": mu, "ci": (mean(x) - mu - h, mean(x) - mu + h)}


def independent_t(a, b, welch=True):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("each group needs at least 2 values")
    va, vb = variance(a), variance(b)
    if va == 0 and vb == 0:
        raise ValueError("both groups have no variation")
    diff = mean(a) - mean(b)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if welch:
        se = math.sqrt(va / na + vb / nb)
        df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    else:
        se = math.sqrt(pooled * (1 / na + 1 / nb))
        df = na + nb - 2
    t = diff / se
    d = diff / math.sqrt(pooled)
    g = d * (1 - 3 / (4 * (na + nb) - 9))
    h = t_critical(df) * se
    return {"t": t, "df": df, "p": t_p(t, df), "d": d, "g": g, "diff": diff, "ci": (diff - h, diff + h), "welch": welch, "ma": mean(a), "sa": math.sqrt(va), "na": na,
            "mb": mean(b), "sb": math.sqrt(vb), "nb": nb}


def paired_t(a, b):
    if len(a) != len(b):
        raise ValueError("paired data need equal lengths")
    d = [x - y for x, y in zip(a, b)]
    r = one_sample_t(d)
    r.update({"diff": mean(d), "ma": mean(a), "mb": mean(b), "sa": sd(a), "sb": sd(b)})
    return r


def anova(groups):
    """One-way between-subjects ANOVA."""
    groups = [g for g in groups if g]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        raise ValueError("need at least two groups and more values than groups")
    grand = sum(sum(g) for g in groups) / n
    ssb = sum(len(g) * (mean(g) - grand) ** 2 for g in groups)
    ssw = sum(sum((v - mean(g)) ** 2 for v in g) for g in groups)
    df1, df2 = k - 1, n - k
    if ssw == 0:
        raise ValueError("no variation within the groups")
    msb, msw = ssb / df1, ssw / df2
    f = msb / msw
    return {"F": f, "df1": df1, "df2": df2, "p": f_p(f, df1, df2), "eta2": ssb / (ssb + ssw), "omega2": max(0.0, (ssb - df1 * msw) / (ssb + ssw + msw)), "ssb": ssb, "ssw": ssw,
            "means": [mean(g) for g in groups], "sds": [sd(g) if len(g) > 1 else 0.0 for g in groups], "ns": [len(g) for g in groups], "msw": msw}


def pairwise_bonferroni(names, groups):
    """Welch t tests for all pairs, with Bonferroni-adjusted p values."""
    pairs = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    out = []
    for i, j in pairs:
        r = independent_t(groups[i], groups[j])
        out.append((names[i], names[j], r["diff"], r["t"], min(1.0, r["p"] * len(pairs))))
    return out


def chi_square(table, correction=False):
    """Test of independence on a table of counts."""
    rows, cols = len(table), len(table[0])
    if rows < 2 or cols < 2:
        raise ValueError("the table needs at least 2 rows and 2 columns")
    rt = [sum(r) for r in table]
    ct = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    n = sum(rt)
    if min(rt) == 0 or min(ct) == 0:
        raise ValueError("a row or column total is zero")
    x = 0.0
    small = 0
    for i in range(rows):
        for j in range(cols):
            e = rt[i] * ct[j] / n
            small += e < 5
            d = abs(table[i][j] - e)
            if correction and rows == 2 and cols == 2:
                d = max(0.0, d - 0.5)
            x += d * d / e
    df = (rows - 1) * (cols - 1)
    return {"chi2": x, "df": df, "p": chi2_p(x, df), "n": n, "v": math.sqrt(x / (n * (min(rows, cols) - 1))), "small": small, "cells": rows * cols}


def pearson(x, y):
    n = len(x)
    if n != len(y) or n < 3:
        raise ValueError("need at least 3 pairs")
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        raise ValueError("a variable has no variation")
    r = max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))
    df = n - 2
    p = 0.0 if abs(r) >= 1 else t_p(r * math.sqrt(df / (1 - r * r)), df)
    if n > 3 and abs(r) < 1:
        z, se = math.atanh(r), 1 / math.sqrt(n - 3)
        ci = (math.tanh(z - 1.959964 * se), math.tanh(z + 1.959964 * se))
    else:
        ci = (r, r)
    return {"r": r, "df": df, "p": p, "n": n, "ci": ci, "r2": r * r}


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    out = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x, y):
    r = pearson(ranks(x), ranks(y))
    return {"rho": r["r"], "p": r["p"], "n": r["n"], "df": r["df"]}


def mann_whitney(a, b):
    """U with a normal approximation, corrected for ties and continuity."""
    n1, n2 = len(a), len(b)
    r = ranks(list(a) + list(b))
    u1 = sum(r[:n1]) - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    n = n1 + n2
    ties = {}
    for v in list(a) + list(b):
        ties[v] = ties.get(v, 0) + 1
    adj = sum(t ** 3 - t for t in ties.values())
    var = n1 * n2 / 12 * ((n + 1) - adj / (n * (n - 1)))
    if var <= 0:
        raise ValueError("all values are tied")
    z = (abs(u1 - n1 * n2 / 2) - 0.5) / math.sqrt(var)
    z = max(0.0, z)
    return {"U": u, "z": z, "p": 2 * (1 - normal_cdf(z)), "r": z / math.sqrt(n), "n1": n1, "n2": n2}


def regression(y, xs, names=None):
    """Least squares with an intercept. xs: list of predictor columns."""
    n, k = len(y), len(xs)
    if n <= k + 1:
        raise ValueError("need more cases than predictors")
    names = names or ["x%d" % (i + 1) for i in range(k)]
    X = [[1.0] + [c[i] for c in xs] for i in range(n)]
    p = k + 1
    xtx = [[sum(X[r][a] * X[r][b] for r in range(n)) for b in range(p)] for a in range(p)]
    xty = [sum(X[r][a] * y[r] for r in range(n)) for a in range(p)]
    inv = _invert(xtx)
    beta = [sum(inv[a][b] * xty[b] for b in range(p)) for a in range(p)]
    fit = [sum(b * v for b, v in zip(beta, row)) for row in X]
    res = [a - b for a, b in zip(y, fit)]
    sse = sum(e * e for e in res)
    my = mean(y)
    sst = sum((v - my) ** 2 for v in y)
    if sst == 0:
        raise ValueError("the outcome has no variation")
    df = n - p
    mse = sse / df
    r2 = 1 - sse / sst
    adj = 1 - (1 - r2) * (n - 1) / df
    coefs = []
    for i in range(p):
        se = math.sqrt(max(0.0, mse * inv[i][i]))
        t = beta[i] / se if se else float("inf")
        h = t_critical(df) * se
        coefs.append({"name": "(Intercept)" if i == 0 else names[i - 1], "b": beta[i], "se": se, "t": t, "p": t_p(t, df) if se else 0.0, "ci": (beta[i] - h, beta[i] + h)})
    f = (sst - sse) / k / mse if mse else float("inf")
    return {"coefs": coefs, "r2": r2, "adj": adj, "F": f, "df1": k, "df2": df, "p": f_p(f, k, df) if mse else 0.0, "n": n, "sigma": math.sqrt(mse)}


def _invert(m):
    n = len(m)
    a = [list(r) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(m)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-12:
            raise ValueError("the predictors are perfectly collinear")
        a[c], a[p] = a[p], a[c]
        d = a[c][c]
        a[c] = [v / d for v in a[c]]
        for r in range(n):
            if r != c:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [r[n:] for r in a]
