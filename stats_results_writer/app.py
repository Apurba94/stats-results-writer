"""Stats Results Writer desktop program."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import appkit as ak
from . import report as rp
from . import stats as st
from .appkit import Entity, Field, entities

SCHEMA = entities(
    Entity("dataset", "Dataset", "Datasets", "name", [
        Field("name", "Name", required=True, list_col=True, width=28),
        Field("rows", "Rows", "int", list_col=True, width=7),
        Field("day", "Added", "date", list_col=True, width=11),
        Field("data", "Data", "long"),
    ], "id DESC"),
    Entity("result", "Result", "Saved results", "title", [
        Field("title", "Analysis", required=True, list_col=True, width=30),
        Field("day", "Date", "date", list_col=True, width=11),
        Field("text", "Write-up", "long", list_col=True, width=60),
    ], "id DESC"),
)
TESTS = ["Independent t test", "Paired t test", "One-sample t test", "One-way ANOVA", "Chi-square test", "Pearson correlation", "Spearman correlation", "Mann-Whitney U", "Linear regression"]


def sample(store):
    store.insert("dataset", name="Sample sleep study", rows=16, day=ak.today(), data=rp.SAMPLE_CSV)


def need(cond, msg):
    if not cond:
        raise ValueError(msg)


def run_test(ds, test, a, b, group, mu=0.0, predictors=()):
    """(write-up, table rows, notes) for one analysis."""
    if test in ("Independent t test", "Mann-Whitney U", "One-way ANOVA"):
        need(a and group, "choose an outcome and a grouping variable")
        g = ds.groups(a, group)
        names = sorted(g)
        if test == "One-way ANOVA":
            need(len(names) >= 2, "the grouping variable needs at least 2 levels")
            lists = [g[n] for n in names]
            r = st.anova(lists)
            rows = [["%s" % n, len(g[n]), rp.num(st.mean(g[n])), rp.num(st.sd(g[n]) if len(g[n]) > 1 else 0)] for n in names]
            text = rp.apa_anova(r, group, a)
            notes = rp.notes_anova(lists)
            if len(names) > 2:
                rows += [["%s vs %s" % (x, y), "diff %s" % rp.num(d), "t = %s" % rp.num(t), "adj. " + rp.p_text(p)] for x, y, d, t, p in st.pairwise_bonferroni(names, lists)]
            return text, rows, notes
        need(len(names) == 2, "the grouping variable must have exactly 2 levels here (it has %d)" % len(names))
        x, y = g[names[0]], g[names[1]]
        rows = [[n, len(g[n]), rp.num(st.mean(g[n])), rp.num(st.sd(g[n]))] for n in names]
        if test == "Mann-Whitney U":
            return rp.apa_mw(st.mann_whitney(x, y), names[0], names[1]), rows, []
        return rp.apa_independent(st.independent_t(x, y), names[0], names[1], a), rows, rp.notes_two_groups(x, y)
    if test == "Paired t test":
        need(a and b, "choose two numeric variables")
        x, y = ds.pairs([a, b])
        return rp.apa_paired(st.paired_t(x, y), a, b), [], []
    if test == "One-sample t test":
        need(a, "choose a variable")
        return rp.apa_one(st.one_sample_t(ds.numeric(a), mu), a), [], []
    if test == "Chi-square test":
        need(a and b, "choose two categorical variables")
        rows, cols, t = ds.crosstab(a, b)
        r = st.chi_square(t)
        return rp.apa_chi(r, "%s and %s" % (a, b)), [[rows[i]] + t[i] for i in range(len(rows))], rp.notes_chi(r)
    if test in ("Pearson correlation", "Spearman correlation"):
        need(a and b, "choose two numeric variables")
        x, y = ds.pairs([a, b])
        if test.startswith("Pearson"):
            return rp.apa_pearson(st.pearson(x, y), a, b), [], []
        return rp.apa_spearman(st.spearman(x, y), a, b), [], []
    if test == "Linear regression":
        need(a and predictors, "choose an outcome and at least one predictor")
        cols = ds.pairs([a] + list(predictors))
        r = st.regression(cols[0], cols[1:], list(predictors))
        rows = [[c["name"], rp.num(c["b"]), rp.num(c["se"]), rp.num(c["t"]), rp.p_text(c["p"])] for c in r["coefs"]]
        return rp.apa_regression(r, a), rows, []
    raise ValueError("unknown test")


class Data(ttk.Frame):
    title = "Data"

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        top = ttk.Frame(self)
        top.pack(fill="x")
        self.name = tk.StringVar()
        self.box = ak.labeled_combo(top, "Dataset", self.name, [], 28, self.load)
        self.box.pack(side="left")
        ttk.Button(top, text="Open CSV", style="Accent.TButton", command=self.open).pack(side="left", padx=8, anchor="s")
        ttk.Button(top, text="Paste", command=self.paste).pack(side="left", anchor="s")
        self.info = ttk.Label(self, text="", style="Muted.TLabel")
        self.info.pack(anchor="w", pady=6)
        self.table = ak.Table(self, app, [("name", "Variable", 16), ("kind", "Type", 11), ("n", "n", 6), ("mean", "Mean", 9), ("sd", "SD", 9), ("median", "Median", 9), ("min", "Min", 9), ("max", "Max", 9), ("ci", "95% CI", 22)])
        self.table.pack(fill="both", expand=True)
        self.rows, self.ds = {}, None

    def add(self, name, text):
        try:
            ds = rp.Dataset(text)
        except (ValueError, csv_error) as exc:
            messagebox.showerror("Data", str(exc), parent=self)
            return False
        self.app.store.insert("dataset", name=name, rows=len(ds.rows), day=ak.today(), data=text)
        self.refresh()
        self.name.set(name)
        self.load()
        return True

    def open(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("Tables", "*.csv *.tsv *.txt"), ("All files", "*.*")])
        if path:
            import os
            with open(path, encoding="utf-8-sig", errors="replace") as fh:
                self.add(os.path.basename(path), fh.read())

    def paste(self):
        win = tk.Toplevel(self)
        win.title("Paste data with a header row")
        win.geometry("%dx%d" % (620 * self.app.scale, 380 * self.app.scale))
        win.transient(self.winfo_toplevel())
        win.configure(bg=self.app.colors["bg"])
        frame, t = ak.make_text(win, self.app.colors, mono=True)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        def go():
            if self.add("Pasted data", ak.get_text(t)):
                win.destroy()
        ttk.Button(win, text="Use this data", style="Accent.TButton", command=go).pack(pady=(0, 8))

    def load(self):
        row = self.rows.get(self.name.get())
        self.ds = rp.Dataset(row["data"]) if row else None
        out = []
        if self.ds:
            for c in self.ds.columns:
                if self.ds.is_numeric(c):
                    d = st.describe(self.ds.numeric(c))
                    f = rp.num
                    out.append([c, "numeric", d["n"], f(d["mean"]), f(d.get("sd", 0)), f(d["median"]), f(d["min"]), f(d["max"]), "[%s, %s]" % (f(d["ci"][0]), f(d["ci"][1])) if "ci" in d else ""])
                else:
                    levels = {}
                    for v in self.ds.col(c):
                        if v:
                            levels[v] = levels.get(v, 0) + 1
                    out.append([c, "categorical", sum(levels.values()), "", "", "", "", "", ak.short(", ".join("%s: %d" % kv for kv in sorted(levels.items())[:6]), 44)])
            self.info.configure(text="%d rows, %d columns" % (len(self.ds.rows), len(self.ds.columns)))
        self.table.fill(out)

    def refresh(self):
        self.rows = {r["name"]: r for r in self.app.store.list("dataset")}
        self.box.box.configure(values=list(self.rows))
        if self.name.get() not in self.rows and self.rows:
            self.name.set(next(iter(self.rows)))
        self.load()


csv_error = __import__("csv").Error


class Analyse(ttk.Frame):
    title = "Analyse"

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        top = ttk.Frame(self)
        top.pack(fill="x")
        self.test, self.a, self.b, self.group, self.mu = [tk.StringVar() for _ in range(5)]
        self.test.set(TESTS[0])
        self.mu.set("0")
        ak.labeled_combo(top, "Test", self.test, TESTS, 22, self.adapt).pack(side="left", padx=(0, 8))
        self.ca = ak.labeled_combo(top, "Outcome or first variable", self.a, [], 18)
        self.ca.pack(side="left", padx=(0, 8))
        self.cb = ak.labeled_combo(top, "Second variable", self.b, [], 16)
        self.cb.pack(side="left", padx=(0, 8))
        self.cg = ak.labeled_combo(top, "Grouping variable", self.group, [], 16)
        self.cg.pack(side="left", padx=(0, 8))
        ak.labeled_entry(top, "Test value", self.mu, 8).pack(side="left", padx=(0, 8))
        ttk.Button(top, text="Run", style="Accent.TButton", command=self.run).pack(side="left", anchor="s")
        ttk.Label(self, text="Predictors for regression: hold Ctrl to pick several", style="Muted.TLabel").pack(anchor="w", pady=(6, 0))
        self.pred = tk.Listbox(self, selectmode="multiple", height=3, exportselection=False, bg=app.colors["panel"], fg=app.colors["ink"], relief="flat",
                               selectbackground=app.colors["sel"], selectforeground=app.colors["sel_ink"], highlightthickness=1, highlightbackground=app.colors["line"])
        self.pred.pack(fill="x")
        ttk.Label(self, text="Write-up", style="Muted.TLabel").pack(anchor="w", pady=(8, 0))
        frame, self.text = ak.make_text(self, app.colors, height=6)
        frame.pack(fill="x")
        self.notes = ttk.Label(self, text="", style="Muted.TLabel", wraplength=int(950 * app.scale), justify="left")
        self.notes.pack(anchor="w", pady=4)
        self.table = ak.Table(self, app, [("a", "", 26), ("b", "", 14), ("c", "", 14), ("d", "", 14), ("e", "", 14)], height=6)
        self.table.pack(fill="both", expand=True)
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Copy", command=self.copy).pack(side="left")
        ttk.Button(row, text="Save to results", command=self.save).pack(side="left", padx=6)

    def ds(self):
        return self.app.view("Data").ds

    def refresh(self):
        data = self.app.view("Data")
        if data.ds is None:
            data.refresh()
        d = data.ds
        cols = d.columns if d else []
        num = [c for c in cols if d.is_numeric(c)]
        for box in (self.ca, self.cb):
            box.box.configure(values=cols)
        self.cg.box.configure(values=cols)
        self.pred.delete(0, "end")
        for c in num:
            self.pred.insert("end", c)
        if d and not self.a.get() and len(num) > 1:
            self.a.set(num[-1 if len(num) < 3 else 2])
            cat = [c for c in cols if not d.is_numeric(c)]
            self.group.set(cat[0] if cat else "")
            self.b.set(num[1])
            self.run()

    def adapt(self):
        self.text.delete("1.0", "end")

    def run(self):
        d = self.ds()
        if d is None:
            self.app.status("Load a dataset first")
            return False
        try:
            mu = float(self.mu.get() or 0)
            preds = [self.pred.get(i) for i in self.pred.curselection()]
            text, rows, notes = run_test(d, self.test.get(), self.a.get(), self.b.get(), self.group.get(), mu, preds)
        except (ValueError, ZeroDivisionError) as exc:
            ak.set_text(self.text, "Cannot run this test: %s" % exc)
            self.notes.configure(text="")
            self.table.fill([])
            return False
        ak.set_text(self.text, text)
        self.notes.configure(text="\n".join("Note: " + n for n in notes))
        self.table.fill([[str(v) for v in r] + [""] * (5 - len(r)) for r in rows])
        return True

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append(ak.get_text(self.text))
        self.app.status("Copied")

    def save(self):
        t = ak.get_text(self.text).strip()
        if t:
            self.app.store.insert("result", title="%s: %s" % (self.test.get(), self.a.get()), day=ak.today(), text=t)
            self.app.status("Saved")


class Guide(ttk.Frame):
    title = "Which test?"

    def __init__(self, parent, app):
        super().__init__(parent)
        t = ak.Table(self, app, [("q", "Your question", 46), ("t", "Test", 24), ("n", "Keep in mind", 40)])
        t.pack(fill="both", expand=True)
        t.fill([list(r) for r in rp.GUIDE])

    def refresh(self):
        pass


class Results(ttk.Frame):
    title = "Saved results"

    def __init__(self, parent, app):
        super().__init__(parent)
        self.view = ak.EntityView(self, app, "result")
        self.view.pack(fill="both", expand=True)
        self.refresh, self.commit = self.view.refresh, self.view.commit


class APP(ak.BaseApp):
    APP_NAME, ACCENT = "Stats Results Writer", "#0369a1"
    SCHEMA, SAMPLE = SCHEMA, staticmethod(sample)
    VIEWS = [Data, Analyse, Guide, Results, ak.DataView]


# the framework's own screen shares the title "Data"
ak.DataView.title = "Backup"


def main(argv=None):
    return ak.main(APP, argv)
