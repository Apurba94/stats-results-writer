# Stats Results Writer

Run a common test on your data and get the paragraph you would write in a results section, in APA style, with the effect size and
confidence interval already in place.

* **Data:** open a CSV or paste from a spreadsheet. Each variable gets its type, mean, SD, median, range and 95% CI.
* **Tests:** independent t (Welch and Student), paired t, one-sample t, one-way ANOVA with Bonferroni pairwise comparisons, chi-square
  with Cramer's V, Pearson and Spearman correlation, Mann-Whitney U, and multiple linear regression.
* **Write-up:** for example, `t(17.78) = -1.86, p = .079, d = -0.83, 95% CI [-3.37, 0.21]`, with p values and leading zeros handled.
* **Assumption notes:** unequal variances, low expected counts, small samples and non-normal groups.
* **Which test?** a short guide from your question to the right test.

Everything is computed locally, with no external statistics package. `--demo` opens a temporary database with a fictional dataset.

## From source

```bash
python -m stats_results_writer
python -m unittest discover -s tests -t .
python build_exe.py
```

The tests check results against published values for Student's sleep data and the PlantGrowth data, and against critical-value
tables for the t, F and chi-square distributions.

## Limits

This covers everyday tests, not mixed models, repeated-measures ANOVA or logistic regression. The normality note uses the Jarque-Bera
test, which is rough in small samples. Effect size labels (small, medium, large) follow Cohen's general rules of thumb, and your field
may set them differently. You still choose the right test for your design, and the wording is a starting point.

## License

MIT
