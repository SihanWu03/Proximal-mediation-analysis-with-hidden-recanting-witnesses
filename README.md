# Path-specific effects with hidden recanting witnesses

Python implementation of the simulation studies and ACTG175-based application accompanying the manuscript on identification and inference for path-specific effects with hidden recanting witnesses.

The repository implements outcome regression (POR1 and POR2), hybrid (PH), inverse probability weighting (PIPW), and multiply robust (PMR) estimators. The nonparametric study includes direct proxy substitution (Naive). The application includes an observed-mediator multiply robust comparator (MR).

## Repository structure

```text
run_parametric.py             Parametric robustness simulation
run_nonparametric.py          Two nonparametric simulation cases
run_application.py            ACTG175-based application
make_figures_and_tables.py    Manuscript tables and boxplots
parametric/                  Parametric moment models and estimators
pse/                         Joint minimax bridges, kernels, and cross-fitting
application/                 Clinical calibration, generation, and MR comparator
data/ACTG175.csv              Public clinical input data
requirements.txt             Simulation dependencies
requirements-application.txt Application dependencies
```

## Installation

Use separate environments to reproduce the numerical environments of the simulations and application. Run the commands below from the repository root. Any installed Python executable may be substituted for `python`; on Windows, a virtual environment uses `Scripts/python.exe` instead of `bin/python`.

For the simulation studies, use Python 3.11:

```bash
python -m venv .venv-simulation
.venv-simulation/bin/python -m pip install -r requirements.txt
```

For the application, use Python 3.12:

```bash
python -m venv .venv-application
.venv-application/bin/python -m pip install -r requirements-application.txt
```

In the examples below, `python` denotes the interpreter from the appropriate environment. The nonparametric simulations were run with CUDA-enabled PyTorch 2.11.0, CUDA 12.8, and an NVIDIA RTX 4090. Install a compatible PyTorch build if necessary. `--device cpu` provides a CPU alternative. All kernel calculations use double precision; small numerical differences may occur across devices, numerical libraries, and platforms. Full nonparametric simulations require substantially more time than the application.

## Parametric simulation

```bash
python run_parametric.py
```

The default run uses 1,000 repetitions at each sample size `1000,2000,4000,8000`. The five scenarios share each generated dataset. Outcome bridges use affine working models, treatment bridges use exponential-affine working models, treatment probabilities use logistic regression, and eta uses linear regression. These models are fitted on the full sample.

The manuscript scenario order is:

| Scenario | Correctly specified models |
| --- | --- |
| 1 | h0, h1, eta |
| 2 | P, h0, h1 |
| 3 | P, h0, q1 |
| 4 | P, q0, q1 |
| 5 | All working models |

Misspecified outcome and treatment bridges omit their respective proxies. The treatment probability model in Scenario 1 and the eta regression in Scenario 2 omit X. Eta retains X in Scenarios 3 and 4 but is fitted to a misspecified h1 response. Thus its feature specification alone does not imply correct specification.

For zero-based repetition `r`, the data seed is `1 + 4*r + index(n)`, where `index(n)` refers to the fixed sample-size order above. The target is the nested counterfactual mean

```text
psi = E[Y(M2(M1(0), 1), M1(0), 0)].
```

Outputs are `outputs/parametric/detailed.csv`, `summary.csv`, and `inference_summary.csv`, together with the run configuration and per-dataset checkpoints. Failed fits are recorded explicitly and included in completion counts. Bias and RMSE summaries use successful fits. PMR intervals reproduce the manuscript's original influence-function calculation, including under misspecification; they do not add a nuisance-estimation correction for misspecified models.

## Nonparametric simulation

```bash
python run_nonparametric.py --device cuda:0
```

The default run uses both cases, 1,000 repetitions per sample size, and five-fold cross-fitting. Case 1 has dimensions `(1,2,2,1,1)` for `(X,Z,W,M1,M2)`; Case 2 has dimensions `(5,3,3,2,1)`. M1 is hidden and Y is scalar in both cases. The complete generating coefficients are specified in `pse/simulation.py`.

The bridge learner uses regularized joint minimax estimation with 128 Nystrom landmarks and a composite Gaussian/linear kernel. The final h0 and q1 are fitted separately; h1 and q0 are obtained from joint problems with auxiliary first-stage bridges. Those auxiliary estimates do not enter the final scores. Treatment probabilities use random forests, and eta uses kernel ridge regression. Naive substitutes the combined proxies `(Z,W)` for the hidden mediator and fits observed-data regressions with random forests.

For case `c` and zero-based repetition `r`, the data seed is `1 + (c-1)*4*R + 4*r + index(n)`, with `R` the requested number of repetitions. The learner seed is `10000 + data_seed`. Consequently, changing `--repetitions` changes Case 2's seed offset. Use the default `R=1000` to reproduce the manuscript sequence. Selecting sample sizes or running cases separately does not alter their canonical indices.

Outputs are written under `outputs/nonparametric/case1/` and `case2/`. Each case contains replicate estimates, estimation errors, bias/RMSE summaries, PMR and Naive interval summaries, the run configuration, and checkpoints. Estimation errors are `estimate - psi`; all completed observations are retained in the boxplots. Interval coverage for both methods is evaluated against the same causal target, psi.

Both simulation runners accept `--resume` with the same scientific settings and output directory:

```bash
python run_parametric.py --resume
python run_nonparametric.py --device cuda:0 --resume
```

Use a fresh output directory for a different configuration. To check execution with a reduced workload, use:

```bash
python run_parametric.py --sample-sizes 1000 --repetitions 1 --workers 1 --output-dir outputs/parametric-small
python run_nonparametric.py --sample-sizes 1000 --repetitions 1 --device cpu --output-dir outputs/nonparametric-small
```

## ACTG175-based application

The clinical input is the public ACTG175 analysis dataset distributed with the R package [speff2trial](https://cran.r-project.org/package=speff2trial). The bundled CSV contains 2,139 participants and includes approximately 96-week CD4 measurements. Its source is the [R-universe ACTG175 data endpoint](https://cran.r-universe.dev/speff2trial/data/ACTG175/csv). The input snapshot is included so that preprocessing does not depend on network access or later changes to the endpoint.

The CSV checksum is:

```text
SHA256: 0cd9133ef7e72c60dd08bbca60ed8939d600a87f4ae4d4b9f8261d8fbd37ba5c
```

Prepare the calibration and run the application in the application environment:

```bash
python -m application.prepare_data
python run_application.py
```

Preprocessing uses age, weight, sex, Karnofsky score, baseline CD4 count, and prior antiretroviral treatment duration. The last two covariates are transformed with `log1p`, and all covariates are standardized. Separate random forests fit `log(1+CD4)` at approximately 96 weeks in the ZDV monotherapy and ZDV + ddI groups. Five-fold prediction residuals determine the outcome scale. Preprocessing uses its fixed calibration seed and verifies the input checksum. An alternative local copy can be supplied with `--source`.

This is a **semi-synthetic application**: clinical data determine the empirical covariate distribution and arm-specific outcome calibration. A single sample of 8,000 baseline vectors is drawn with replacement, using seed 42. Treatment is generated independently with probability 0.5. Early toxicity, later adherence, and two proxies in each proxy group follow the specified linear Gaussian mechanisms; mediator and proxy noise variances are 0.5. Outcomes combine the clinical calibration with prespecified toxicity and adherence effects. These mechanisms and their coefficients are defined in `application/clinical_dgp.py`; they are not inferred from observed clinical toxicity or adherence histories.

The application estimates the path-specific contrast

```text
PSE0 = E[Y(M2(M1(0), 1), M1(0), 0)] - E[Y(0)],
```

in `log(1+CD4)` units. It uses three-fold cross-fitting, known treatment probabilities, and the same joint bridge learner as the nonparametric study. M1 is hidden from the five proxy-based estimators. MR receives the generated M1 and uses normal linear mediator models and a linear outcome model. All six methods share the sample, fold partition, and cross-fitted AIPW control-mean estimate. PMR's standard error is computed from its paired score difference with the control score, retaining their covariance. Its interval is conditional on the fitted clinical calibration.

The run writes `results.csv`, `result.json`, `run_config.json`, the generated observed sample, and held-out scores to `outputs/application/`. The JSON includes the known generating truth and PMR interval. Choose a fresh `--output-dir` to repeat a completed run.

## Manuscript figures and tables

After completing the three runs:

```bash
python make_figures_and_tables.py
```

This generates the following files in `outputs/manuscript/`:

| File | Content |
| --- | --- |
| `parametric_table.tex` | Bias and RMSE for the five estimators, with PMR interval coverage and mean length |
| `table2_nonparametric_pmr_naive.tex` | Bias, RMSE, interval coverage, and mean length for PMR and Naive in both cases |
| `application_table.tex` | Application point estimates and the generating truth |
| `nonparametric_figure.tex` | Two-panel horizontal boxplot of estimation errors |
| `nonparametric_errors.pdf` and `.png` | Boxplot graphics with a shared legend, narrow boxes, and all estimation errors retained |

The nonparametric table groups PMR and Naive under each sample size, with four metrics for each case. Coverage is expressed as a percentage in the table note. Its reference label is `tab:nonparametric_pmr_naive`; the figure and application table use `fig:nonparametric_residuals` and `tab:clinical_pse`, respectively. The application's PMR interval is recorded in its analysis output. LaTeX table and figure fragments require `booktabs` and `graphicx`.

Individual studies can be exported with `--study parametric`, `--study nonparametric`, or `--study application`. `--results-dir` and `--output-dir` select alternative input and export directories. The exporter refuses incomplete or failed simulation runs. Keep the exported PDF with the figure fragment and set the LaTeX graphics search path to that directory.
