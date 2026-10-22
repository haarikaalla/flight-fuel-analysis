# Related research and the precise gap

This focused review was first assembled in October 2026 and is **not** a systematic novelty survey. Scores from papers with different inputs, labels and splits are not directly comparable.

| Work | What is already established | What this project studies instead |
| --- | --- | --- |
| [Sun, Spinielli and Strohmeier (2026)](https://doi.org/10.59490/joas.2026.8750) | The public PRC dataset pairs ACARS-derived fuel differences with OpenSky ADS-B and aircraft metadata. | Whether a selected *subset* of intervals remains representative of the duration and aircraft mix in this benchmark. |
| [Grigoriou et al. (2026)](https://doi.org/10.59490/joas.2026.8764) and [Kačar and Ćulibrk (2026)](https://doi.org/10.59490/joas.2026.8766) | Strong fuel prediction with physics-guided and weather/trajectory features. | An auditable decision about when this simpler ADS-B and metadata estimate is included, with error and coverage shown together. We do not claim a better universal predictor. |
| [Stawiarska (2026)](https://doi.org/10.3390/en19204797) | Missingness, data quality and prediction uncertainty have already been investigated, including a smaller observed subset from this benchmark. | A full released-benchmark batch-level access audit across interval duration and aircraft type. Overlap remains; distinctiveness is empirical and narrow. |
| [Zaffran et al. (2023)](https://proceedings.mlr.press/v202/zaffran23a.html) and [Romano et al. (2019)](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) | Established uncertainty methods and problems with missing inputs. | No new conformal theorem or calibrated per-interval confidence statement is claimed. |
| [Shah et al. (2022)](https://proceedings.mlr.press/v162/shah22a.html) | Selective regression can magnify subgroup disparities; fairness criteria require group performance to improve as coverage falls. | We operationalize the access problem for aircraft and duration groups and expose its fuel-representation cost. |
| [Gibbs, Cherian and Candès (2023)](https://arxiv.org/abs/2305.12616) | Conditional conformal guarantees can target predefined group families under explicit assumptions. | We compare global and Mondrian diagnostics and identify aircraft-transfer failure; we do not claim a new coverage theorem. |
| [Jarry et al. (2025)](https://doi.org/10.1016/j.trc.2025.105143) | Domain-generalized fuel-flow models can be evaluated on aircraft types unseen in training. | Our current aircraft holdout applies only to the risk calibrator; the frozen protocol requires full fuel-model aircraft holdout when raw inputs are available. |

**Question:** If an error-ranking rule keeps roughly 75% of a batch, does it also retain enough long intervals and major aircraft types to support a comparison? What accepted-case error and labeled-fuel representation result from requiring group access floors? This is a test of a service criterion, not proof that a greedy quota rule is a novel ML method.

The original Phase 4 October result motivated the floor: low accepted MAE concealed near-zero access to long intervals and B744. Therefore both the September comparison and post-revision October scores are **exploratory** for this new rule. A later independently labeled period and a systematic comparison with related selection policies are required for a stronger publication claim. See [methods](methods.md) for the exact evaluation boundary.

The strongest present finding is not that every group can be made equally
accurate. It is that marginal uncertainty and selective accuracy can both look
acceptable while wide-body transfer remains poor. Duration-Mondrian intervals
are narrower and more uniform on the exploratory batch, whereas aircraft-held-
out risk calibration falls to 75.76% coverage for B744. This motivates the
prospective test rather than completing it.
