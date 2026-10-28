[README.txt](https://github.com/user-attachments/files/34065271/README.txt)
ADAPT-PM study: outputs package (v1.1)
======================================
Manuscript_ADAPT-PM_v1.1.docx / .pdf   Revised manuscript (v1.1, after review)
Tables_ADAPT-PM.xlsx                   All result tables, one sheet each
                                       (T3a, T5-T8, T7a and S1 match the manuscript)
figures/                               Fig 1 (PNG) and Figs 2-6 (PNG 300 dpi + SVG)
  fig1_adapt_pm_loop       ADAPT-PM decision loop
  fig2_cost_by_scenario    RQ1: normalised cost by scenario
  fig3_cost_by_delay       RQ2: cost vs label delay (records)
  fig4_ablations           RQ3: ablations by scenario group
  fig5_update_cost         Cost gate vs update cost C_U
  fig6_metropt3            RQ4: MetroPT-3 onset detection (exploratory)
tables_csv/                            Raw analysis tables as CSV, incl. S1 (all 38 conditions with CIs)
results_raw/                           Per-run results (pandas pickles) + Friedman stats
code/                                  Full pipeline (Python 3.13, scikit-learn 1.9, pandas 3.0, scipy 1.18)

Reproduce:
  python3 code/ai4i.py ai4i2020.csv results/ai4i_main.pkl --plan main --seeds 10
  python3 code/ai4i.py ai4i2020.csv results/ai4i_s6rest.pkl --plan s6rest --seeds 10   (concatenate with main)
  python3 code/ai4i.py ai4i2020.csv results/ai4i_gate.pkl --plan gate --seeds 10
  python3 code/metro_features.py "MetroPT3(AirCompressor).csv" results/metro_windows.pkl
  python3 code/metro_sim.py results/metro_windows.pkl results/metro_runs.pkl
  python3 code/analyze_ai4i.py results ; python3 code/analyze_metro.py results
  python3 code/make_outputs.py results out

Design: 38 AI4I conditions = 7 scenarios x 4 delays at the middle magnitude (28)
        + low and high magnitudes of S1-S5 at d = 50 (10); 10 seeds each = 380 replays.
        All 380 enter every analysis; none excluded.

Known limits: 10 seeds; own ADWIN / ARF-style baselines (River unavailable); Random Forest only;
rejected gate candidates not charged in main results (see Table 7a); MetroPT-3 exploratory
(design revised after test inspection, Table 2a). See manuscript Section 7.3.

Changelog v1.1 (18 Oct 2026, after review)
- Reduction under concept drift corrected to 34-50% (S3 33.7%), not 40-50%.
- Oracle gap reported per scenario (-0.007 to +0.025); 0.005 is the mean only.
- Condition count fixed to 38 everywhere; replay accounting stated.
- Table S1 added: P6-P1 and P6-P2 with 95% bootstrap CIs for all 38 conditions
  (vs P1: 23 better, 5 worse; vs periodic: 5 better (one marginal), 15 worse).
- Table 7a added: sensitivity to charging rejected gate candidates.
- Table 3a added (failures before/after relabelling); Table 2a (MetroPT-3 post-hoc decisions).
- Counting rules, threshold assumptions and the full ADAPT-PM specification added.
- References 4, 14 and 16 completed.
