# PQ-FL: Post-Quantum Secure Federated Learning with Distributed Differential Privacy and Secure Aggregation

This repository contains the source code, data, and results for the manuscript



## Main results

| Benchmark | PQ-FL ($\sigma=1.0$) | DP-FedAvg ($\sigma_{\text{local}}=1.0$) | PQ-FL at equal aggregate privacy ($\sigma=\sqrt{10}$) | Non-private FedAvg |
|---|---|---|---|---|
| Synthetic 6G-IoT ($N=100$) | 95.50 ± 0.25 % | 91.18 ± 1.00 % | 91.29 ± 1.08 % | 95.88 ± 0.44 % |
| UCI Digits ($N=50$) | 82.67 ± 2.21 % | 60.89 ± 3.32 % | 60.78 ± 5.86 % | 97.33 ± 1.16 % |
| LEAF FEMNIST ($N=100$ writers) | 26.78 ± 3.26 % | 10.82 ± 3.84 % | 10.94 ± 2.42 % | 66.84 ± 1.62 % |

Values are the mean and sample standard deviation of the final test accuracy over the five seeds 42, 123, 456, 789, and 1024, with $T=200$ rounds and $|S_t|=10$ clients per round.

* At $\sigma=1.0$, PQ-FL and DP-FedAvg give the honest-but-curious server the same view of each update ($\epsilon_{\text{RDP}}=11.06$ on 6G-IoT and FEMNIST, $23.42$ on UCI Digits), and PQ-FL is more accurate on all three benchmarks (paired $t$-tests, $p<10^{-3}$).
* At equal aggregate privacy ($\sigma=\sqrt{10}$), the two methods cannot be distinguished, but PQ-FL limits the server's view of each update to the aggregate budget and encrypts every upload.
* Privacy budgets at $\delta=10^{-5}$ from the dp-accounting library: $\epsilon_{\text{RDP}}=11.06$ and $\epsilon_{\text{PLD}}=9.97$ at $\sigma=1.0$, and $3.68$, $2.06$, and $1.56$ ($\epsilon_{\text{PLD}}=3.36$, $1.88$, and $1.42$) at $\sigma=2.0$, $\sqrt{10}$, and $4.0$ on 6G-IoT.



## Installation

Python 3.10 or later is required. The reported results were obtained with Python 3.12.10, PyTorch 2.2.2 (CPU), NumPy 1.26.4, pandas 3.0.5, SciPy 1.17.1, scikit-learn 1.3.2, and dp-accounting 0.6.0 on Windows 11 with an Intel Core x86-64 processor.

```bash
pip install -r requirements.txt
```

## Datasets

**Synthetic 6G-IoT.** 10,000 records with seven features (signal strength, network latency, heart rate, spatial coordinates $X$ and $Y$, battery level, and data volume) and four device-category labels. The label is drawn first from the uniform distribution, and each feature is then drawn from a class-conditional distribution and clipped to its range. The records are split into 8,000 training and 2,000 test records, and the training records are partitioned across 100 clients by per-class Dirichlet allocation with $\alpha=0.5$. The generator is `src/data/synthetic.py`, the generated files are `dataset/6G_IoT_Federated_Data_seed<seed>.csv`, and all parameters are listed in `dataset/Dataset_Description.txt` The pipeline regenerates these files from the seeds.

**UCI Optical Recognition of Handwritten Digits** (Alpaydin and Kaynak, 1998, https://doi.org/10.24432/C50P49). 1,797 images of $8\times8$ pixels in 10 classes, loaded through scikit-learn and partitioned across 50 clients by per-class Dirichlet allocation with $\alpha=0.5$.

**LEAF FEMNIST** (Caldas et al., 2018, https://leaf.cmu.edu/). 3,400 writers and 62 classes with the natural writer partition; 100 writers are sampled for each seed. The archive (about 170 MB) is downloaded on first use from the TensorFlow Federated public storage, verified against its SHA-256 checksum, and extracted to `dataset/femnist_cache/`. It is not stored in this repository.

## Reproducing the results

All commands are run from the repository root.

1. Unit tests of the cryptographic layer (a few seconds):
   ```bash
   python tests/test_crypto.py
   ```
2. Dataset checks:
   ```bash
   python scripts/check_dataset.py
   ```
3. Smoke test of every pipeline phase with 2 seeds and 2 rounds. Its output is written to `results_smoke/`, 
   ```bash
   python main.py --smoke
   ```
4. Complete pipeline with 5 seeds and 200 rounds. This regenerates all files in `results/` and `Supplementary_Data_Sheet/` and takes several hours on one CPU thread:
   ```bash
   python main.py
   ```
   The phases are: pairwise Ring-LWE key establishment, the 6G-IoT study with eight methods, the privacy budget sweep and the comparison at equal aggregate privacy, UCI Digits, LEAF FEMNIST, privacy accounting with payload and decryption audits, figures, and the key-establishment scaling benchmark. The FEMNIST benchmark can also be run on its own with `python femnist_benchmark.py`.
5. Verification of all reported values against the per-seed results:
   ```bash
   python scripts/verify_reported_numbers.py
   ```
  

Timing values depend on the hardware. The training results are deterministic for a given seed and software environment.

## Where each result is found

|  item | File in `results/` |
|---|---|
| Section 5.2, privacy budgets | `privacy_budget_report.txt` |
| Figures 2 to 5, accuracy, $F_1$, AUC-ROC, and MAE curves | `comparison_accuracy.png`, `comparison_f1.png`, `comparison_auc.png`, `comparison_mae.png`, `comprehensive_metrics.json` |
| Section 6.5 and Table 4, per-seed accuracies and paired $t$-tests | `statistical_report.txt` |
| Figure 6, radar profile with Seed Stability | `comparison_radar.png`, `radar_metrics.json` |
| Figure 7, privacy-accuracy trade-off | `comparison_tradeoff.png` |
| Tables 5 and 6, ablation and privacy fairness | `Privacy_Fairness_Table.csv`, `statistical_report.txt` |
| Table 8, privacy budget sweep and equal aggregate privacy | `Privacy_Matched_Ablation.csv`, `privacy_ablation_report.txt` |
| Table 9 and Figure 8, UCI Digits | `Public_Benchmark_UCI_Digits.csv`, `public_digits_stats.txt`, `public_digits_accuracy.png` |
| Table 10 and Figure 9, LEAF FEMNIST | `Public_Benchmark_LEAF_FEMNIST.csv`, `femnist_stats.txt`, `femnist_accuracy.png` |
| Table 11 and Figure 10, pairwise key establishment | `scaling_analysis.csv`, `scaling_report.txt`, `scaling_combined.png` |
| Table 12, measured per-round overhead | `Cryptographic_Overhead_Metrics.csv`, `Timing_Raw_Per_Seed.csv`, `payload_audit.txt` |
| Section 7.2, decryption error margin | `decryption_failure_report.txt` |
| Clipping diagnostics | `Run_Diagnostics_Per_Seed.csv` |
| Software environment of the reported run | `environment.txt` |

The per-round accuracy, macro $F_1$-score, macro one-vs-rest AUC-ROC, and MAE of every run are in `Supplementary_Data_Sheet/Training_Raw_Metrics.csv` (6G-IoT) and `Supplementary_Data_Sheet/FEMNIST_Raw_Metrics.csv` (LEAF FEMNIST).

## Scope

The guarantees hold for an honest-but-curious server, passive eavesdroppers, collusion of fewer than $n-1$ clients with the server, and synchronous round completion. The protocol does not protect against Byzantine clients or other active adversaries, and it does not tolerate client dropout. The Ring-LWE parameters are a proof-of-concept instantiation and are not a standardized ML-KEM parameter set. The privacy budgets use Poisson subsampling as an approximation of the fixed-size client sampling of the implementation. All experiments are single-machine simulations in which the clients run one after another.

## Citation

If you use this code, please cite the manuscript above. The citation will be updated with the publication details.

## Contact

Corresponding author: M.Mann, Rakesh p. Badoni and Deepak Panwar.


