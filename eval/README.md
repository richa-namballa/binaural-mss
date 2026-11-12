## Evaluation

To evaluate the models' performances, we compute the Signal to Spatial Distortion Ratio (**SSR**) and Signal to Residual Distortion (**SRR**) ratios.
The reference and estimated directories should contain subdirectories for each song, containing the stems as WAV files.

Run the scripts from the `2025/` or `2026/` directory.
The years correspond to the version of the metrics used in the [ISMIR 2025 paper](https://arxiv.org/abs/2507.00155) and [ISMIR 2026 paper](https://arxiv.org/abs/2607.25919), respectively.

```
python spauq_metrics.py \
    /path/to/reference/directories \
    /path/to/estimated/directories \
    -o /path/to/output/directory \
    -n name_of_output_csv_file \
    -s [list_of_sources_to_evaluate]
```

Similarly, the difference in interaural time difference (**ΔITD**) and interaural level difference (**ΔILD**) can be calculated.
```
python interaural_metrics.py \
    /path/to/reference/directories \
    /path/to/estimated/directories \
    -o /path/to/output/directory \
    -n name_of_output_csv_file \
    -s [list_of_sources_to_evaluate]
```


The summary statistics can be computed in `notebook/table_analysis.ipynb`. Plots can be replicated in `notebook/plot_analysis.ipynb` but you must run `metadata.ipynb` to aggregate the angle metadata first.

## References

[1] K. N. Watcharasupat and A. Lerch, "Quantifying spatial audio quality impairment," in _2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_. IEEE, 2024, pp. 746–750.

[2] B. Veluri, M. Itani, J. Chan, T. Yoshioka, and S. Gollakota, “Semantic hearing: Programming acoustic scenes with binaural hearables,” in _Proceedings of the
36th Annual ACM Symposium on User Interface Software and Technology_, 2023, pp. 1–15.