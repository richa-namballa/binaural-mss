### Inference

To run inference on either MUSDB18-HQ or Binaural-MUSDB from the `inference/` directory:

**HT Demucs (FT)**
```
python demucs.py \
    /path/to/input/data/split \
    -o /path/to/output/directory
```

**Open-Unmix**
```
python open_umx.py \
    /path/to/input/data/split \
    -o /path/to/output/directory
```

**Spleeter**
```
python spleeter.py \
    /path/to/input/data/split \
    -o /path/to/output/directory
```

**Example Usage**
```
python demucs.py \
    data/binaural_musdb18/test \
    -o data/output/htdemucs_ft/binaural/test
```

## References

[1] S. Rouard, F. Massa, and A. Défossez, "Hybrid transformers for music source separation," in _2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_. IEEE, 2023, pp. 1–5.

[2] F.-R. Stöter, S. Uhlich, A. Liutkus, and Y. Mitsufuji, "Open-Unmix - a reference implementation for music source separation," _Journal of Open Source Software_,
vol. 4, no. 41, p. 1667, 2019. [Online]. Available: https://doi.org/10.21105/joss.01667

[3] R. Hennequin, A. Khlif, F. Voituret, and M. Moussallam, "Spleeter: a fast and efficient music source separation tool with pre-trained models," _Journal of Open Source Software_, vol. 5, no. 50, p. 2154, 2020.
