# Binaural Musical Source Separation (MSS)

# Your Project Title

[![arXiv](https://img.shields.io/badge/arXiv-2507.00155-b31b1b.svg)](https://arxiv.org/abs/2507.00155)[![arXiv](https://img.shields.io/badge/arXiv-2607.25919-b31b1b.svg)](https://arxiv.org/abs/2607.25919) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Bi--SCNet-yellow)](https://huggingface.co/richa-namballa/Bi-SCNet)

This is the accompanying repository for:
* **"Spacing Out: On the Reliability of Binaural Music Source Separation Metrics"** 
by R. Namballa and M. Fuentes, (2026). [Demo Page](https://richa-namballa.github.io/binaural-mss-demo).
* **"Do Music Source Separation Models Preserve Spatial Information in Binaural Audio?"** 
by R. Namballa, A. Roginska, and M. Fuentes, (2025). [Demo Page](https://richa-namballa.github.io/spacing-out-demo/).

#### Binaural-MUSDB

See [`data/`](https://github.com/richa-namballa/binaural-mss/tree/main/data) for instructions on how to recreate Binaural-MUSDB or perform additional binaural synthesis experiments.

#### Training

For training a binaural music source separation model, see [Bi-SCNet](https://github.com/richa-namballa/Bi-SCNet/tree/main).

#### Inference

See [`inference/`](https://github.com/richa-namballa/binaural-mss/tree/main/inference) to test existing stereophonic music source separation models on binaural music.

#### Evaluation

See [`eval/`](https://github.com/richa-namballa/binaural-mss/tree/main/eval) for implementations of metrics we use to evaluate binaural music source separation.

## BibTeX Citations

```bibtex
@inproceedings{Namballa2026,
  title = {Spacing Out: On the Reliability of
  Binaural Music Source Separation Metrics},
  author = {Namballa, Richa and Fuentes, Magdalena},
  booktitle = {Proceedings of the 27th International
  Society for Music Information Retrieval Conference},
  year = {2026},
  address = {Abu Dhabi, UAE},
  organization = {ISMIR}
}
```

```bibtex
@inproceedings{Namballa2025,
  title = {Do Music Source Separation Models Preserve
  Spatial Information in Binaural Audio?},
  author = {Namballa, Richa and Roginska, Agnieszka and Fuentes, Magdalena},
  booktitle = {Proceedings of the 26th International
  Society for Music Information Retrieval Conference},
  year = {2025},
  address = {Daejeon, South Korea},
  organization = {ISMIR}
}
```
