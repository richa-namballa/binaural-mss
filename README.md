# Binaural Musical Source Separation (MSS)

Repository for **Do Music Source Separation Models Preserve Spatial Information in Binaural Audio?** 
by Richa Namballa, Dr. Agnieszka Roginska, and Dr. Magdalena Fuentes. [arXiv](arxiv.org)
> Binaural audio remains underexplored within the music information retrieval community. Motivated by the rising popularity of virtual and augmented reality experiences as well as potential applications to accessibility, we investigate how well existing music source separation (MSS) models perform on binaural audio. Although these models process two-channel inputs, it is unclear how effectively they retain spatial information. In this work, we evaluate how several popular MSS models preserve spatial information on both standard stereo and novel binaural datasets. Our binaural data is synthesized using stems from MUSDB18-HQ and open-source head-related transfer functions by positioning instrument sources randomly along the horizontal plane. We then assess the spatial quality of the separated stems using signal processing and interaural cue-based metrics. Our results show that stereo MSS models fail to preserve  the spatial information critical for maintaining the immersive quality of binaural audio, and that the degradation depends on model architecture as well as the target instrument. Finally, we highlight valuable opportunities for future work at the intersection of MSS and immersive audio.


### Data

Binaural-MUSDB is available on Zenodo.

It's synthesis can also be reproduced with [MUSDB18-HQ](https://zenodo.org/records/3338373), Subject D1's HRIRs from [SADIE II](https://www.york.ac.uk/sadie-project/database.html), and the provided code (and random seed) in `/notebook/binaural_data_syn.ipynb`.



### BibTeX Citation

```
@inproceedings{Namballa2025,
  title = {Do Music Source Separation Models Preserve Spatial Information in Binaural Audio?},
  author = {Namballa, Richa and Rogisnka, Agnieszka and Fuentes Magdalena},
  booktitle = {Proceedings of the 26th International Society for Music Information Retrieval Conference},
  year = {2025},
  address = {Daejeon, South Korea},
  organization = {ISMIR}
}
```

### References

* Z. Rafii, A. Liutkus, F.-R. Stöter, S. I. Mimilakis, and R. Bittner, "MUSDB18-HQ - an uncompressed version of MUSDB18," Dec. 2019. [Online]. Available: https://doi.org/10.5281/zenodo.3338373
* C. Armstrong, L. Thresh, D. Murphy, and G. Kearney, "A perceptual evaluation of individual and non-individual HRTFs: A case study of the SADIE II database," _Applied Sciences_, vol. 8, no. 11, p. 2029, 2018.
