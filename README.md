# sepx-paper

This repository contains all the source codes to reproduce the experimental results reported in paper "Shortest Edit Path Crossover: A Theory-driven Solution to the Permutation Problem in Evolutionary Neural Architecture Search", which is published in ICML 2023. (Arxiv Link: https://arxiv.org/abs/2210.14016)

## How to use

Please ```pip install networkx==2.5``` first, then replace ```networkx/algorithms/similarity.py``` with the attached ```similarity.py```. 

For other dependency requirements and specific guidelines about how to run the source codes for reproducing the experimental results, please see the README files in
each individual folder. More specifically: \
- ```nas-bench-101``` includes all the source codes for reproducing the main experimental results, which are using NAS-Bench-101 dataset.
- ```nas-bench-nlp``` includes all the source codes for reproducing the experimental results related to NAS-Bench-NLP dataset.
- ```nas-bench-301``` includes all the source codes for reproducing the experimental results related to NAS-Bench-301 dataset.
- ```expected_improvement_heatmap``` includes all the source codes for reproducing the results of the numerical analysis in Section "Comparisons based on Theory".

## Citation

If you use SEPX in your research, please cite it using the following BibTeX entry:
```
@InProceedings{qiu:icml23,
  title={Shortest Edit Path Crossover: A Theory-driven Solution to the Permutation Problem in Evolutionary Neural Architecture Search},
  author={Qiu, Xin and Miikkulainen, Risto},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  pages={28422--28447},
  year={2023},
  volume={202},
  series={Proceedings of Machine Learning Research},
  month={23--29 Jul},
  publisher={PMLR},
}
```
