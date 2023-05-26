# sepx-paper

This repository contains all the source codes to reproduce the experimental results reported in paper "Shortest Edit Path Crossover: A Theory-driven Solution to the Permutation Problem in Evolutionary Neural Architecture Search", which is published in ICML 2023. ((Arxiv Link: https://arxiv.org/abs/2210.14016)

## How to use

Please '''pip install networkx==2.5''' first, then replace '''networkx/algorithms/similarity.py''' with the attached '''similarity.py'''. 

For other dependency requirements and specific guidelines about how to run the source codes for reproducing the experimental results, please see the README files in
each individual folder.

## Citation

If you use SEPX in your research, please cite it using the following BibTeX entry:
'''
@misc{qiu2023sepx,
      title={Shortest Edit Path Crossover: A Theory-driven Solution to the Permutation Problem in Evolutionary Neural Architecture Search}, 
      author={Xin Qiu and Risto Miikkulainen},
      year={2023},
      eprint={2210.14016},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
'''
