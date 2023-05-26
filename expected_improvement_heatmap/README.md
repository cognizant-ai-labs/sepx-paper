Scripts in this folder generate heatmap figures for comparing expected improvement of different approaches.

## How to run
The file name for each ```.py``` indicates which two methods are compared on which dataset. More specifically:
- ```SEPX``` stands for the SEP crossover operator.
- ```mutation``` stands for the mutation operator.
- ```STDX``` stands for the standard crossover operator.
- ```RL``` stands for the RL methods (including both the unbiased agent and oracle agent).
- ```nas101``` stands for NAS-Bench-101 dataset.
- ```nasnlp``` stands for NAS-Bench-NLP dataset.

Example: ```expected_improvement_heatmap_SEPX_vs_mutation_nas101.py``` generates the heatmap that compares the expected improvement of the SEP crossover and mutation in NAS-Bench-101 space.
