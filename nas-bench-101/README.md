Please install ```NAS-bench-101``` from: https://github.com/google-research/nasbench.
Before running the scripts, please create a folder named ```Results``` so that experimental results can be automatically saved in it.

The file ```RE_crossover_edit_path_center_NoIsomorphism_generation_multiprocessing_statistics_mean_test_icml.py```
is for generating the results of mutation, standard crossover and SEP crossover in noisy environments on nas-bench-101. 
The bash script file
```icml_multiple_mean_test.sh``` can used to parallelize the running of this experiment.
The file ```RE_crossover_edit_path_center_GEDfitness_NoIsomorphism_generation_multiprocessing_statistics_theory.py```
is for generating the results of mutation, standard crossover and SEP crossover in noise-free environments on nas-bench-101.

For running RL experiments, first install: https://github.com/automl/nas_benchmarks, then run file ```run_rl_record_logit.py``` with command
```
python run_rl_record_logit.py --benchmark nas_cifar10a --n_iters 10000 --n_runs 50 --lr 0.5
``` 
to obtain the results for noisy environment.
For obtaining RL results in noise-free environments, replace ```tabular_benchmarks/nas_cifar10.py``` inside ```nas_benchmarks``` with
the attached nas_cifar10.py, then run file ```run_rl_GED_to_best_record_logit.py``` with command 
```
python run_rl_GED_to_best_record_logit.py --benchmark nas_cifar10a --n_iters 10000 --n_runs 50 --lr 0.1
``` 
to obtain the results for noise-free environment.

The file ```ged-to-optimal_heatmap_empirical_nas101_icml.py``` is for generating the results used for Section "Applicability of the Theory".
The file ```plot_crossover_GED_to_optimal_edit_path_center_NoIsomorphism_statistics_icml.py``` is for generating the plots in noise-free environment.
The file ```plot_crossover_edit_path_center_NoIsomorphism_statistics_mean_test_icml.py``` is for generating the plots regarding mean test accuracy in noisy environment.
The file ```plot_crossover_edit_path_center_NoIsomorphism_statistics_optimal_ratio_icml.py``` is for generating the plots regarding ratio to reach optimal solution in noisy environment.
The file ```plot_crossover_edit_path_center_NoIsomorphism_statistics_val_icml.py``` is for generating the plots regarding validation accuracy in noisy environment.
