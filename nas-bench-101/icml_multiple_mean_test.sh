#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for run in {0..50}
do
	python RE_crossover_edit_path_center_NoIsomorphism_generation_multiprocessing_statistics_mean_test_icml.py --run_id "$run" &
done
