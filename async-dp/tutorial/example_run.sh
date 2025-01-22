#!/bin/bash

graph=/async-dp/tutorial/example_fullyConnected_60.edges
config_file=/async-dp/tutorial/example_config.ini
run_path=/results # Path to the folder where the graph and config file will be copied and the results will be stored

# If run_path does not exist, create it
if [ ! -d "$run_path" ]; then
    mkdir -p $run_path
fi

env_python=python3 
machines=60 # number of machines/containers in the runtime
iterations=1000 # number of iterations
test_after=10
eval_file=/async-dp/tutorial/example_testingDivShare.py # decentralized driver code (run on each machine)
log_level=INFO # DEBUG | INFO | WARN | CRITICAL
procs_per_machine=1

# Id is in /identity.txt
machine=$(cat /identity.txt)
# Read parameters of the experiment
com_factor=$(cat /docker_logs/com_factor.txt)
nb_stragg=$(cat /docker_logs/nb_stragg.txt)
chunking_fraction=$(cat /docker_logs/chunking_fraction.txt)
chunking_name=$(echo $chunking_fraction | sed 's/\.//')
seed=$(cat /docker_logs/seed.txt)
degree=$(cat /docker_logs/degree.txt)

log_dir=$run_path/${degree}d_${seed}sd_${nb_stragg}ns_${com_factor}cf_${chunking_name}cf

$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $machine -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level -wsd $log_dir -dataset_seed $seed -chunk_fraction $chunking_fraction