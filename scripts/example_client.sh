#!/bin/bash

# Script run at launching on each worker:

# Install libraries 

cd /async-dp/decentralizepy 
pip3 install --no-cache-dir --editable .[dev]
cd /async-dp
pip3 install --no-cache-dir --editable .

# Parameters
com_factor=$(cat /docker_logs/com_factor.txt)
nb_stragg=$(cat /docker_logs/nb_stragg.txt)
chunking_fraction=$(cat /docker_logs/chunking_fraction.txt)
chunking_name=$(echo $chunking_fraction | sed 's/\.//')
seed=$(cat /docker_logs/seed.txt)
degree=$(cat /docker_logs/degree.txt)

cd /

# Collect ip addresses of all workers and distribute them to all workers
host fastWorker-$KOLLAPS_UUID | grep -E -o "([0-9]{1,3}[\.]){3}[0-9]{1,3}" | sort -u | tee fastips.txt
# For i in range(0, nb_stragg (excluded)), we need to get the ip addresses of the slow workers
for i in $(seq 0 $(($nb_stragg-1))); do
  host slowWorker$i-$KOLLAPS_UUID | grep -E -o "([0-9]{1,3}[\.]){3}[0-9]{1,3}" | sort -u | tee -a slowips.txt
done
selfip=$(ifconfig eth0 | awk '/inet / {print $2}')
echo $selfip | tee /selfip.txt
python3 /scripts/pre_runner.py
mkdir -p /results

# Run experience
cd /async-dp/tutorial/
./example_run.sh

# Compute test results
python3 /async-dp/read_and_test/read_and_test.py /results/${degree}d_${seed}sd_${nb_stragg}ns_${com_factor}cf_${chunking_name}cf

# Delete stored models
rm -rf /results/${degree}d_${seed}sd_${nb_stragg}ns_${com_factor}cf_${chunking_name}cf/models

# Copy results to the master TO COMPLETE WITH YOUR MOUNTED DIRECTORY
machine=$(cat /identity.txt)
cd /results
cp -r ${degree}d_${seed}sd_${nb_stragg}ns_${com_factor}cf_${chunking_name}cf /TOCOMPLETE/$machine/