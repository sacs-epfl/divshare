# CONFIG 

n=60
timeout_time=1800
mounted_folder="TOCOMPLETE"

# Function to remove all Docker stacks
remove_all_stacks() {
  # Get a list of all stack names
  stacks=$(docker stack ls --format "{{.Name}}")

  # Check if there are any stacks
  if [ -n "$stacks" ]; then
    # Loop through each stack and remove it
    for stack in $stacks; do
      docker stack rm "$stack"
    done
    echo "All stacks have been removed."
  else
    echo "No stacks to remove."
  fi
}

# Function to check if there are any running Docker containers
check_running_containers() {
  # Check if there are any running containers
  if [ -z "$(docker ps -q)" ]; then
    return 1 # No running containers
  else
    return 0 # There are running containers
  fi
}

# Loop over parameters
for seed in TOCOMPLETE ; do # Set the seed for reproducing the experiments
  for nb_stragglers in TOCOMPLETE (eg. 30 stragglers for 60 nodes) ; do
    for com_factor in TOCOMPLETE ; do
      for chunking_fraction in TO COMPLETE (eg. 0.1 0.5 1.0) ; do 
        for degree in 3 6 9 12 ; do 
          chunking_name=$(echo $chunking_fraction | sed 's/\.//')

          echo "Running experiment with $n nodes, $nb_stragglers stragglers, $com_factor communication factor, seed $seed, chunking fraction $chunking_fraction, degree $degree"
          # Append in config files
          echo "$com_factor" > ${mounted_folder}/com_factor.txt
          echo "$seed" > ${mounted_folder}/seed.txt
          echo "$nb_stragglers" > ${mounted_folder}/nb_stragg.txt
          echo "$chunking_fraction" > ${mounted_folder}/chunking_fraction.txt
          echo "$degree" > ${mounted_folder}/degree.txt

          # Deploy the stack
          docker stack deploy -c PATH_TO_TOPOLOGY.yaml NAME_OF_EXPERIMENT
          # Wait for the deployment to finish
          echo "Waiting for the deployment to finish"
          sleep 60
          # Start workers using Kollaps dashboard
          start_time=$(date +%s)
          echo "Starting workers"
          curl CLUSTER_URL/start
          # Wait for the experiment to finish
          number_of_logs_done=0
          crashed_machine=0
          elapsed_time=0
          # Loop until all logs are done or if crashed_machine is greater 5 (meaning that the experiment has crashed) or if the timeout has been reached
          while [ $number_of_logs_done -lt $n ] && [ $crashed_machine -lt 5 ] && [ $elapsed_time -lt $timeout_time ]; do

            #Compute number of logs done
            number_of_logs_done=0
            for i in $(seq 0 $n); do
              #Check if the directory exists
              if [ -d "${mounted_folder}/$i/${degree}d_${seed}sd_${nb_stragglers}ns_${com_factor}cf_${chunking_name}cf" ]; then
                number_of_logs_done=$((number_of_logs_done +1))
              fi
            done
            echo "Number of logs done: $number_of_logs_done"

            # If number_of_logs_done is between 1 and n/4 (inclusive), increase crashed_machine
            if [ $number_of_logs_done -ge 1 ] && [ $number_of_logs_done -le $(($n * 3 / 4)) ] ; then
              crashed_machine=$((crashed_machine + 1))
              echo "Crashed machine: $crashed_machine"
            fi

            # Compute elapsed time
            elapsed_time=$(($(date +%s) - $start_time))

            sleep 60
          done
          # Remove
          remove_all_stacks
          # Wait for the removal to finish
          # Loop until there are no running containers
          while check_running_containers; do
          echo "There are still running containers."
          sleep 30 # Wait before checking again
          remove_all_stacks
          done
          sleep 60 # Wait after all containers have stopped in case of any delays on the other machine

          # If the experiment has crashed, register it in the logs, append the parameters to a file 
          if [ $crashed_machine -ge 5 ] || [ $elapsed_time -ge $timeout_time ]; then
            echo "Experiment has crashed"
            echo "$n $seed $nb_stragglers $com_factor $chunking_fraction $degree" >> ${mounted_folder}/crashed_experiments.txt
          fi

          echo "Experiment done !"
        done
      done
    done
  done
done
