#!/usr/bin/env bash
#SBATCH -A C3SE2016-1-11
#SBATCH -p glenn
#SBATCH -J A1_2-25_75
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -o A1_2-25_75.stdout
#SBATCH -e A1_2-25_75.stderr
module purge 
export PATH=/usr/lib64/qt-3.3/bin:/local/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/opt/thinlinc/bin:/c3se/users/lyaa/Glenn/bin
module load python gcc/4.8/4.8.1 
module load acml/gfortran64_fma4_mp/5.3.0
module load numpy/py27/1.8.1-gcc48-acml_mp scipy/py27/0.14.0-gcc48-acml
module load untested gurobi

pdcp *.py $TMPDIR
pdcp ../../trafficMatrix/traffic_matrix_2.csv $TMPDIR
pdcp ../../../sdm.py $TMPDIR
cd $TMPDIR

python A1_2.py

cp *.csv $SLURM_SUBMIT_DIR

# End script
