#! /bin/bash
#COBALT -A CSC249ADCD08 --attrs enable_ssh=1

# Load up the Python environment
module load miniconda-3/latest
conda activate ../../env

# Add NWChem and Psi4 to the path
export PATH=~/software/psi4/bin:$PATH
export OMP_NUM_THREADS=64
export KMP_INIT_AT_FORK=FALSE

export PATH="/lus/theta-fs0/projects/CSC249ADCD08/software/nwchem-6.8.1/bin/LINUX64:$PATH"
mkdir -p scratch  # For the NWChem tasks
which nwchem
hostname
module load atp
export MPICH_GNI_MAX_EAGER_MSG_SIZE=16384
export MPICH_GNI_MAX_VSHORT_MSG_SIZE=10000
export MPICH_GNI_MAX_EAGER_MSG_SIZE=131072
export MPICH_GNI_NUM_BUFS=300
export MPICH_GNI_NDREG_MAXSIZE=16777216
export MPICH_GNI_MBOX_PLACEMENT=nic
export MPICH_GNI_LMT_PATH=disabled
export COMEX_MAX_NB_OUTSTANDING=6
export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2020.0.166/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH

# Copy over the database files
db_path=/tmp/edw-db-$RANDOM
cp -r /lus/theta-fs0/projects/CSC249ADCD08/edw/ai-components/sc-2021/db $db_path
chmod -R u+rwX $db_path
mongoport=271${RANDOM::2}
mongod --dbpath $db_path --port $mongoport &> mongo.out &
mongo=$!

# Start the redis-server
port=631${RANDOM::2}
redis-server --port $port &> redis.out &
redis=$!

# Run!
./nwc-run.sh --redisport $port --mongoport $mongoport $@

# Kill the servers
kill $redis
kill $mongo
rm -r $db_path
