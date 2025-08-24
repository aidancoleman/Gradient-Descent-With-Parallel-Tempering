#!/usr/bin/env bash
set -euo pipefail
# -e -> exit when we get error
# -u -> error when variables are unset
# -o -> pipefail makes pipelines fail for any stage fails

CSV=${CSV:-covtype.csv} #path to dataset
STAB_STRIDE=${STAB_STRIDE:-0}
TRACE_STRIDE=${TRACE_STRIDE:-0}
EXTRA_ARGS="--snap_stride 0 --seed ${SEED:-42} --balance_sgld 1"
if (( STAB_STRIDE > 0 ));  then EXTRA_ARGS+=" --stab_stride ${STAB_STRIDE}"; fi
if (( TRACE_STRIDE > 0 )); then EXTRA_ARGS+=" --trace_stride ${TRACE_STRIDE}"; fi
K=${K:-20000} #num. of training iters
S=${S:-0.4} #our target swap rate between replicas
ETA_LO=${ETA_LO:-3e-3} #the lowest learning rate for our cold chain that explores the true posterior distribution
ETA_HI=${ETA_HI:-2e-2} #the highest learning rate for our hottest, exploratory chain that explores the "flattest" version of posterior distribution
G0=${G0:-0.05} #algo adaptation step size
GDECAY=${GDECAY:-1e-4} #decay in gamma
VAL=${VAL:-0.2} #validation fraction -> how much of the dataset we hold out to test accuracy of model on, e.g. 0.3 => 30% of the data won't be used for training the model but we will use this to test accuracy in prediction. Larger val_frac means more epochs over a smaller training set for the same number of iterations K
SEED=${SEED:-42} #seed for RNG

# OpenMP binds
export OMP_PROC_BIND=close #keep threads of a rank close to each other
export OMP_PLACES=cores #pin to CPU cores
export OMP_DYNAMIC=false #don't let OpenMP change thread count dynamically!!

# weak scaling with fixed threads per rank, each rank has its own address space
# P = 6 so we use all CPUs available on Callan
FIXED_T=${FIXED_T:-6} #can be overriden e.g. in terminal use "FIXED_T=N"

#Logical CPUs on this node
TOTAL_CORES=$(nproc) #e.g. 48 on my node Callan to maximise compute resources

#Max ranks allowed without oversubscription (when we try to run more processes and threads than CPUs available on cluster) -> leads to cache thrashing and slower runtimes
MAXP=$(( TOTAL_CORES / FIXED_T ))
if (( MAXP < 1 )); then
  echo "Not enough CPUs for FIXED_T=${FIXED_T} (TOTAL_CORES=${TOTAL_CORES})"
  exit 1
fi

run_weak() {
  local P=$1
  local T=${FIXED_T}
#guard so we don't try use more than available CPUs
  if (( P * T > TOTAL_CORES )); then
    echo "Skipping P=${P}: need $((P*T)) CPUs but only ${TOTAL_CORES} available."
    return 0
  fi

  export OMP_NUM_THREADS=$T #how many threads each rank should spawn

  #Build mpirun "multi-app" spec with per rank CPU pinning
  # We launch P application contexts, each "-np 1" (one rank),
  # separated by ":"; each rank is wrapped in `taskset -c start-end`
  # to bind it to a unique, non overlapping CPU range of size T
  local spec=""
  local start=0
  for r in $(seq 1 $P); do
    local end=$(( start + T - 1 ))
    spec+=" -np 1 taskset -c ${start}-${end} ./train_pt_mpi \
      --csv ${CSV} --iters ${K} --S ${S} \
      --eta_lo ${ETA_LO} --eta_hi ${ETA_HI} \
      --gamma0 ${G0} --gdecay ${GDECAY} \
      --val ${VAL} ${EXTRA_ARGS} :"
    start=$(( end + 1 ))
  done
  spec=${spec% :} #get rid of the colon that is trailing

  echo "=== P=${P} (OMP_NUM_THREADS=${T}) ==="

  # True wall clock time for the whole mpirun
  local start_wall end_wall
  start_wall=$(date +%s)
  # shellcheck disable=SC2086 -> we want word splitting
  mpirun $spec | tee "weak_P${P}.log"
  end_wall=$(date +%s)
  echo "[wall] seconds=$(( end_wall - start_wall ))" | tee -a "weak_P${P}.log"
  #Aggregate per-replica telemetry into a CSV row
  {
    log="weak_P${P}.log"

    # Wall clock for the whole multi-app mpirun
    wall=$(grep '^\[wall\] seconds=' "$log" | tail -n1 | sed 's/.*seconds=//')

    # Reduce [timings] across replicas: take the worst (max) per category
    t_lines=$(grep '^\[timings\]' "$log" | tail -n ${P} || true)
    compute_max=$(echo "$t_lines" | sed -n 's/.*compute=\([0-9.]*\)s.*/\1/p' | sort -nr | head -1)
    energy_max=$(echo  "$t_lines" | sed -n 's/.*energy_comm=\([0-9.]*\)s.*/\1/p' | sort -nr | head -1)
    swap_max=$(echo    "$t_lines" | sed -n 's/.*swap_comm=\([0-9.]*\)s.*/\1/p' | sort -nr | head -1)
    allr_max=$(echo    "$t_lines" | sed -n 's/.*allreduce=\([0-9.]*\)s.*/\1/p' | sort -nr | head -1)

    # Reduce [stab] across replicas: mean of each metric
    # Replace any Unicode '≈' with '=' to simplify parsing
    stab_lines=$(grep '^\[stab\]' "$log" | tail -n ${P} | sed 's/≈/=/' || true)
    awkfile=$(mktemp)
    cat > "$awkfile" <<'AWK'
/^\[stab\]/ {
  for (i=1;i<=NF;i++){
    if ($i ~ /mean\|\|θ\|\|=/) { split($i,a,"="); theta_mean+=a[2]; n++ }
    if ($i ~ /max\|\|θ\|\|=/)  { split($i,a,"="); theta_max+=a[2] }
    if ($i ~ /mean_step=/)     { split($i,a,"="); step_mean+=a[2] }
    if ($i ~ /max_step=/)      { split($i,a,"="); step_max+=a[2] }
    if ($i ~ /mean\|∇\|=/)     { split($i,a,"="); grad_mean+=a[2] }
    if ($i ~ /max\|∇\|=/)      { split($i,a,"="); grad_max+=a[2] }
    if ($i ~ /mean_rel_gap=/)  { split($i,a,"="); relgap_mean+=a[2] }
    if ($i ~ /max_rel_gap=/)   { split($i,a,"="); relgap_max+=a[2] }
    if ($i ~ /growth=/)        { split($i,a,"="); growth+=a[2] }
  }
}
END {
  if (n>0) {
    printf("%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
      theta_mean/n, theta_max/n,
      step_mean/n,  step_max/n,
      grad_mean/n,  grad_max/n,
      relgap_mean/n,relgap_max/n,
      growth/n);
  }
}
AWK
    stab_csv=$(echo "$stab_lines" | awk -f "$awkfile")
    rm -f "$awkfile"
    # Write header once, then append the row for this P
    if [ ! -f weak_scaling.csv ]; then
      echo "P,T,wall_s,compute_max_s,energy_comm_max_s,swap_comm_max_s,allreduce_max_s,theta_mean,theta_max_mean,step_mean,step_max_mean,grad_mean,grad_max_mean,rel_gap_mean,rel_gap_max_mean,growth_mean" > weak_scaling.csv
    fi
    if [ -n "${stab_csv:-}" ]; then
      echo "${P},${T},${wall},${compute_max},${energy_max},${swap_max},${allr_max},${stab_csv}" >> weak_scaling.csv
    else
      echo "${P},${T},${wall},${compute_max},${energy_max},${swap_max},${allr_max}" >> weak_scaling.csv
    fi
  }
}

# Powers-of-two up to MAXP, then ensure a final full-CPU run at MAXP
P_LIST=(1 2 4 8 16 32 64)
last=0
for P in "${P_LIST[@]}"; do
  (( P > MAXP )) && break
  run_weak "$P"
  last=$P
done
if (( last != MAXP )); then
  run_weak "$MAXP"
fi
