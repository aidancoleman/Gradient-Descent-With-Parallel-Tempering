#!/usr/bin/env bash
set -euo pipefail

# Defaults

CSV=""
BIN=""
REPLICAS=8
K_TOTAL=24000
SWAP_STRIDE=100
PRINT_STRIDE=1000
ETA_LO=0.003
ETA_HI=0.02
VAL=0.2
SEED=42
S=0.4
GAMMA0=0.05
GDECAY=1e-4
LOGCSV="results_strongscale.csv"

# We allow preset OMP threads if user exported it; can be overridden by --threads
FIXED_THREADS="${OMP_NUM_THREADS:-}"

# Allow forcing MPI flavor (openmpi|intelmpi|mvapich2|mpich|slurm|auto) etc
MPI_FLAVOR="${MPI_FLAVOR:-auto}"

# Optional: user can set CORES or P_PER_NODE in env for multi-node layouts
CORES="${CORES:-}"
P_PER_NODE="${P_PER_NODE:-}"


# Parse CLI

while (($#)); do
  case "$1" in
    --csv) CSV="${2:?missing path after --csv}"; shift 2;;
    --bin) BIN="${2:?missing path after --bin}"; shift 2;;
    --replicas) REPLICAS="${2:?}"; shift 2;;
    --K_total) K_TOTAL="${2:?}"; shift 2;;
    --swap_stride) SWAP_STRIDE="${2:?}"; shift 2;;
    --print_stride) PRINT_STRIDE="${2:?}"; shift 2;;
    --eta_lo) ETA_LO="${2:?}"; shift 2;;
    --eta_hi) ETA_HI="${2:?}"; shift 2;;
    --val) VAL="${2:?}"; shift 2;;
    --seed) SEED="${2:?}"; shift 2;;
    --S) S="${2:?}"; shift 2;;
    --gamma0) GAMMA0="${2:?}"; shift 2;;
    --gdecay) GDECAY="${2:?}"; shift 2;;
    --log_csv) LOGCSV="${2:?}"; shift 2;;
    --threads) FIXED_THREADS="${2:?}"; shift 2;;  # fixed OMP threads per rank
    --) shift; break;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done


# Dataset validation

if [[ -n "$CSV" && -n "$BIN" ]]; then
  echo "Error: pass only one of --csv or --bin"; exit 1
fi
if [[ -n "$CSV" ]]; then
  [[ -f "$CSV" ]] || { echo "CSV not found: $CSV"; exit 1; }
  DATA_ARGS=(--csv "$CSV")
elif [[ -n "$BIN" ]]; then
  BIN=${BIN/#\~/$HOME}
  [[ -f "$BIN" ]] || { echo "BIN not found: $BIN"; exit 1; }
  DATA_ARGS=(--bin "$BIN")
else
  echo "Error: provide --csv <file> or --bin <file>"; exit 1
fi

# Build w our Makefile

make

# Helpers

detect_mpi_flavor() {
  # Prefer Slurm if active so we can queue jobs
  if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "slurm"; return
  fi
  if mpirun --version 2>/dev/null | head -n1 | grep -qi "Open MPI"; then
    echo "openmpi"; return
  fi
  if mpirun -help 2>&1 | grep -qi "Intel(R) MPI"; then
    echo "intelmpi"; return
  fi
  if mpirun --version 2>&1 | grep -qi "MVAPICH2"; then
    echo "mvapich2"; return
  fi
  if mpirun -help 2>&1 | grep -qi "HYDRA"; then
    echo "mpich"; return
  fi
  echo "unknown"
}
# no. available cores
get_cores() {
  if [[ -n "$CORES" ]]; then
    echo "$CORES"
  elif command -v lscpu >/dev/null 2>&1; then
    lscpu | awk '/^CPU\(s\):/ {print $2; exit}'
  else
    echo 8
  fi
}

if [[ "$MPI_FLAVOR" == "auto" ]]; then
  MPI_FLAVOR="$(detect_mpi_flavor)"
fi
echo "[launcher] Detected MPI flavor: $MPI_FLAVOR"

# OpenMP settings common to all launches
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_SCHEDULE=static
# Uncomment for diagnostics:
# export OMP_DISPLAY_ENV=true
# export OMP_DISPLAY_AFFINITY=true

# Compute T (threads per rank) if not fixed
compute_threads() {
  local p="$1"
  if [[ -n "$FIXED_THREADS" ]]; then
    echo "$FIXED_THREADS"
    return
  fi
  local cores
  cores="$(get_cores)"
  local ppn="${P_PER_NODE:-$p}"   # if single node, assume all ranks on one node
  local t=$(( cores / ppn ))
  (( t < 1 )) && t=1
  echo "$t"
}

launch_mpi() {
  local P="$1"; shift
  local exe_and_args=("$@")

  # Set OMP_NUM_THREADS for this launch
  local T
  T="$(compute_threads "$P")"
  export OMP_NUM_THREADS="$T"

  case "$MPI_FLAVOR" in
    slurm)
      echo "[launcher] Slurm: srun -n $P --cpus-per-task=$T --cpu-bind=cores ..."
      srun -n "$P" --cpus-per-task="$T" --cpu-bind=cores "${exe_and_args[@]}"
      ;;

    openmpi)
      # Bind each rank to PE=T cores and report bindings
      echo "[launcher] Open MPI: mpirun -np $P --bind-to core --map-by core:PE=$T --report-bindings ..."
      mpirun -np "$P" \
        --bind-to core --map-by core:PE="$T" --report-bindings \
        "${exe_and_args[@]}"
      ;;

    intelmpi)
      # Give each rank exactly OMP_NUM_THREADS cores
      echo "[launcher] Intel MPI: I_MPI_PIN=1, I_MPI_PIN_DOMAIN=core:${OMP_NUM_THREADS} (mapping printed with I_MPI_DEBUG=5)"
      export I_MPI_PIN=1
      export I_MPI_PIN_DOMAIN=core:${OMP_NUM_THREADS} # explicit N-core domain per rank
      export I_MPI_PIN_ORDER=scatter # spread ranks across cores/sockets
      export I_MPI_DEBUG=5
      mpirun -np "$P" "${exe_and_args[@]}"
      ;;

    mvapich2)
      echo "[launcher] MVAPICH2: enabling core-level affinity"
      export MV2_ENABLE_AFFINITY=1
      export MV2_CPU_BINDING_POLICY=scatter
      export MV2_CPU_BINDING_LEVEL=core
      export MV2_SHOW_CPU_BINDING=1
      mpirun -np "$P" "${exe_and_args[@]}"
      ;;

    mpich)
      echo "[launcher] MPICH/Hydra detected; for robust binding prefer Slurm. Launching without explicit core mapping."
      mpirun -np "$P" "${exe_and_args[@]}"
      ;;

    *)
      echo "[launcher] Unknown MPI flavor; launching without explicit rank binding."
      mpirun -np "$P" "${exe_and_args[@]}"
      ;;
  esac
}


# Strong-scaling sweep w/ fixed replicas and growing MPI ranks

for P in 1 2 4 8; do
  echo "Running with $P MPI ranks..."
  launch_mpi "$P" \
    ./train_strongscale \
      "${DATA_ARGS[@]}" \
      --replicas "$REPLICAS" \
      --K_total "$K_TOTAL" \
      --swap_stride "$SWAP_STRIDE" \
      --print_stride "$PRINT_STRIDE" \
      --eta_lo "$ETA_LO" \
      --eta_hi "$ETA_HI" \
      --val "$VAL" \
      --seed "$SEED" \
      --S "$S" \
      --gamma0 "$GAMMA0" \
      --gdecay "$GDECAY" \
      --log_csv "$LOGCSV"
done

echo "Done. Results appended to $LOGCSV"
