#!/bin/bash
# Concurrency scaling probe: does the server batch, and how well does it scale?
#
#   ./bench_concurrency.sh <model-id> [port]
#
# Serialized  -> per-request times stagger (1x, 2x, 3x, 4x) and throughput stays ~1.00x
# Batched     -> all requests finish within ms of each other; throughput > 1
#
# Server must already be running, e.g.:
#   MLX_MPI_LIBNAME=/opt/homebrew/opt/open-mpi/lib/libmpi.dylib \
#   mlx_lm.server --model <id> --port 8081 --prefill-step-size 512
# (without MLX_MPI_LIBNAME, MLX finds Anaconda's MPICH and exits 1 with no traceback)

MODEL="${1:?usage: bench_concurrency.sh <model-id> [port]}"
PORT="${2:-8081}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"
P='Count from 1 to 120, separated by single spaces. Output only the numbers.'
D="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":300,\"temperature\":0}"

req() { curl -s -o /dev/null -w "%{time_total}" -m 900 "$URL" -H 'Content-Type: application/json' -d "$D"; }
now() { python3 -c 'import time;print("%.3f"%time.time())'; }

echo "model: $MODEL  port: $PORT"
curl -s -o /dev/null -m 900 "$URL" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}"

echo "N  wall(s)  per-req(s)   throughput  efficiency"
BASE=""
for N in 1 2 4 8; do
  T0=$(now)
  for i in $(seq 1 $N); do req > "/tmp/bc_$i" & done
  wait
  T1=$(now)
  SPREAD=$(for i in $(seq 1 $N); do cat "/tmp/bc_$i"; echo; done | sort -n \
           | awk 'NR==1{min=$1} END{printf "%.1f-%.1f", min, $1}')
  [ -z "$BASE" ] && BASE=$(python3 -c "print($T1-$T0)")
  python3 -c "
w=$T1-$T0; b=$BASE; n=$N; t=n*b/w
print('%-2d %-8.2f %-12s %.2fx       %d%%' % (n, w, '$SPREAD', t, round(100*t/n)))"
  rm -f /tmp/bc_*
done
