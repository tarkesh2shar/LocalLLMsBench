#!/bin/bash
# Independent memory watchdog. Runs outside the harness so it still acts while the
# harness is blocked inside a long prefill HTTP call.
#
# Kills mlx_lm.server (NOT the harness) when memory gets dangerous. The harness sees
# the request fail, records it, and moves on cleanly.

LOG="${WATCHDOG_LOG:-$(cd "$(dirname "$0")/.." && pwd)/results/watchdog.log}"

MIN_AVAIL_GIB=3      # kill if reclaimable memory drops below this
MAX_SWAP_GIB=6       # kill if swap grows past this (thrashing precedes a hang)
MAX_RSS_GIB=30       # kill if the server alone exceeds this

echo "$(date '+%H:%M:%S') watchdog armed: avail<${MIN_AVAIL_GIB} swap>${MAX_SWAP_GIB} rss>${MAX_RSS_GIB}" >> "$LOG"

while true; do
  PID=$(pgrep -f "mlx_lm.server" | head -1)

  if [ -n "$PID" ]; then
    PGSZ=$(vm_stat | head -1 | grep -oE '[0-9]+')
    read FREE SPEC INACT PURGE < <(vm_stat | awk -v p="$PGSZ" '
      /Pages free/{f=$3} /Pages speculative/{s=$3} /Pages inactive/{i=$3} /Pages purgeable/{u=$3}
      END{gsub(/\./,"",f);gsub(/\./,"",s);gsub(/\./,"",i);gsub(/\./,"",u);
          print f*p/1073741824, s*p/1073741824, i*p/1073741824, u*p/1073741824}')
    AVAIL=$(echo "$FREE $SPEC $INACT" | awk '{printf "%.2f", $1+$2+$3}')
    SWAP=$(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M' | grep -oE '[0-9.]+' \
           | awk '{printf "%.2f", $1/1024}')
    RSS=$(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.2f", $1/1048576}')
    [ -z "$RSS" ] && RSS=0

    KILL=""
    awk "BEGIN{exit !($AVAIL < $MIN_AVAIL_GIB)}" && KILL="low memory (avail ${AVAIL} GiB)"
    awk "BEGIN{exit !($SWAP > $MAX_SWAP_GIB)}"  && KILL="swap thrash (${SWAP} GiB)"
    awk "BEGIN{exit !($RSS > $MAX_RSS_GIB)}"    && KILL="server rss ${RSS} GiB"

    if [ -n "$KILL" ]; then
      echo "$(date '+%H:%M:%S') TRIP: $KILL -> killing mlx_lm.server pid $PID" >> "$LOG"
      kill -TERM "$PID" 2>/dev/null
      sleep 5
      kill -KILL "$PID" 2>/dev/null
      sleep 20
    else
      echo "$(date '+%H:%M:%S') ok avail=${AVAIL} swap=${SWAP} rss=${RSS}" >> "$LOG"
    fi
  fi
  sleep 3
done
