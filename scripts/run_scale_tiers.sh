#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

TIERS_CSV="${TIERS_CSV:-3,10,20}"
BENCHMARK="${BENCHMARK:-wiki}"
EXECUTOR="${EXECUTOR:-direct_wiki}"
TASKS_PER_AGENT="${TASKS_PER_AGENT:-5}"
GLOBAL_TIMEOUT_SECONDS="${GLOBAL_TIMEOUT_SECONDS:-900}"
MAX_ITERATIONS="${MAX_ITERATIONS:-8}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-120}"
BASE_ROUTER_PORT="${BASE_ROUTER_PORT:-5655}"
BASE_SUB_PORT="${BASE_SUB_PORT:-5656}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/scale-tiers-$(date +%s)}"

mkdir -p "$OUTPUT_ROOT"

IFS=',' read -r -a TIERS <<< "$TIERS_CSV"

stop_workers() {
  if [ "$#" -eq 0 ]; then
    return 0
  fi

  for pid in "$@"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  for pid in "$@"; do
    wait "$pid" >/dev/null 2>&1 || true
  done
}

REPORT_PATHS=()

printf "Scale tiers run\n"
printf "  output: %s\n" "$OUTPUT_ROOT"
printf "  tiers: %s\n" "$TIERS_CSV"
printf "  benchmark: %s\n" "$BENCHMARK"
printf "  executor: %s\n" "$EXECUTOR"
printf "  tasks/agent: %s\n\n" "$TASKS_PER_AGENT"

for idx in "${!TIERS[@]}"; do
  workers="${TIERS[$idx]}"
  if ! [[ "$workers" =~ ^[0-9]+$ ]]; then
    echo "Invalid tier value: $workers" >&2
    exit 1
  fi

  router_port=$((BASE_ROUTER_PORT + idx * 10))
  sub_port=$((BASE_SUB_PORT + idx * 10))

  broker_router="tcp://127.0.0.1:${router_port}"
  broker_sub="tcp://127.0.0.1:${sub_port}"

  tier_dir="$OUTPUT_ROOT/tier-${workers}"
  logs_dir="$tier_dir/worker-logs"
  mkdir -p "$logs_dir"

  task_count=$((workers * TASKS_PER_AGENT))

  printf "[%s] starting %s workers on %s / %s (tasks=%s)\n" \
    "tier-${workers}" "$workers" "$broker_router" "$broker_sub" "$task_count"

  worker_pids=()
  for w in $(seq 1 "$workers"); do
    worker_id=$(printf "tier%02d-worker-%03d" "$workers" "$w")
    "$PYTHON_BIN" "$ROOT_DIR/runtime/worker_daemon.py" \
      --worker-id "$worker_id" \
      --broker-router "$broker_router" \
      --broker-sub "$broker_sub" \
      --default-executor "$EXECUTOR" \
      --work-root "$tier_dir/worker-local" \
      >"$logs_dir/${worker_id}.log" 2>&1 &
    worker_pids+=("$!")
  done

  bench_rc=0
  sleep 1
  if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/run_scale_benchmark.py" \
    --benchmark "$BENCHMARK" \
    --executor "$EXECUTOR" \
    --task-count "$task_count" \
    --start-broker \
    --broker-router "$broker_router" \
    --broker-sub "$broker_sub" \
    --output-dir "$tier_dir" \
    --global-timeout-seconds "$GLOBAL_TIMEOUT_SECONDS" \
    --max-iterations "$MAX_ITERATIONS" \
    --max-runtime-seconds "$MAX_RUNTIME_SECONDS" \
    | tee "$tier_dir/benchmark.log"; then
    bench_rc=1
  fi

  stop_workers "${worker_pids[@]}"

  report_path=$(find "$tier_dir" -type f -name 'scale_report.json' | head -n 1 || true)
  if [ -z "$report_path" ]; then
    echo "No scale_report.json found for tier ${workers}" >&2
    exit 1
  fi
  REPORT_PATHS+=("$report_path")

  if [ "$bench_rc" -ne 0 ]; then
    echo "Benchmark command failed for tier ${workers}" >&2
    exit 1
  fi

  printf "[%s] complete report=%s\n\n" "tier-${workers}" "$report_path"
done

summary_json_path="$OUTPUT_ROOT/tiers_summary.json"
summary_text_path="$OUTPUT_ROOT/tiers_summary.txt"

"$PYTHON_BIN" - "$summary_json_path" "$summary_text_path" "${REPORT_PATHS[@]}" <<'PY'
import json
import pathlib
import sys

summary_json_path = pathlib.Path(sys.argv[1])
summary_text_path = pathlib.Path(sys.argv[2])
report_paths = [pathlib.Path(p) for p in sys.argv[3:]]

rows = []
for report_path in report_paths:
    report = json.loads(report_path.read_text())
    tier_dir = report_path.parent.parent.name
    workers = int(tier_dir.split("-")[-1])
    rows.append(
        {
            "workers": workers,
            "task_count": report["task_count"],
            "completed": report["completed"],
            "success_count": report["success_count"],
            "failure_count": report["failure_count"],
            "missing_count": report["missing_count"],
            "throughput_tasks_per_min": report["throughput_tasks_per_min"],
            "p50_ms": report["latency_ms"]["p50"],
            "p95_ms": report["latency_ms"]["p95"],
            "report_path": str(report_path),
        }
    )

rows.sort(key=lambda x: x["workers"])

summary = {"tiers": rows}
summary_json_path.write_text(json.dumps(summary, indent=2))

lines = []
lines.append("workers\ttasks\tcompleted\tsuccess\tfailure\tmissing\tthroughput_tpm\tp50_ms\tp95_ms")
for row in rows:
    lines.append(
        f"{row['workers']}\t{row['task_count']}\t{row['completed']}\t{row['success_count']}\t"
        f"{row['failure_count']}\t{row['missing_count']}\t{row['throughput_tasks_per_min']:.2f}\t"
        f"{row['p50_ms']}\t{row['p95_ms']}"
    )
summary_text_path.write_text("\n".join(lines) + "\n")

print(summary_text_path.read_text(), end="")
print(f"\nJSON summary: {summary_json_path}")
PY

printf "\nDone. Artifacts under %s\n" "$OUTPUT_ROOT"
