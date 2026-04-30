#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LAUNCHER_CONFIG="${MULTI_CAMERA_CONFIG:-configs/multi_camera_launcher.yaml}"

usage() {
  cat >&2 <<'EOF2'
Usage:
  ./run_multi_camera.sh --list
  ./run_multi_camera.sh preset <preset-name> [extra app args...]
  ./run_multi_camera.sh <mode-a> <mode-b> [extra app args...]

Examples:
  ./run_multi_camera.sh preset web-ipad
  ./run_multi_camera.sh webcam-visual ipad
  ./run_multi_camera.sh webcam-safe iphone --show-window true

Notes:
  - available modes/presets come from configs/multi_camera_launcher.yaml
  - presets are tuned for two simultaneous windows and keep the original single-camera profiles untouched
  - extra args after the mode/preset are passed to every launched app process
  - Ctrl+C stops all child camera processes started by this launcher
EOF2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$LAUNCHER_CONFIG" ]]; then
  echo "Launcher config not found: $LAUNCHER_CONFIG" >&2
  exit 1
fi

RESOLVED=$(
python3 - "$LAUNCHER_CONFIG" "$@" <<'PY'
import json
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
args = sys.argv[2:]
if not args:
    print(json.dumps({"action": "error", "error": "missing arguments"}))
    sys.exit(0)

cfg = yaml.safe_load(cfg_path.read_text()) or {}
defaults = cfg.get("defaults", {}) or {}
modes_cfg = cfg.get("modes", {}) or {}
presets_cfg = cfg.get("presets", {}) or {}

aliases = {}
for canonical, payload in modes_cfg.items():
    aliases[canonical] = canonical
    for alias in payload.get("aliases", []) or []:
        aliases[alias] = canonical

if args[0] == "--list":
    print(json.dumps({
        "action": "list",
        "defaults": defaults,
        "modes": modes_cfg,
        "presets": presets_cfg,
    }))
    sys.exit(0)

extra_args = []
selected = []
if args[0] == "preset":
    if len(args) < 2:
        print(json.dumps({"action": "error", "error": "preset name is required"}))
        sys.exit(0)
    preset_name = args[1]
    preset_modes = presets_cfg.get(preset_name)
    if not preset_modes:
        print(json.dumps({"action": "error", "error": f"unknown preset: {preset_name}"}))
        sys.exit(0)
    selected = list(preset_modes)
    extra_args = args[2:]
else:
    if len(args) < 2:
        print(json.dumps({"action": "error", "error": "two mode names are required"}))
        sys.exit(0)
    selected = [args[0], args[1]]
    extra_args = args[2:]

resolved = []
for raw_name in selected:
    canonical = aliases.get(raw_name)
    if canonical is None:
        print(json.dumps({"action": "error", "error": f"unknown mode: {raw_name}"}))
        sys.exit(0)
    payload = modes_cfg.get(canonical) or {}
    config_path = payload.get("config")
    if not config_path:
        print(json.dumps({"action": "error", "error": f"mode {canonical} has no config path"}))
        sys.exit(0)
    resolved.append({
        "name": canonical,
        "config": config_path,
        "cpu_affinity": str(payload.get("cpu_affinity", "")),
    })

print(json.dumps({
    "action": "run",
    "defaults": defaults,
    "resolved": resolved,
    "extra_args": extra_args,
}))
PY
)

ACTION=$(python3 - <<'PY' "$RESOLVED"
import json, sys
print(json.loads(sys.argv[1]).get("action", ""))
PY
)

if [[ -z "$ACTION" ]]; then
  echo "Failed to parse launcher config response" >&2
  exit 1
fi

if [[ "$ACTION" == "list" ]]; then
  python3 - <<'PY' "$RESOLVED"
import json, sys
payload = json.loads(sys.argv[1])
print("Modes:")
for name, mode in payload["modes"].items():
    aliases = ", ".join(mode.get("aliases", [])) or "-"
    affinity = mode.get("cpu_affinity") or "-"
    print(f"  - {name}: {mode.get('config')} (aliases: {aliases}; cpu: {affinity})")
print("\nPresets:")
for name, members in payload["presets"].items():
    print(f"  - {name}: {' + '.join(members)}")
print("\nDefaults:")
for key, value in payload["defaults"].items():
    print(f"  - {key}: {value}")
PY
  exit 0
fi

ERROR=$(python3 - <<'PY' "$RESOLVED"
import json, sys
print(json.loads(sys.argv[1]).get("error", ""))
PY
)
if [[ -n "$ERROR" ]]; then
  echo "$ERROR" >&2
  usage
  exit 2
fi

readarray -t RESOLVED_LINES < <(python3 - <<'PY' "$RESOLVED"
import json, sys
payload = json.loads(sys.argv[1])
defaults = payload["defaults"]
for item in payload["resolved"]:
    print(f"NAME={item['name']}")
    print(f"CONFIG={item['config']}")
    print(f"CPU_AFFINITY={item.get('cpu_affinity', '')}")
print("--EXTRA--")
for arg in payload["extra_args"]:
    print(arg)
print("--DEFAULTS--")
for key in ["device", "stagger_seconds", "python_bin", "app_module", "log_dir"]:
    print(f"{key}={defaults.get(key, '')}")
PY
)

NAMES=()
CONFIGS=()
CPU_AFFINITIES=()
EXTRA_ARGS=()
DEVICE="cpu"
STAGGER_SECONDS="1.0"
PYTHON_BIN=".venv/bin/python"
APP_MODULE="app.main"
LOG_DIR="logs/multi-camera"
SECTION="modes"

for line in "${RESOLVED_LINES[@]}"; do
  if [[ "$line" == "--EXTRA--" ]]; then
    SECTION="extra"
    continue
  fi
  if [[ "$line" == "--DEFAULTS--" ]]; then
    SECTION="defaults"
    continue
  fi
  case "$SECTION" in
    modes)
      if [[ "$line" == NAME=* ]]; then
        NAMES+=("${line#NAME=}")
      elif [[ "$line" == CONFIG=* ]]; then
        CONFIGS+=("${line#CONFIG=}")
      elif [[ "$line" == CPU_AFFINITY=* ]]; then
        CPU_AFFINITIES+=("${line#CPU_AFFINITY=}")
      fi
      ;;
    extra)
      EXTRA_ARGS+=("$line")
      ;;
    defaults)
      key="${line%%=*}"
      value="${line#*=}"
      case "$key" in
        device) DEVICE="$value" ;;
        stagger_seconds) STAGGER_SECONDS="$value" ;;
        python_bin) PYTHON_BIN="$value" ;;
        app_module) APP_MODULE="$value" ;;
        log_dir) LOG_DIR="$value" ;;
      esac
      ;;
  esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIDS=()

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo
    echo "Stopping camera processes..."
    for pid in "${PIDS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

HAS_TASKSET=0
if command -v taskset >/dev/null 2>&1; then
  HAS_TASKSET=1
fi

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  config="${CONFIGS[$i]}"
  logfile="$LOG_DIR/${TIMESTAMP}_${name}.log"
  affinity="${CPU_AFFINITIES[$i]:-}"
  cmd=("$PYTHON_BIN" -m "$APP_MODULE" --config "$config" --device "$DEVICE")
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi
  launch_cmd=()
  if [[ -n "$affinity" && "$HAS_TASKSET" -eq 1 ]]; then
    launch_cmd=(taskset -c "$affinity")
  fi
  launch_cmd+=(stdbuf -oL -eL)
  launch_cmd+=("${cmd[@]}")

  echo "Launching [$name]"
  echo "  config: $config"
  echo "  log:    $logfile"
  if [[ -n "$affinity" && "$HAS_TASKSET" -eq 1 ]]; then
    echo "  cpu:    $affinity"
  fi
  printf '  cmd:    '
  printf '%q ' "${launch_cmd[@]}"
  echo

  "${launch_cmd[@]}" >>"$logfile" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "  pid:    $pid"

  if [[ "$i" -lt $((${#NAMES[@]} - 1)) ]]; then
    sleep "$STAGGER_SECONDS"
  fi
done

echo
printf 'Running: '
printf '%s ' "${PIDS[@]}"
echo
wait "${PIDS[@]}"
