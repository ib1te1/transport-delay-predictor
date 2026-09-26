#!/bin/sh
# Starts the demo over: stops every service that holds run state, clears
# the database and Redis, and brings the stack back up. api and matcher
# keep state in memory, so clearing under them is not enough.
#
# Services from every profile are stopped, not only the default ones: a
# running replay or emulator would keep writing into the cleared tables.
# Postgres and Redis stay up; reset needs them. A stream source is not
# started again: which one to run next is a choice, printed at the end.
set -eu
cd "$(dirname "$0")/.."

running=$(docker compose --profile '*' ps --services --status running | grep -vx -e postgres -e redis || true)
if [ -n "$running" ]; then
  # shellcheck disable=SC2086 # one service name per word
  echo "Stopping:" $running
  # shellcheck disable=SC2086
  docker compose --profile '*' stop $running
fi

docker compose --profile reset run --rm reset
docker compose up -d

cat <<'EOF'

Demo reset. Start a stream source:
  dataset replay:  docker compose --profile replay up -d replay
  NDTP emulator:   docker compose --profile emulator up -d emulator
EOF
