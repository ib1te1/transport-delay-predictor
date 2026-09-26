# Starts the demo over: stops every service that holds run state, clears
# the database and Redis, and brings the stack back up. api and matcher
# keep state in memory, so clearing under them is not enough.
#
# Services from every profile are stopped, not only the default ones: a
# running replay or emulator would keep writing into the cleared tables.
# Postgres and Redis stay up; reset needs them. A stream source is not
# started again: which one to run next is a choice, printed at the end.
#
# ASCII only: Windows PowerShell 5.1 reads a file without a BOM as ANSI.

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

# A failing native command does not stop the script on its own.
function Invoke-Compose {
    & docker compose @args
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose $args failed with exit code $LASTEXITCODE"
    }
}

$running = @(Invoke-Compose --profile '*' ps --services --status running |
    Where-Object { $_ -and $_ -notin @('postgres', 'redis') })
if ($running.Count -gt 0) {
    Write-Output "Stopping: $($running -join ' ')"
    Invoke-Compose --profile '*' stop @running
}

Invoke-Compose --profile reset run --rm reset
Invoke-Compose up -d

Write-Output @'

Demo reset. Start a stream source:
  dataset replay:  docker compose --profile replay up -d replay
  NDTP emulator:   docker compose --profile emulator up -d emulator
'@
