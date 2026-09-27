# Recreate the local demo state and start the configured CSV replay.
# The source dataset is bind-mounted and remains untouched.

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCommand) {
    $script:DemoDocker = $dockerCommand.Source
} else {
    $script:DemoDocker = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
    if (-not (Test-Path -LiteralPath $script:DemoDocker)) {
        throw 'Docker Desktop was not found. Install or start Docker Desktop and retry.'
    }
}

function Invoke-Compose {
    & $script:DemoDocker compose @args
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose $args failed with exit code $LASTEXITCODE"
    }
}

Invoke-Compose --profile '*' down -v
Invoke-Compose up --build --wait --wait-timeout 300
Invoke-Compose --profile replay up -d replay

Write-Output 'Demo replay started: http://localhost:5173/'
