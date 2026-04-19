Set-StrictMode -Version Latest

function check-redis-e2e {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $false)]
        [string]$RedisUrl
    )

    $repoRoot = Split-Path -Parent $PSScriptRoot
    $pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $checkScript = Join-Path $repoRoot "scripts\redis_e2e_check.py"

    if (-not (Test-Path $pythonExe)) {
        throw "Python executable not found: $pythonExe"
    }
    if (-not (Test-Path $checkScript)) {
        throw "Redis E2E script not found: $checkScript"
    }

    $args = @($checkScript)
    if ($RedisUrl) {
        $args += @("--redis-url", $RedisUrl)
    }

    $previousEncoding = $env:PYTHONIOENCODING
    $env:PYTHONIOENCODING = "utf-8"
    try {
        & $pythonExe @args
        return $LASTEXITCODE
    }
    finally {
        if ($null -eq $previousEncoding) {
            Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
        }
        else {
            $env:PYTHONIOENCODING = $previousEncoding
        }
    }
}

Set-Alias -Name cr2e -Value check-redis-e2e -Scope Global
