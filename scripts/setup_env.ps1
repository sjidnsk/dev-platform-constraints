param(
    [string]$Conda = "conda",
    [string]$EnvName = "dev-platform-constraints",
    [switch]$RunValidation,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$EnvironmentFile = Join-Path $RepoRoot "environment.yml"
$PythonSpec = "python=3.12"
$PythonVersionCheck = "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.version)"

function Invoke-SetupStep {
    param(
        [string]$Display,
        [scriptblock]$Action
    )

    if ($DryRun) {
        Write-Host "[DRY RUN] $Display"
        return
    }

    Write-Host "==> $Display"
    & $Action
}

function Invoke-NativeCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

function Test-CondaEnv {
    param([string]$Name)

    & $Conda run -n $Name python --version *> $null
    return $LASTEXITCODE -eq 0
}

Write-Host "Repository: $RepoRoot"
Write-Host "Conda environment: $EnvName"
Write-Host "Environment file: $EnvironmentFile"

Invoke-SetupStep `
    -Display "$Conda env create/update -n `"$EnvName`" -f `"$EnvironmentFile`"" `
    -Action {
        if (Test-CondaEnv -Name $EnvName) {
            Invoke-NativeCommand -Executable $Conda -Arguments @("env", "update", "-n", $EnvName, "-f", $EnvironmentFile, "--prune")
        } else {
            Invoke-NativeCommand -Executable $Conda -Arguments @("env", "create", "-n", $EnvName, "-f", $EnvironmentFile)
        }
    }

Invoke-SetupStep `
    -Display "$Conda install -n `"$EnvName`" -c conda-forge $PythonSpec --yes" `
    -Action { Invoke-NativeCommand -Executable $Conda -Arguments @("install", "-n", $EnvName, "-c", "conda-forge", $PythonSpec, "--yes") }

Invoke-SetupStep `
    -Display "$Conda run -n `"$EnvName`" python -c `"$PythonVersionCheck`"" `
    -Action { Invoke-NativeCommand -Executable $Conda -Arguments @("run", "-n", $EnvName, "python", "-c", $PythonVersionCheck) }

Invoke-SetupStep `
    -Display "$Conda run -n `"$EnvName`" python -m pip install -e `"$RepoRoot`"" `
    -Action { Invoke-NativeCommand -Executable $Conda -Arguments @("run", "-n", $EnvName, "python", "-m", "pip", "install", "-e", $RepoRoot) }

if ($RunValidation) {
    Push-Location $RepoRoot
    try {
        Invoke-SetupStep `
            -Display "$Conda run -n `"$EnvName`" python -m unittest discover -s tests" `
            -Action { Invoke-NativeCommand -Executable $Conda -Arguments @("run", "-n", $EnvName, "python", "-m", "unittest", "discover", "-s", "tests") }

        Invoke-SetupStep `
            -Display "$Conda run -n `"$EnvName`" python scripts\run_minimal_closure.py" `
            -Action { Invoke-NativeCommand -Executable $Conda -Arguments @("run", "-n", $EnvName, "python", (Join-Path $RepoRoot "scripts\run_minimal_closure.py")) }
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Next commands:"
Write-Host "  conda activate $EnvName"
Write-Host "  python -m unittest discover -s tests"
Write-Host "  python scripts\run_minimal_closure.py"
