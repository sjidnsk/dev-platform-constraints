param(
    [string]$Conda = "conda",
    [string]$EnvPath = "D:\conda_envs\lunar-explorer",
    [switch]$RunValidation,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$EnvironmentFile = Join-Path $RepoRoot "environment.yml"
$SourcePath = Join-Path $RepoRoot "src"
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

function Invoke-WithProjectPythonPath {
    param([scriptblock]$Action)

    $PreviousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $SourcePath, "Process")
    try {
        & $Action
    } finally {
        [Environment]::SetEnvironmentVariable("PYTHONPATH", $PreviousPythonPath, "Process")
    }
}

function Test-CondaEnv {
    & $Conda run -p $EnvPath python --version *> $null
    return $LASTEXITCODE -eq 0
}

Write-Host "Repository: $RepoRoot"
Write-Host "Conda environment: $EnvPath"
Write-Host "Environment file: $EnvironmentFile"
Write-Host "Project PYTHONPATH: $SourcePath"

Invoke-SetupStep `
    -Display "$Conda env create/update -p `"$EnvPath`" -f `"$EnvironmentFile`"" `
    -Action {
        if (Test-CondaEnv) {
            Invoke-NativeCommand -Executable $Conda -Arguments @("env", "update", "-p", $EnvPath, "-f", $EnvironmentFile, "--prune")
        } else {
            Invoke-NativeCommand -Executable $Conda -Arguments @("env", "create", "-p", $EnvPath, "-f", $EnvironmentFile)
        }
    }

Invoke-SetupStep `
    -Display "$Conda install -p `"$EnvPath`" -c conda-forge $PythonSpec --yes" `
    -Action { Invoke-NativeCommand -Executable $Conda -Arguments @("install", "-p", $EnvPath, "-c", "conda-forge", $PythonSpec, "--yes") }

Invoke-SetupStep `
    -Display "$Conda run -p `"$EnvPath`" python -c `"$PythonVersionCheck`"" `
    -Action { Invoke-NativeCommand -Executable $Conda -Arguments @("run", "-p", $EnvPath, "python", "-c", $PythonVersionCheck) }

if ($RunValidation) {
    Push-Location $RepoRoot
    try {
        Invoke-SetupStep `
            -Display "PYTHONPATH=`"$SourcePath`"; $Conda run -p `"$EnvPath`" python -m unittest discover -s tests" `
            -Action {
                Invoke-WithProjectPythonPath {
                    Invoke-NativeCommand -Executable $Conda -Arguments @("run", "-p", $EnvPath, "python", "-m", "unittest", "discover", "-s", "tests")
                }
            }

        Invoke-SetupStep `
            -Display "PYTHONPATH=`"$SourcePath`"; $Conda run -p `"$EnvPath`" python scripts\run_minimal_closure.py" `
            -Action {
                Invoke-WithProjectPythonPath {
                    Invoke-NativeCommand -Executable $Conda -Arguments @("run", "-p", $EnvPath, "python", (Join-Path $RepoRoot "scripts\run_minimal_closure.py"))
                }
            }
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Next commands:"
Write-Host "  conda activate $EnvPath"
Write-Host "  `$env:PYTHONPATH='src'"
Write-Host "  python -m unittest discover -s tests"
Write-Host "  python scripts\run_minimal_closure.py"
