param(
    [string]$BaseRef = "",
    [string]$PythonExe = "",
    [switch]$SkipUnitTests,
    [switch]$SkipCompile,
    [switch]$SkipSecretScan,
    [switch]$RequireClean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ProjectRoot "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SummaryPath = Join-Path $LogDir "review_gate_$Timestamp.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not $PythonExe) {
    $Candidate = Join-Path $ProjectRoot "SongFormer_install\venv_gpu\Scripts\python.exe"
    if (Test-Path -LiteralPath $Candidate) {
        $PythonExe = $Candidate
    } else {
        $PythonExe = "python"
    }
}

$summary = [ordered]@{
    started_at = (Get-Date).ToString("s")
    project_root = $ProjectRoot
    base_ref = $BaseRef
    python = $PythonExe
    result = "PASS"
    checks = @()
}

function Add-Check {
    param(
        [string]$Name,
        [string]$Status,
        [int]$ExitCode,
        [double]$DurationSec,
        [string]$Output
    )
    $script:summary.checks += [ordered]@{
        name = $Name
        status = $Status
        exit_code = $ExitCode
        duration_sec = [Math]::Round($DurationSec, 2)
        output_tail = (($Output -split "`r?`n") | Select-Object -Last 80) -join "`n"
    }
    if ($Status -eq "FAIL") {
        $script:summary.result = "FAIL"
    }
}

function Invoke-NativeCheck {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments
    )
    Write-Host "[RUN] $Name"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $output = ""
    $exitCode = 0
    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()
    try {
        Push-Location $ProjectRoot
        $process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $Arguments `
            -WorkingDirectory $ProjectRoot `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile
        $stdout = if (Test-Path -LiteralPath $stdoutFile) { Get-Content -LiteralPath $stdoutFile -Raw -ErrorAction SilentlyContinue } else { "" }
        $stderr = if (Test-Path -LiteralPath $stderrFile) { Get-Content -LiteralPath $stderrFile -Raw -ErrorAction SilentlyContinue } else { "" }
        $output = (($stdout, $stderr) | Where-Object { $_ }) -join "`n"
        $exitCode = [int]$process.ExitCode
    } catch {
        $output = $_.Exception.Message
        $exitCode = 1
    } finally {
        Pop-Location
        Remove-Item -LiteralPath $stdoutFile, $stderrFile -Force -ErrorAction SilentlyContinue
        $sw.Stop()
    }

    $status = if ($exitCode -eq 0) { "PASS" } else { "FAIL" }
    Add-Check -Name $Name -Status $status -ExitCode $exitCode -DurationSec $sw.Elapsed.TotalSeconds -Output $output
    Write-Host "[$status] $Name"
}

function Invoke-ScriptCheck {
    param(
        [string]$Name,
        [scriptblock]$ScriptBlock
    )
    Write-Host "[RUN] $Name"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $output = ""
    $exitCode = 0
    try {
        Push-Location $ProjectRoot
        $output = (& $ScriptBlock 2>&1 | Out-String)
    } catch {
        $output = $_.Exception.Message
        $exitCode = 1
    } finally {
        Pop-Location
        $sw.Stop()
    }

    $status = if ($exitCode -eq 0) { "PASS" } else { "FAIL" }
    Add-Check -Name $Name -Status $status -ExitCode $exitCode -DurationSec $sw.Elapsed.TotalSeconds -Output $output
    Write-Host "[$status] $Name"
}

Invoke-NativeCheck -Name "git status" -FilePath "git" -Arguments @("status", "--short", "--branch")

if ($RequireClean) {
    Invoke-ScriptCheck -Name "working tree clean" -ScriptBlock {
        $status = git status --porcelain
        if ($status) {
            throw "Working tree is not clean.`n$status"
        }
        "Working tree is clean."
    }
}

if (-not $SkipCompile) {
    Invoke-NativeCheck -Name "python compileall" -FilePath $PythonExe -Arguments @("-m", "compileall", "-q", "app.py", "src", "tests")
}

if (-not $SkipUnitTests) {
    Invoke-NativeCheck -Name "unit tests" -FilePath $PythonExe -Arguments @("-m", "unittest", "discover", "-s", "tests", "-v")
}

if ($BaseRef) {
    Invoke-NativeCheck -Name "git diff check ($BaseRef)" -FilePath "git" -Arguments @("diff", "--check", $BaseRef, "--")
} else {
    Invoke-NativeCheck -Name "git diff check (unstaged)" -FilePath "git" -Arguments @("diff", "--check")
    Invoke-NativeCheck -Name "git diff check (staged)" -FilePath "git" -Arguments @("diff", "--cached", "--check")
}

if (-not $SkipSecretScan) {
    Invoke-ScriptCheck -Name "secret scan (diff)" -ScriptBlock {
        if ($BaseRef) {
            $diff = git diff --no-ext-diff --unified=0 $BaseRef --
        } else {
            $diff = git diff --no-ext-diff --unified=0 HEAD --
        }

        $scanText = @()
        if ($diff) {
            $scanText += $diff
        }

        $untrackedFiles = git ls-files --others --exclude-standard
        $textExtensions = @(
            ".py", ".ps1", ".bat", ".vbs", ".md", ".txt", ".json", ".toml",
            ".yaml", ".yml", ".ini", ".cfg", ".env", ".example"
        )
        foreach ($file in $untrackedFiles) {
            $path = Join-Path $ProjectRoot $file
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                continue
            }
            $extension = [System.IO.Path]::GetExtension($path)
            $length = (Get-Item -LiteralPath $path).Length
            if ($textExtensions -contains $extension -and $length -le 1048576) {
                $scanText += "----- UNTRACKED: $file -----"
                $scanText += (Get-Content -LiteralPath $path -ErrorAction SilentlyContinue)
            }
        }

        if ($scanText.Count -eq 0) {
            "No diff or untracked text files to scan."
            return
        }

        $patterns = @(
            "(?i)(access[_-]?token|access[_-]?key|api[_-]?key|secret|password)\s*[:=]\s*['""][^'""]{12,}['""]",
            ("629f" + "3a92"),
            ("gRe" + "Eb1")
        )
        $hits = @()
        foreach ($pattern in $patterns) {
            $matches = $scanText | Select-String -Pattern $pattern
            foreach ($match in $matches) {
                $hits += $match.Line
            }
        }

        if ($hits.Count -gt 0) {
            $sample = ($hits | Select-Object -First 20) -join "`n"
            throw "Potential secret found in diff:`n$sample"
        }

        "No potential secrets found in diff or untracked text files."
    }
}

$summary.finished_at = (Get-Date).ToString("s")
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8

Write-Host ""
Write-Host "Review gate result: $($summary.result)"
Write-Host "Summary: $SummaryPath"

if ($summary.result -ne "PASS") {
    exit 1
}
