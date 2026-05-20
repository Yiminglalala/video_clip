param(
    [int]$Port = 8501,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot "SongFormer_install\venv_gpu\Scripts\python.exe"
$AppPath = Join-Path $ProjectRoot "app.py"
$LogDir = Join-Path $ProjectRoot "logs"
$OutLog = Join-Path $LogDir "streamlit_$Port.out.log"
$ErrLog = Join-Path $LogDir "streamlit_$Port.err.log"
$Url = "http://localhost:$Port"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $AppPath)) {
    throw "App not found: $AppPath"
}

Write-Host "[1/3] Stopping existing Streamlit service on port $Port ..."
$oldProcs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "python|streamlit" -and
    $_.CommandLine -and
    $_.CommandLine -like "*streamlit*" -and
    $_.CommandLine -like "*app.py*" -and
    (
        $_.CommandLine -like "*--server.port $Port*" -or
        $_.CommandLine -like "*--server.port*$Port*"
    )
}
foreach ($proc in $oldProcs) {
    try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        Write-Host "  stopped PID $($proc.ProcessId)"
    } catch {
        Write-Warning "  failed to stop PID $($proc.ProcessId): $($_.Exception.Message)"
    }
}
Start-Sleep -Seconds 2

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (Test-Path -LiteralPath $OutLog) { Remove-Item -LiteralPath $OutLog -Force }
if (Test-Path -LiteralPath $ErrLog) { Remove-Item -LiteralPath $ErrLog -Force }

Write-Host "[2/3] Starting service ..."
$args = @(
    "-m", "streamlit", "run", $AppPath,
    "--server.port", "$Port",
    "--server.headless", "true",
    "--server.fileWatcherType", "none"
)
$process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $args `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "  started PID $($process.Id)"

Write-Host "[3/3] Waiting for HTTP 200 ..."
$ready = $false
for ($i = 1; $i -le 60; $i++) {
    try {
        $request = [System.Net.WebRequest]::Create($Url)
        $request.Timeout = 2000
        $response = $request.GetResponse()
        $statusCode = [int]$response.StatusCode
        $response.Close()
        if ($statusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    Write-Host "Service failed to start. Recent logs:"
    if (Test-Path -LiteralPath $ErrLog) {
        Write-Host "--- stderr ---"
        Get-Content -LiteralPath $ErrLog -Tail 80
    }
    if (Test-Path -LiteralPath $OutLog) {
        Write-Host "--- stdout ---"
        Get-Content -LiteralPath $OutLog -Tail 80
    }
    exit 1
}

Write-Host "Service ready: $Url"
Write-Host "Logs: $LogDir"
if ($OpenBrowser) {
    Start-Process $Url
}
