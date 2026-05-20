param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

Write-Host "Stopping Streamlit service on port $Port ..."

$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "python|streamlit" -and
    $_.CommandLine -and
    $_.CommandLine -like "*streamlit*" -and
    $_.CommandLine -like "*app.py*" -and
    (
        $_.CommandLine -like "*--server.port $Port*" -or
        $_.CommandLine -like "*--server.port*$Port*"
    )
}

if (-not $procs) {
    Write-Host "No matching Streamlit process found."
    exit 0
}

foreach ($proc in $procs) {
    try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        Write-Host "Stopped PID $($proc.ProcessId)"
    } catch {
        Write-Warning "Failed to stop PID $($proc.ProcessId): $($_.Exception.Message)"
    }
}
