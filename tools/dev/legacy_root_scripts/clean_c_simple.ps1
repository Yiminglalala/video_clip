# C Drive Cleanup Script - Simple Version
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "        C Drive Cleanup Tool" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Show current disk space
Write-Host "Current C Drive Space:" -ForegroundColor Yellow
Get-PSDrive C | Select-Object @{Name='Total(GB)';Expression={[math]::Round(($_.Used+$_.Free)/1GB,2)}},@{Name='Used(GB)';Expression={[math]::Round($_.Used/1GB,2)}},@{Name='Free(GB)';Expression={[math]::Round($_.Free/1GB,2)}} | Format-Table -AutoSize
Write-Host ""

# Define cleanup paths
$cleanPaths = @(
    "$env:TEMP",
    "$env:USERPROFILE\AppData\Local\Temp",
    "$env:WINDIR\Temp",
    "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache",
    "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Cache",
    "$env:WINDIR\SoftwareDistribution\Download",
    "$env:WINDIR\Prefetch"
)

Write-Host "Starting cleanup..." -ForegroundColor Green
Write-Host ""

$freedSpace = 0
$deletedCount = 0

foreach ($path in $cleanPaths) {
    if (Test-Path $path) {
        Write-Host "Processing: $path" -ForegroundColor White
        try {
            $files = Get-ChildItem $path -Recurse -ErrorAction SilentlyContinue
            $size = ($files | Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
            $count = ($files | Measure-Object).Count
            Remove-Item -Path "$path\*" -Recurse -Force -ErrorAction SilentlyContinue
            $freedSpace += $size
            $deletedCount += $count
            $sizeMB = [math]::Round($size/1MB, 2)
            Write-Host "  Deleted $count files, freed $sizeMB MB" -ForegroundColor Green
        } catch {
            Write-Host "  Some files may be in use" -ForegroundColor Yellow
        }
    }
}

# Clean Recycle Bin
Write-Host ""
Write-Host "Emptying Recycle Bin..." -ForegroundColor White
try {
    Clear-RecycleBin -Force -ErrorAction SilentlyContinue
    Write-Host "  Recycle Bin emptied" -ForegroundColor Green
} catch {
    Write-Host "  Recycle Bin already empty or access denied" -ForegroundColor Yellow
}

Write-Host ""
$freedGB = [math]::Round($freedSpace/1GB, 2)
Write-Host "Cleanup Complete!" -ForegroundColor Green
Write-Host "Total space freed: $freedGB GB" -ForegroundColor Cyan
Write-Host "Total files deleted: $deletedCount" -ForegroundColor Cyan
Write-Host ""

Write-Host "New C Drive Space:" -ForegroundColor Yellow
Get-PSDrive C | Select-Object @{Name='Total(GB)';Expression={[math]::Round(($_.Used+$_.Free)/1GB,2)}},@{Name='Used(GB)';Expression={[math]::Round($_.Used/1GB,2)}},@{Name='Free(GB)';Expression={[math]::Round($_.Free/1GB,2)}} | Format-Table -AutoSize

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Note: AI model cache (.cache folder) was not deleted" -ForegroundColor Yellow
Write-Host "Delete manually if needed" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

