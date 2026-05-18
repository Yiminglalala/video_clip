# C盘空间清理脚本
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "        C盘空间清理工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 显示当前C盘空间
Write-Host "当前C盘空间状态：" -ForegroundColor Yellow
Get-PSDrive C | Select-Object @{Name='总容量(GB)';Expression={[math]::Round(($_.Used+$_.Free)/1GB,2}},@{Name='已使用(GB)';Expression={[math]::Round($_.Used/1GB,2)}},@{Name='剩余(GB)';Expression={[math]::Round($_.Free/1GB,2)}} | Format-Table -AutoSize
Write-Host ""

# 定义可安全删除的路径
$cleanPaths = @(
    # 用户临时文件
    @{Path="$env:TEMP"; Description="用户临时文件"},
    @{Path="$env:USERPROFILE\AppData\Local\Temp"; Description="本地临时文件"},
    # Windows临时文件
    @{Path="$env:WINDIR\Temp"; Description="Windows临时文件"},
    # 浏览器缓存
    @{Path="$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache"; Description="Chrome缓存"},
    @{Path="$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Cache"; Description="Edge缓存"},
    # Windows更新缓存
    @{Path="$env:WINDIR\SoftwareDistribution\Download"; Description="Windows更新下载文件"},
    # Prefetch
    @{Path="$env:WINDIR\Prefetch"; Description="预读取文件"},
    # 回收站
    @{Path="$env:SystemDrive\`$Recycle.Bin"; Description="回收站"},
    # 缩略图缓存
    @{Path="$env:LOCALAPPDATA\Microsoft\Windows\Explorer"; Description="缩略图缓存"}
)

Write-Host "可清理的项目：" -ForegroundColor Green
Write-Host ""
$totalSize = 0

foreach ($item in $cleanPaths) {
    if (Test-Path $item.Path) {
        $size = (Get-ChildItem $item.Path -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Sum
        $sizeGB = [math]::Round($size/1GB, 2)
        $totalSize += $size
        Write-Host "  [$($item.Description)]" -ForegroundColor White
        Write-Host "    路径: $($item.Path)" -ForegroundColor Gray
        Write-Host "    大小: $sizeGB GB" -ForegroundColor Yellow
        Write-Host ""
    }
}

$totalSizeGB = [math]::Round($totalSize/1GB, 2)
Write-Host "预计可释放空间: $totalSizeGB GB" -ForegroundColor Cyan
Write-Host ""

# 询问是否继续
$confirm = Read-Host "是否继续清理？(Y/N)"
if ($confirm -eq 'Y' -or $confirm -eq 'y') {
    Write-Host ""
    Write-Host "开始清理..." -ForegroundColor Green
    
    $freedSpace = 0
    
    foreach ($item in $cleanPaths) {
        if (Test-Path $item.Path) {
            Write-Host "  清理: $($item.Description)..." -ForegroundColor White
            try {
                $files = Get-ChildItem $item.Path -Recurse -ErrorAction SilentlyContinue
                $size = ($files | Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
                Remove-Item -Path "$($item.Path)\*" -Recurse -Force -ErrorAction SilentlyContinue
                $freedSpace += $size
                Write-Host "    完成!" -ForegroundColor Green
            } catch {
                Write-Host "    部分文件可能被占用" -ForegroundColor Yellow
            }
        }
    }
    
    Write-Host ""
    $freedGB = [math]::Round($freedSpace/1GB, 2)
    Write-Host "清理完成！已释放 $freedGB GB 空间" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "新的C盘空间状态：" -ForegroundColor Yellow
    Get-PSDrive C | Select-Object @{Name='总容量(GB)';Expression={[math]::Round(($_.Used+$_.Free)/1GB,2)}},@{Name='已使用(GB)';Expression={[math]::Round($_.Used/1GB,2)}},@{Name='剩余(GB)';Expression={[math]::Round($_.Free/1GB,2)}} | Format-Table -AutoSize
} else {
    Write-Host "已取消清理" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "提示：对于 .cache 文件夹中的AI模型文件可能需要手动评估是否删除" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

