param([string]$SpecPath = (Join-Path $PSScriptRoot 'generation-spec.json'))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$spec = Get-Content -LiteralPath $SpecPath -Raw -Encoding utf8 | ConvertFrom-Json
$entries = @()
foreach ($job in $spec.jobs) {
    $sourcePath = Join-Path $projectRoot $job.source
    $targetPath = Join-Path $projectRoot $job.target
    $source = [System.Drawing.Bitmap]::new($sourcePath)
    if (Test-Path -LiteralPath $targetPath) {
        $old = [System.Drawing.Bitmap]::new($targetPath)
        $width = $old.Width
        $height = $old.Height
        $old.Dispose()
    } else {
        $width = [int]$job.width
        $height = [int]$job.height
        if ($width -le 0 -or $height -le 0) { throw ('Missing dimensions: '+$job.target) }
        New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
    }
    $sheet = [System.Drawing.Bitmap]::new($width, $height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($sheet)
    $graphics.Clear([System.Drawing.Color]::Transparent)
    $graphics.CompositingMode = [System.Drawing.Drawing2D.CompositingMode]::SourceCopy
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $frames = @()
    for ($index = 0; $index -lt 4; $index++) {
        $left = [int][Math]::Round($width * $index / 4.0)
        $right = [int][Math]::Round($width * ($index + 1) / 4.0)
        $sourceLeft = [int][Math]::Round($source.Width * ($index % 2) / 2.0)
        $sourceTop = [int][Math]::Round($source.Height * [Math]::Floor($index / 2.0) / 2.0)
        $sourceRight = [int][Math]::Round($source.Width * (($index % 2) + 1) / 2.0)
        $sourceBottom = [int][Math]::Round($source.Height * ([Math]::Floor($index / 2.0) + 1) / 2.0)
        $cellWidth = $sourceRight - $sourceLeft
        $cellHeight = $sourceBottom - $sourceTop
        $scale = [Math]::Min(($right - $left) / [double]$cellWidth, $height / [double]$cellHeight)
        $drawWidth = [int][Math]::Round($cellWidth * $scale)
        $drawHeight = [int][Math]::Round($cellHeight * $scale)
        $x = $left + [int][Math]::Floor((($right - $left) - $drawWidth) / 2.0)
        $y = [int][Math]::Floor(($height - $drawHeight) / 2.0)
        $destination = [System.Drawing.Rectangle]::new($x, $y, $drawWidth, $drawHeight)
        $attributes = [System.Drawing.Imaging.ImageAttributes]::new()
        $matrix = [System.Drawing.Imaging.ColorMatrix]::new()
        if ($job.action -eq 'die' -and $index -eq 3) { $matrix.Matrix33 = 0.16 }
        $attributes.SetColorMatrix($matrix)
        $graphics.DrawImage($source, $destination, $sourceLeft, $sourceTop, $cellWidth, $cellHeight, [System.Drawing.GraphicsUnit]::Pixel, $attributes)
        $attributes.Dispose()
        $frames += [ordered]@{ x = $left; y = 0; width = ($right - $left); height = $height }
    }
    $graphics.Dispose()
    $source.Dispose()
    $sheet.Save($targetPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $sheet.Dispose()
    foreach ($alias in $job.aliases) { Copy-Item -LiteralPath $targetPath -Destination (Join-Path $projectRoot $alias) -Force }
    $entries += [ordered]@{
        file = $job.target; aliases = @($job.aliases); action = $job.action; piece = $job.piece; color = $job.color
        width = $width; height = $height; frames = $frames
        frameDurationsMs = $(switch ($job.action) { 'move' { @(120,120,120,120) }; 'attack' { @(90,90,80,140) }; 'impact' { @(40,70,100,160) }; default { @(140,160,180,200) } })
        loop = ($job.action -eq 'move'); hideAfterPlayback = ($job.action -eq 'die')
    }
    Write-Output $job.target
}
$manifestPath = Join-Path $projectRoot 'img\animation-fix-manifest.json'
if (Test-Path -LiteralPath $manifestPath) {
    $priorManifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
    $newFiles = @($entries | ForEach-Object { $_.file })
    $entries = @($priorManifest.animations | Where-Object { $_.file -notin $newFiles }) + $entries
}
$manifest = [ordered]@{
    schemaVersion = 1; generatedWith = 'imagegen built-in'; frameOrder = 'left-to-right'
    notes = @('Use explicit integer frame rectangles: 1254 is not divisible by four.', 'die and defeat are aliases; play once and remove the captured piece afterwards.', 'For kings, use defeat only for the end-of-game visual; kings are never captured in chess.')
    animations = $entries
}
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $projectRoot 'img\animation-fix-manifest.json') -Encoding utf8
