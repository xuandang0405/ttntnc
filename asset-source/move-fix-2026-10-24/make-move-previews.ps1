$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName PresentationCore
$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$out = Join-Path $root 'animation-preview\move-review'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$manifest = Get-Content -LiteralPath (Join-Path $root 'img\animation-fix-manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
$jobs = @($manifest.animations | Where-Object { $_.action -eq 'move' } | Sort-Object color,piece)
$font = [System.Drawing.Font]::new('Arial',12)
$smallFont = [System.Drawing.Font]::new('Arial',10)
$textBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(238,241,247))
$brushA = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(37,43,53))
$brushB = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(47,54,64))
function Paint-Checker($g, $x, $y, $w, $h) {
    $g.FillRectangle($brushA,$x,$y,$w,$h)
    for ($cy = 0; $cy -lt $h; $cy += 16) {
        for ($cx = 0; $cx -lt $w; $cx += 16) {
            if ((($cx / 16 + $cy / 16) % 2) -eq 0) { $g.FillRectangle($brushB,$x+$cx,$y+$cy,[Math]::Min(16,$w-$cx),[Math]::Min(16,$h-$cy)) }
        }
    }
}
function Draw-Sprite($g,$image,$frame,$x,$y,$size) {
    $ratio = [Math]::Min($size / [double]$frame.width,$size / [double]$frame.height)
    $w = [int][Math]::Round($frame.width*$ratio)
    $h = [int][Math]::Round($frame.height*$ratio)
    $dest = [System.Drawing.Rectangle]::new($x+[int](($size-$w)/2),$y+[int](($size-$h)/2),$w,$h)
    $g.DrawImage($image,$dest,[int]$frame.x,0,[int]$frame.width,[int]$frame.height,[System.Drawing.GraphicsUnit]::Pixel)
}
$contact = [System.Drawing.Bitmap]::new(1000,70+$jobs.Count*200)
$cg = [System.Drawing.Graphics]::FromImage($contact)
$cg.Clear([System.Drawing.Color]::FromArgb(21,25,33))
$cg.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$cg.DrawString('MOVE - all 12 pieces - frames left to right',$font,$textBrush,18,16)
for ($j = 0; $j -lt $jobs.Count; $j++) {
    $job = $jobs[$j]
    $image = [System.Drawing.Bitmap]::new((Join-Path $root $job.file))
    $top = 55+$j*200
    $cg.DrawString(($job.action+' / '+$job.piece+' / '+$job.color),$font,$textBrush,16,$top+78)
    for ($f = 0; $f -lt 4; $f++) {
        $left = 215+$f*192
        Paint-Checker $cg $left $top 180 180
        Draw-Sprite $cg $image $job.frames[$f] $left $top 180
        $cg.DrawString(($f+1).ToString(),$smallFont,$textBrush,$left+4,$top+3)
    }
    $image.Dispose()
}
$cg.Dispose()
$contact.Save((Join-Path $out 'all-move-frames.png'),[System.Drawing.Imaging.ImageFormat]::Png)
$contact.Dispose()
$encoder = [System.Windows.Media.Imaging.GifBitmapEncoder]::new()
$delays = @(12,12,12,12)
for ($step = 0; $step -lt 4; $step++) {
    $canvas = [System.Drawing.Bitmap]::new(900,1050)
    $g = [System.Drawing.Graphics]::FromImage($canvas)
    $g.Clear([System.Drawing.Color]::FromArgb(21,25,33))
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.DrawString('MOVE - all 12 pieces - 4 frames / 480 ms loop',$font,$textBrush,16,12)
    for ($j = 0; $j -lt $jobs.Count; $j++) {
        $job = $jobs[$j]
        $x = 8+($j % 3)*298
        $y = 42+[int][Math]::Floor($j/3.0)*250
        $g.DrawString(($job.action+' / '+$job.piece+' / '+$job.color),$font,$textBrush,$x+8,$y)
        Paint-Checker $g ($x+34) ($y+27) 220 220
        if ($step -lt 4 -or $job.action -eq 'move') {
            $index = $step % 4
            $image = [System.Drawing.Bitmap]::new((Join-Path $root $job.file))
            Draw-Sprite $g $image $job.frames[$index] ($x+34) ($y+27) 220
            $image.Dispose()
        }
    }
    $g.Dispose()
    $memory = [System.IO.MemoryStream]::new()
    $canvas.Save($memory,[System.Drawing.Imaging.ImageFormat]::Png)
    $canvas.Dispose()
    $memory.Position = 0
    $bitmap = [System.Windows.Media.Imaging.BitmapFrame]::Create($memory,[System.Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat,[System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad)
    $metadata = [System.Windows.Media.Imaging.BitmapMetadata]::new('gif')
    $metadata.SetQuery('/grctlext/Delay',[UInt16]$delays[$step])
    $metadata.SetQuery('/grctlext/Disposal',[byte]1)
    $frame = [System.Windows.Media.Imaging.BitmapFrame]::Create($bitmap,$null,$metadata,$null)
    $encoder.Frames.Add($frame)
    $memory.Dispose()
}
$gifPath = Join-Path $out 'all-move-animations.gif'
$stream = [System.IO.File]::Create($gifPath)
$encoder.Save($stream)
$stream.Dispose()
$bytes = [System.IO.File]::ReadAllBytes($gifPath)
$paletteSize = 0
if (($bytes[10] -band 128) -ne 0) { $paletteSize = 3*[int][Math]::Pow(2,($bytes[10] -band 7)+1) }
$insertAt = 13+$paletteSize
$loop = [byte[]]@(0x21,0xFF,0x0B,0x4E,0x45,0x54,0x53,0x43,0x41,0x50,0x45,0x32,0x2E,0x30,0x03,0x01,0x00,0x00,0x00)
$joined = [byte[]]::new($bytes.Length+$loop.Length)
[Array]::Copy($bytes,0,$joined,0,$insertAt)
[Array]::Copy($loop,0,$joined,$insertAt,$loop.Length)
[Array]::Copy($bytes,$insertAt,$joined,$insertAt+$loop.Length,$bytes.Length-$insertAt)
# WPF may discard GIF frame timing. Set delays on actual parsed GCE blocks.
$cursor = $insertAt
$timedFrames = 0
while ($cursor -lt $joined.Length) {
    $marker = $joined[$cursor]
    if ($marker -eq 0x3B) { break }
    if ($marker -eq 0x21) {
        if ($joined[$cursor+1] -eq 0xF9) {
            if ($timedFrames -ge $delays.Count -or $joined[$cursor+2] -ne 4) { throw 'Invalid GIF frame control block' }
            $joined[$cursor+4] = [byte]($delays[$timedFrames] -band 255)
            $joined[$cursor+5] = [byte](($delays[$timedFrames] -shr 8) -band 255)
            $joined[$cursor+3] = [byte](($joined[$cursor+3] -band 0xE3) -bor 4)
            $timedFrames++
        }
        $cursor += 2
    } elseif ($marker -eq 0x2C) {
        $imageFlags = $joined[$cursor+9]
        $cursor += 10
        if (($imageFlags -band 128) -ne 0) { $cursor += 3*[int][Math]::Pow(2,($imageFlags -band 7)+1) }
        $cursor++ # LZW minimum code size
    } else { throw ('Unexpected GIF marker at '+$cursor) }
    while ($true) {
        $blockLength = $joined[$cursor]
        $cursor += 1+$blockLength
        if ($blockLength -eq 0) { break }
    }
}
if ($timedFrames -ne $delays.Count) { throw 'GIF timing count does not match frame count' }
[System.IO.File]::WriteAllBytes($gifPath,$joined)
$font.Dispose(); $smallFont.Dispose(); $textBrush.Dispose(); $brushA.Dispose(); $brushB.Dispose()
Write-Output $out
