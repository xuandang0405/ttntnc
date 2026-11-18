$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$layout = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'kit-layout.json') -Raw -Encoding utf8 | ConvertFrom-Json
$entries = @()
foreach ($job in $layout.jobs) {
    $source = [System.Drawing.Bitmap]::new((Join-Path $root $job.source))
    $sheet = [System.Drawing.Bitmap]::new(1280,360,[System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $sg = [System.Drawing.Graphics]::FromImage($sheet)
    $sg.Clear([System.Drawing.Color]::Transparent)
    for ($i=0;$i -lt 4;$i++) {
        $f = $job.frames[$i]
        # An isolated cell surface guarantees that neighboring frames never bleed.
        $cell = [System.Drawing.Bitmap]::new(320,360,[System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
        $cg = [System.Drawing.Graphics]::FromImage($cell)
        $cg.Clear([System.Drawing.Color]::Transparent)
        $cg.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $cg.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $dest = [System.Drawing.RectangleF]::new([single]$f.drawX,[single]$f.drawY,[single]($f.width*$f.scale),[single]($f.height*$f.scale))
        $src = [System.Drawing.RectangleF]::new([single]$f.x,0,[single]$f.width,[single]$f.height)
        $cg.DrawImage($source,$dest,$src,[System.Drawing.GraphicsUnit]::Pixel)
        $cg.Dispose()
        $sg.DrawImageUnscaled($cell,$i*320,0)
        $cell.Dispose()
    }
    $sg.Dispose(); $source.Dispose()
    $target = Join-Path $root $job.target
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    $sheet.Save($target,[System.Drawing.Imaging.ImageFormat]::Png); $sheet.Dispose()
    $entries += [ordered]@{
        file = $job.target.Substring('img/animations/'.Length); action=$job.action; piece=$job.piece; color=$job.color
        frameWidth=320; frameHeight=360; frameCount=4; anchor=@{x=160;y=316}
        frameDurationsMs=$(if($job.action -eq 'attack'){@(140,140,140,180)}elseif($job.action -eq 'die'){@(170,200,230,260)}else{@($job.frameDurationsMs)}); loop=$job.loop; hideAfterPlayback=$job.hideAfterPlayback
    }
    if ($job.action -eq 'move') {
        $entries += [ordered]@{file=$job.target.Substring('img/animations/'.Length);action='idle';piece=$job.piece;color=$job.color;frameWidth=320;frameHeight=360;frameCount=4;frameIndices=@(0,1,2,3);anchor=@{x=160;y=316};frameDurationsMs=@(240,280,240,280);loop=$true;hideAfterPlayback=$false}
    }
}
foreach ($side in @('white','black')) {
    $relative='effects/capture_impact_'+$side+'.png'
    $dest=Join-Path $root ('img/animations/'+$relative)
    New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $root ('img/'+$relative)) -Destination $dest -Force
    $entries += [ordered]@{file=$relative;action='impact';piece='effect';color=$side;frameWidth=320;frameHeight=320;frameCount=4;anchor=@{x=160;y=192};frameDurationsMs=@(70,90,120,160);loop=$false;hideAfterPlayback=$true}
}
$manifest = [ordered]@{
    schemaVersion=3; generatedWith='imagegen built-in; deterministic sprite registration and packing'
    frameOrder='left-to-right'; animations=$entries
    motion=@{easing='easeInOutCubic';durationMs=700;minDurationMs=650;maxDurationMs=1100;knightArcHeightInSquares=0.45;rook='straight';bishop='diagonal';queen='straight-or-diagonal';pawnMove='forward';pawnCapture='diagonal';king='one-square'}
    capture=@{travelMs=480;attackStartMs=480;impactMs=760;victimDieStartMs=760;attackerArrivesMs=1160;victimHiddenMs=1620;completeMs=1760;captureSquare='explicit; defaults to targetSquare';victimMayBeKing=$false}
    notes=@('Render sprite origin at board anchor minus animation.anchor; use the same anchor for every state.','Idle loops all four registered move frames with its own timing; the game adds breathing, base aura and slow motes without changing piece identity.','Only the attacker stays intact; play the victim die once, then remove it.','Castling animates king and rook together; en passant removes the victim at captureSquare; promotion swaps the piece after arrival.')
}
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $root 'img/animations/manifest.json') -Encoding utf8
Write-Output ('Packed '+$layout.jobs.Count+' piece sheets and 2 impact effects')
