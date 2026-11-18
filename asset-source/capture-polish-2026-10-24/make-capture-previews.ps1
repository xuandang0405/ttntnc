$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName PresentationCore
$root=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$out=Join-Path $root 'animation-preview\capture-complete'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$kit=Get-Content -LiteralPath (Join-Path $root 'img\animations\manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
$images=@{}; $states=@{}
foreach($entry in $kit.animations){
    $key=$entry.color+'/'+$entry.piece+'/'+$entry.action
    $states[$key]=$entry
    if(-not $images.ContainsKey($entry.file)){$images[$entry.file]=[System.Drawing.Bitmap]::new((Join-Path $root ('img/animations/'+$entry.file)))}
}
$font=[System.Drawing.Font]::new('Arial',11)
$titleFont=[System.Drawing.Font]::new('Arial',14,[System.Drawing.FontStyle]::Bold)
$white=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(235,239,247))
$dim=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(160,175,192))
$lightSquare=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(64,75,86))
$darkSquare=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(43,53,65))
$goldPen=[System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(235,184,76),2)
$purplePen=[System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(189,112,248),2)
$targetPen=[System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(224,108,108),2)
$names=@{pawn='Tốt';knight='Mã';bishop='Tượng';rook='Xe';queen='Hậu';king='Vua'}
function Draw-Sprite($g,$side,$piece,$action,$index,$x,$y,$scale){
    $entry=$states[$side+'/'+$piece+'/'+$action]
    if(-not $entry){throw ('Missing sprite '+$side+'/'+$piece+'/'+$action)}
    if($entry.action -eq 'idle'){$index=0}
    $dest=[System.Drawing.RectangleF]::new([single]($x-$entry.anchor.x*$scale),[single]($y-$entry.anchor.y*$scale),[single]($entry.frameWidth*$scale),[single]($entry.frameHeight*$scale))
    $src=[System.Drawing.RectangleF]::new([single]($index*$entry.frameWidth),0,[single]$entry.frameWidth,[single]$entry.frameHeight)
    $g.DrawImage($images[$entry.file],$dest,$src,[System.Drawing.GraphicsUnit]::Pixel)
}
function Ease($t){$t=[Math]::Clamp([double]$t,0,1); if($t -lt .5){return 4*$t*$t*$t}; return 1-[Math]::Pow(-2*$t+2,3)/2}
function Frame-At($durations,$ms){$acc=0; for($i=0;$i -lt $durations.Count;$i++){$acc+=$durations[$i];if($ms -lt $acc){return $i}};return -1}
function Add-GifFrame($encoder,$canvas){
    $memory=[System.IO.MemoryStream]::new();$canvas.Save($memory,[System.Drawing.Imaging.ImageFormat]::Png);$memory.Position=0
    $source=[System.Windows.Media.Imaging.BitmapFrame]::Create($memory,[System.Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat,[System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad)
    $encoder.Frames.Add($source);$memory.Dispose()
}
function Save-Gif($encoder,$path,$delays){
    $stream=[System.IO.File]::Create($path);$encoder.Save($stream);$stream.Dispose()
    $bytes=[System.IO.File]::ReadAllBytes($path);$paletteSize=0
    if(($bytes[10] -band 128) -ne 0){$paletteSize=3*[int][Math]::Pow(2,($bytes[10] -band 7)+1)}
    $insertAt=13+$paletteSize
    $loop=[byte[]]@(0x21,0xFF,0x0B,0x4E,0x45,0x54,0x53,0x43,0x41,0x50,0x45,0x32,0x2E,0x30,0x03,0x01,0x00,0x00,0x00)
    $joined=[byte[]]::new($bytes.Length+$loop.Length)
    [Array]::Copy($bytes,0,$joined,0,$insertAt);[Array]::Copy($loop,0,$joined,$insertAt,$loop.Length);[Array]::Copy($bytes,$insertAt,$joined,$insertAt+$loop.Length,$bytes.Length-$insertAt)
    $cursor=$insertAt;$count=0
    while($cursor -lt $joined.Length){
        $marker=$joined[$cursor];if($marker -eq 0x3B){break}
        if($marker -eq 0x21){
            if($joined[$cursor+1] -eq 0xF9){$joined[$cursor+4]=[byte]($delays[$count] -band 255);$joined[$cursor+5]=[byte](($delays[$count] -shr 8) -band 255);$joined[$cursor+3]=[byte](($joined[$cursor+3] -band 0xE3) -bor 4);$count++}
            $cursor+=2
        }elseif($marker -eq 0x2C){$flags=$joined[$cursor+9];$cursor+=10;if(($flags -band 128) -ne 0){$cursor+=3*[int][Math]::Pow(2,($flags -band 7)+1)};$cursor++}else{throw 'Invalid GIF block'}
        while($true){$length=$joined[$cursor];$cursor+=1+$length;if($length -eq 0){break}}
    }
    if($count -ne $delays.Count){throw 'GIF frame timing mismatch'}
    [System.IO.File]::WriteAllBytes($path,$joined)
}

foreach($action in @('move','attack','die','impact')){
    $entries=@($kit.animations | Where-Object {$_.action -eq $action} | Sort-Object color,piece)
    $canvas=[System.Drawing.Bitmap]::new(1000,60+195*$entries.Count)
    $g=[System.Drawing.Graphics]::FromImage($canvas);$g.Clear([System.Drawing.Color]::FromArgb(20,25,34));$g.InterpolationMode=[System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.DrawString(($action.ToUpper()+' - registered frames - left to right'),$titleFont,$white,15,12)
    for($row=0;$row -lt $entries.Count;$row++){
        $e=$entries[$row];$top=50+$row*195
        $g.DrawString(($e.piece+' / '+$e.color),$font,$white,12,$top+82)
        for($f=0;$f -lt 4;$f++){
            $x=205+$f*195
            for($cy=0;$cy -lt 180;$cy+=15){for($cx=0;$cx -lt 180;$cx+=15){$b=$darkSquare;if((($cx/15+$cy/15)%2) -eq 0){$b=$lightSquare};$g.FillRectangle($b,$x+$cx,$top+$cy,15,15)}}
            $scale=180.0/$e.frameHeight
            $dest=[System.Drawing.RectangleF]::new([single]($x+(180-$e.frameWidth*$scale)/2),[single]$top,[single]($e.frameWidth*$scale),180)
            $src=[System.Drawing.RectangleF]::new([single]($f*$e.frameWidth),0,[single]$e.frameWidth,[single]$e.frameHeight)
            $g.DrawImage($images[$e.file],$dest,$src,[System.Drawing.GraphicsUnit]::Pixel)
        }
    }
    $g.Dispose();$canvas.Save((Join-Path $out ($action+'-frames.png')),[System.Drawing.Imaging.ImageFormat]::Png);$canvas.Dispose()
}

$pieces=@('pawn','knight','bishop','rook','queen','king')
$victims=@('knight','rook','bishop','queen','pawn','rook')
$times=@(0..25 | ForEach-Object {$_*50})+@(1500,1750)
$delays=@();for($i=0;$i -lt $times.Count;$i++){if($i+1 -lt $times.Count){$delays+=[int](($times[$i+1]-$times[$i])/10)}else{$delays+=40}}
foreach($side in @('white','black')){
    $enemy='black';$sideLabel='trắng';$enemyLabel='đen';if($side -eq 'black'){$enemy='white';$sideLabel='đen';$enemyLabel='trắng'}
    $encoder=[System.Windows.Media.Imaging.GifBitmapEncoder]::new()
    foreach($time in $times){
        $canvas=[System.Drawing.Bitmap]::new(900,700)
        $g=[System.Drawing.Graphics]::FromImage($canvas);$g.Clear([System.Drawing.Color]::FromArgb(17,23,32));$g.InterpolationMode=[System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $g.DrawString(('Di chuyển → ăn quân địch | bên '+$sideLabel),$titleFont,$white,15,10)
        $phase='Di chuyển';if($time -ge 360){$phase='Ra đòn'};if($time -ge 540){$phase='Va chạm / vỡ'};if($time -ge 1020){$phase='Tan biến'};if($time -ge 1220){$phase='Hoàn tất'}
        $g.DrawString(($phase+'   '+$time+' ms'),$font,$dim,580,15)
        for($j=0;$j -lt 6;$j++){
            $piece=$pieces[$j];$victim=$victims[$j];$px=($j%3)*300;$py=40+[int][Math]::Floor($j/3.0)*325
            $g.DrawString(($names[$piece]+' '+$sideLabel+' ăn '+$names[$victim]+' '+$enemyLabel),$font,$white,$px+12,$py+4)
            $bx=$px+18;$by=$py+36;$cell=52
            for($r=0;$r -lt 5;$r++){for($c=0;$c -lt 5;$c++){$b=$darkSquare;if((($r+$c)%2) -eq 0){$b=$lightSquare};$g.FillRectangle($b,$bx+$c*$cell,$by+$r*$cell,$cell,$cell)}}
            $sr=3;$tr=2;$tc=2;if($side -eq 'black'){$sr=1}
            if($piece -in @('bishop','queen')){$tc=3;$tr=1;if($side -eq 'black'){$tr=3}}
            if($piece -eq 'rook'){$tc=3;$tr=$sr}
            if($piece -eq 'knight'){$tc=3;$tr=2}
            $sx=$bx+1.5*$cell;$sy=$by+($sr+.5)*$cell;$tx=$bx+($tc+.5)*$cell;$ty=$by+($tr+.5)*$cell
            $pen=$goldPen;if($side -eq 'black'){$pen=$purplePen}
            $g.DrawRectangle($pen,$bx+$cell+1,$by+$sr*$cell+1,$cell-2,$cell-2);$g.DrawRectangle($targetPen,$bx+$tc*$cell+1,$by+$tr*$cell+1,$cell-2,$cell-2)
            if($time -lt 540){Draw-Sprite $g $enemy $victim 'idle' 0 $tx $ty .36}
            elseif($time -lt 1220){$df=Frame-At @(140,160,180,200) ($time-540);if($df -ge 0){Draw-Sprite $g $enemy $victim 'die' $df $tx $ty .36}}
            $progress=1.0
            if($time -lt 360){$progress=.72*(Ease ($time/360.0))}elseif($time -lt 540){$progress=.72+.13*(Ease (($time-360)/180.0))}elseif($time -lt 760){$progress=.85+.15*(Ease (($time-540)/220.0))}
            $x=$sx+($tx-$sx)*$progress;$y=$sy+($ty-$sy)*$progress
            if($piece -eq 'knight' -and $time -lt 360){$y-=.45*$cell*[Math]::Sin([Math]::PI*$time/360)}
            if($time -lt 360){$f=[int][Math]::Floor($time/120)%4;Draw-Sprite $g $side $piece 'move' $f $x $y .36}
            elseif($time -lt 760){$f=Frame-At @(90,90,80,140) ($time-360);Draw-Sprite $g $side $piece 'attack' $f $x $y .36}
            else{Draw-Sprite $g $side $piece 'idle' 0 $x $y .36}
            if($time -ge 540 -and $time -lt 910){$f=Frame-At @(40,70,100,160) ($time-540);if($f -ge 0){Draw-Sprite $g $side 'effect' 'impact' $f $tx ($ty-34) .4}}
        }
        $g.Dispose()
        if($time -eq 600){$canvas.Save((Join-Path $out ('capture-'+$side+'-preview.png')),[System.Drawing.Imaging.ImageFormat]::Png)}
        if($time -eq 0){$canvas.Save((Join-Path $out ('capture-'+$side+'-start.png')),[System.Drawing.Imaging.ImageFormat]::Png)}
        if($time -eq 1750){$canvas.Save((Join-Path $out ('capture-'+$side+'-finished.png')),[System.Drawing.Imaging.ImageFormat]::Png)}
        Add-GifFrame $encoder $canvas;$canvas.Dispose()
    }
    Save-Gif $encoder (Join-Path $out ('move-and-capture-'+$side+'.gif')) $delays
    Write-Output ('Preview '+$side+' done')
}
foreach($im in $images.Values){$im.Dispose()}
$font.Dispose();$titleFont.Dispose();$white.Dispose();$dim.Dispose();$lightSquare.Dispose();$darkSquare.Dispose();$goldPen.Dispose();$purplePen.Dispose();$targetPen.Dispose()
