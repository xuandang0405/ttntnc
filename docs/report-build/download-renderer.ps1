$ErrorActionPreference = 'Stop'
$downloadRoot = Join-Path $env:LOCALAPPDATA 'CodexReportTools/LibreOffice-26.2.6'
New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
$totalBytes = 373252096L
$partCount = 12
$chunkBytes = [long][Math]::Ceiling($totalBytes / $partCount)
$downloadUrl = 'https://ftp.jaist.ac.jp/pub/tdf/libreoffice/stable/26.2.6/win/x86_64/LibreOffice_26.2.6_Win_x86-64.msi'
$records = 0..($partCount - 1) | ForEach-Object -Parallel {
    $partIndex = $_
    $partStart = $partIndex * $using:chunkBytes
    $partEnd = [Math]::Min($using:totalBytes - 1, $partStart + $using:chunkBytes - 1)
    $partFile = Join-Path $using:downloadRoot ('part-{0:d2}.bin' -f $partIndex)
    & curl.exe --fail --silent --show-error --retry 3 --range "$partStart-$partEnd" --output $partFile $using:downloadUrl
    if ($LASTEXITCODE -ne 0) { throw "Download failed for part $partIndex" }
    $partLength = (Get-Item -LiteralPath $partFile).Length
    if ($partLength -ne ($partEnd - $partStart + 1)) { throw "Wrong size for part $partIndex" }
    [PSCustomObject]@{ Index = $partIndex; File = $partFile; Length = $partLength }
} -ThrottleLimit 12
if ($records.Count -ne $partCount) { throw 'Missing download parts' }
$installerFile = Join-Path $downloadRoot 'LibreOffice.msi'
$targetStream = [IO.File]::Create($installerFile)
try {
    foreach ($record in ($records | Sort-Object Index)) {
        $sourceStream = [IO.File]::OpenRead($record.File)
        try { $sourceStream.CopyTo($targetStream) } finally { $sourceStream.Dispose() }
    }
} finally { $targetStream.Dispose() }
$expectedHash = [Convert]::ToHexString([Convert]::FromBase64String('+YdwMv2Qi+ucDd8G30r1wuhfQZxC4Uh2xMzlquX7JmA='))
$actualHash = (Get-FileHash -LiteralPath $installerFile -Algorithm SHA256).Hash
if ($actualHash -ne $expectedHash) { throw 'Installer checksum mismatch' }
Write-Output "Verified renderer installer: $installerFile"
$extractPath = Join-Path $downloadRoot 'extracted'
$installLog = Join-Path $downloadRoot 'extract.log'
$process = Start-Process -FilePath 'msiexec.exe' -ArgumentList @('/a', ('"' + $installerFile + '"'), '/qn', ('TARGETDIR="' + $extractPath + '"'), '/L*v', ('"' + $installLog + '"')) -WindowStyle Hidden -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "Renderer extraction failed: $($process.ExitCode)" }
Get-ChildItem -LiteralPath $extractPath -Recurse -Filter soffice.exe | Select-Object -ExpandProperty FullName
