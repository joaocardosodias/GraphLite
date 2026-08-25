# Graphite Windows Installer
$ErrorActionPreference = 'Stop'

$Repo = "joaocardosodias/Graphite"
$InstallDir = if ($env:GRAPHITE_INSTALL_DIR) { $env:GRAPHITE_INSTALL_DIR } else { "$HOME\.graphite\bin" }

function Write-Info($msg) {
    Write-Host "info: $msg"
}

function Write-ErrorExit($msg) {
    Write-Error "error: $msg"
    exit 1
}

$Target = "x86_64-pc-windows-msvc"

try {
    $Release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
    $Tag = $Release.tag_name
} catch {
    $Tag = "v0.1.0"
}

$Version = $Tag.TrimStart('v')
$ArchiveName = "graphite-v$Version-$Target.zip"
$DownloadUrl = "https://github.com/$Repo/releases/download/$Tag/$ArchiveName"

Write-Info "Downloading graphite $Tag for $Target..."

$TempZip = Join-Path $env:TEMP $ArchiveName
$TempExtract = Join-Path $env:TEMP "graphite-extract"

try {
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $TempZip -UseBasicParsing
    
    if (Test-Path $TempExtract) { Remove-Item -Recurse -Force $TempExtract }
    Expand-Archive -Path $TempZip -DestinationPath $TempExtract
    
    if (!(Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    }
    
    Get-ChildItem -Path $TempExtract -Recurse -Filter "graphite.exe" | Copy-Item -Destination $InstallDir -Force
    
    Write-Info "Graphite binary installed to $InstallDir\graphite.exe"
} catch {
    Write-ErrorExit "Failed to download and install binary archive."
} finally {
    if (Test-Path $TempZip) { Remove-Item -Force $TempZip -ErrorAction SilentlyContinue }
    if (Test-Path $TempExtract) { Remove-Item -Recurse -Force $TempExtract -ErrorAction SilentlyContinue }
}
