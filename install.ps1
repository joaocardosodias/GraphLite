# Graphite Windows Installer
$ErrorActionPreference = 'Stop'

$Repo = "joaocardosodias/Graphite"
$InstallDir = "$HOME\.graphite\bin"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  🚀 Installing Graphite (GraphRAG Embedded Engine)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

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

Write-Host "Downloading $DownloadUrl..." -ForegroundColor Yellow

$TempZip = Join-Path $env:TEMP $ArchiveName
$TempExtract = Join-Path $env:TEMP "graphite-extract"

Invoke-WebRequest -Uri $DownloadUrl -OutFile $TempZip

if (Test-Path $TempExtract) { Remove-Item -Recurse -Force $TempExtract }
Expand-Archive -Path $TempZip -DestinationPath $TempExtract

if (!(Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
}

Get-ChildItem -Path $TempExtract -Recurse -Filter "graphite.exe" | Copy-Item -Destination $InstallDir -Force

Remove-Item -Force $TempZip
Remove-Item -Recurse -Force $TempExtract

Write-Host ""
Write-Host "✅ Successfully installed to $InstallDir" -ForegroundColor Green
Write-Host "Make sure $InstallDir is in your User PATH environment variable." -ForegroundColor Gray
Write-Host "Run 'graphite --help' to verify installation." -ForegroundColor Cyan
