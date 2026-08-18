# Install the reFire CEP panel for Premiere Pro.
#
#   powershell -ExecutionPolicy Bypass -File refire\ppro\install.ps1
#
# Junctions this folder into Adobe's extensions dir (so edits here are live -- no
# reinstall) and flips PlayerDebugMode, which lets Premiere load an unsigned extension.
# Restart Premiere afterwards: Window > Extensions > reFire.
# Installs to its own folder, so the AE panel (refire\ae\install.ps1) can stay installed.
# ponytail: no ZXP signing/packaging -- that's only needed to ship to other machines.

$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
$dst = Join-Path $env:APPDATA 'Adobe\CEP\extensions\reFirePpro'

if (Test-Path $dst) {
    $item = Get-Item $dst -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        Remove-Item $dst -Force            # our own junction from a previous install
        Write-Host "removed old link  $dst"
    } else {
        throw "$dst exists and is a real folder, not a link. Move it aside first."
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
New-Item -ItemType Junction -Path $dst -Target $src | Out-Null
Write-Host "linked            $dst  ->  $src"

# Unsigned extensions load only with PlayerDebugMode=1. The CSXS version varies by
# AE release, so stamp the range that covers AE 2020..2026 rather than guessing.
foreach ($v in 9..13) {
    $key = "HKCU:\Software\Adobe\CSXS.$v"
    New-Item -Path $key -Force | Out-Null
    Set-ItemProperty -Path $key -Name PlayerDebugMode -Value '1' -Type String
}
Write-Host "PlayerDebugMode   set for CSXS 9-13"
Write-Host ''
Write-Host 'Done. Restart Premiere Pro, then Window > Extensions > reFire.'
Write-Host 'Blank/white panel? PlayerDebugMode did not take -- rerun this script.'
