<#
.SYNOPSIS
Safely move disposable historical files into a reversible backup directory.

.DESCRIPTION
The default mode is a dry run. Nothing is moved unless -Execute is supplied.
#>

[CmdletBinding()]
param(
    [string]$Root = "E:\projects\doc_auto_handling",
    [string]$BackupDir,
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($BackupDir)) {
    $BackupDir = Join-Path $Root "_backup"
}

if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    throw "Root directory does not exist: $Root"
}

$rootFull = [System.IO.Path]::GetFullPath($Root)
$backupFull = [System.IO.Path]::GetFullPath($BackupDir)
$candidates = New-Object System.Collections.Generic.List[object]

function Get-RelativePath {
    param([string]$Path)
    return [System.IO.Path]::GetRelativePath($rootFull, $Path)
}

function Get-ByteCount {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        return [int64](Get-Item -LiteralPath $Path).Length
    }
    $files = Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue
    return [int64](($files | Measure-Object -Property Length -Sum).Sum)
}

function Add-Candidate {
    param(
        [string]$Source,
        [ValidateSet("File", "Directory")][string]$Kind
    )
    $sourceFull = [System.IO.Path]::GetFullPath($Source)
    if (-not (Test-Path -LiteralPath $sourceFull)) {
        return
    }
    $relative = Get-RelativePath $sourceFull
    $destination = Join-Path $backupFull $relative
    if ($sourceFull.StartsWith($backupFull.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
            [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        return
    }
    $candidates.Add([pscustomobject]@{
        Source = $sourceFull
        Destination = $destination
        Kind = $Kind
        Bytes = Get-ByteCount $sourceFull
    })
}

function Add-DirectoryIfPresent {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path -PathType Container) {
        Add-Candidate -Source $Path -Kind Directory
    }
}

$dataRoot = Join-Path $Root "data"
$historical = Join-Path $dataRoot "historical"
$vouchers = Join-Path $historical "vauchers"
$orders = Join-Path $historical "orders"

if (Test-Path -LiteralPath $vouchers -PathType Container) {
    $duplicatePattern = '^(?<base>.+)\((?<number>\d+)\)(?<extension>\.[^.]+)$'
    foreach ($file in Get-ChildItem -LiteralPath $vouchers -Recurse -File) {
        $match = [regex]::Match($file.Name, $duplicatePattern)
        if (-not $match.Success) {
            continue
        }
        $baseName = $match.Groups["base"].Value + $match.Groups["extension"].Value
        $basePath = Join-Path $file.DirectoryName $baseName
        if (Test-Path -LiteralPath $basePath -PathType Leaf) {
            Add-Candidate -Source $file.FullName -Kind File
        }
        else {
            Write-Warning "Duplicate has no base file; leaving in place: $($file.FullName)"
        }
    }
}

Add-DirectoryIfPresent (Join-Path $dataRoot "arch")
Add-DirectoryIfPresent (Join-Path $dataRoot "raw")

if (Test-Path -LiteralPath $historical -PathType Container) {
    $ordersFull = [System.IO.Path]::GetFullPath($orders)
    foreach ($file in Get-ChildItem -LiteralPath $historical -Recurse -File -Filter "*.rar") {
        $fileFull = [System.IO.Path]::GetFullPath($file.FullName)
        if ($fileFull.StartsWith($ordersFull.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
                [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        Add-Candidate -Source $file.FullName -Kind File
    }
}

$oldIndexer = Join-Path $historical "index_files.ps1"
if (Test-Path -LiteralPath $oldIndexer -PathType Leaf) {
    Add-Candidate -Source $oldIndexer -Kind File
}

$uniqueCandidates = $candidates |
    Group-Object Source |
    ForEach-Object { $_.Group | Select-Object -First 1 }
$totalBytes = [int64](($uniqueCandidates | Measure-Object -Property Bytes -Sum).Sum)
$mode = if ($Execute) { "EXECUTE" } else { "DRY RUN" }
Write-Host "Mode: $mode"

foreach ($candidate in $uniqueCandidates) {
    if ($Execute) {
        $parent = Split-Path -Parent $candidate.Destination
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        Move-Item -LiteralPath $candidate.Source -Destination $candidate.Destination
        Write-Host "Moved: $($candidate.Source) -> $($candidate.Destination)"
    }
    else {
        Write-Host "Would move: $($candidate.Source) -> $($candidate.Destination)"
    }
}

$fileCount = @($uniqueCandidates | Where-Object Kind -eq "File").Count
$directoryCount = @($uniqueCandidates | Where-Object Kind -eq "Directory").Count
Write-Host ""
Write-Host "Files: $fileCount"
Write-Host "Directories: $directoryCount"
Write-Host "Total bytes: $totalBytes"
Write-Host "Backup: $backupFull"
Write-Host "Run with -Execute to move files. Everything is reversible: items are only moved into _backup."
