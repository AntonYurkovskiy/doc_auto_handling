# Аудит локальных файлов проекта: только ЧИТАЕТ, ничего не удаляет и не двигает.
# Показывает структуру папок, где лежат данные/скрипты/репозиторий, дубли и мусор.
# Запуск:
#   powershell -ExecutionPolicy Bypass -File .\audit_files.ps1 -Root "E:\projects\doc_auto_handling"
param(
    [string]$Root   = ".",
    [string]$OutFile = ".\audit_report.txt"
)

$ErrorActionPreference = 'Stop'
$lines = New-Object System.Collections.Generic.List[string]
function Add-Line { param([string]$s = ''); $lines.Add($s); Write-Host $s }

$rootItem = Get-Item -LiteralPath $Root
Add-Line "==================================================================="
Add-Line "АУДИТ: $($rootItem.FullName)"
Add-Line "Дата:  $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Add-Line "==================================================================="

$all = Get-ChildItem -LiteralPath $Root -Recurse -File -Force -ErrorAction SilentlyContinue
Add-Line ""
Add-Line ("Всего файлов: {0}" -f @($all).Count)
$totalMb = [math]::Round((($all | Measure-Object Length -Sum).Sum) / 1MB, 1)
Add-Line ("Общий размер: {0} МБ" -f $totalMb)

# ---------- 1. Дерево папок: файлов и размер по каждой папке ----------
Add-Line ""
Add-Line "------------------------------------------------------------------"
Add-Line "1) ПАПКИ (кол-во файлов / размер) — только прямые файлы папки"
Add-Line "------------------------------------------------------------------"
$byDir = $all | Group-Object DirectoryName | Sort-Object Name
foreach ($g in $byDir) {
    $rel = $g.Name.Replace($rootItem.FullName, '.').TrimStart('\')
    if ($rel -eq '') { $rel = '.' }
    $mb = [math]::Round((($g.Group | Measure-Object Length -Sum).Sum) / 1MB, 2)
    Add-Line ("  {0,6} файл(ов)  {1,8} МБ   {2}" -f $g.Count, $mb, $rel)
}

# ---------- 2. Расширения ----------
Add-Line ""
Add-Line "------------------------------------------------------------------"
Add-Line "2) РАСШИРЕНИЯ (сколько файлов какого типа; пусто = без расширения)"
Add-Line "------------------------------------------------------------------"
$all | Group-Object { $_.Extension.ToLower() } | Sort-Object Count -Descending |
    ForEach-Object { Add-Line ("  {0,6}  '{1}'" -f $_.Count, $_.Name) }

# ---------- 3. Репозитории git (папки .git) ----------
Add-Line ""
Add-Line "------------------------------------------------------------------"
Add-Line "3) РЕПОЗИТОРИИ (папки .git — где лежит рабочая копия кода)"
Add-Line "------------------------------------------------------------------"
$gits = Get-ChildItem -LiteralPath $Root -Recurse -Directory -Force -Filter '.git' -ErrorAction SilentlyContinue
if (@($gits).Count -eq 0) { Add-Line "  (не найдено)" }
foreach ($g in $gits) {
    $repo = Split-Path $g.FullName -Parent
    Add-Line ("  РЕПО: {0}" -f $repo)
}

# ---------- 4. Ключевые файлы проекта ----------
Add-Line ""
Add-Line "------------------------------------------------------------------"
Add-Line "4) КЛЮЧЕВЫЕ ФАЙЛЫ (выгрузка, индексы, скрипты, правила, БД)"
Add-Line "------------------------------------------------------------------"
$keyPatterns = @('*extraction*.xls','*extraction*.rar','vouchers_index.csv','orders_index.csv',
                 'reconciled_dataset.csv','reconcile_report.md','*.ps1','*.odt','*.db','*.sqlite*',
                 'index_files.ps1','reconcile.py','eml_parser.py')
foreach ($pat in $keyPatterns) {
    $hits = $all | Where-Object { $_.Name -like $pat }
    foreach ($h in $hits) {
        $rel = $h.FullName.Replace($rootItem.FullName, '.')
        Add-Line ("  [{0}]  {1}  ({2:N0} байт, {3})" -f $pat, $rel, $h.Length, $h.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))
    }
}

# ---------- 5. Дубли по имени файла ----------
Add-Line ""
Add-Line "------------------------------------------------------------------"
Add-Line "5) ДУБЛИ ПО ИМЕНИ (один и тот же файл лежит в нескольких местах)"
Add-Line "------------------------------------------------------------------"
$dups = $all | Group-Object Name | Where-Object { $_.Count -gt 1 } | Sort-Object Count -Descending
Add-Line ("  Имён с дублями: {0}" -f @($dups).Count)
foreach ($d in ($dups | Select-Object -First 40)) {
    Add-Line ("  x{0}  {1}" -f $d.Count, $d.Name)
    foreach ($f in $d.Group) {
        Add-Line ("        {0}" -f $f.FullName.Replace($rootItem.FullName, '.'))
    }
}
if (@($dups).Count -gt 40) { Add-Line ("  ... ещё {0} имён с дублями" -f (@($dups).Count - 40)) }

# ---------- 6. Копии вида "имя(2).pdf" ----------
Add-Line ""
Add-Line "------------------------------------------------------------------"
Add-Line "6) КОПИИ вида 'имя(2).pdf', 'имя - копия' и т.п."
Add-Line "------------------------------------------------------------------"
$copies = $all | Where-Object { $_.Name -match '\(\d+\)' -or $_.Name -match '(?i)копия|copy' }
Add-Line ("  Найдено копий: {0}" -f @($copies).Count)
foreach ($c in ($copies | Select-Object -First 40)) {
    Add-Line ("  {0}" -f $c.FullName.Replace($rootItem.FullName, '.'))
}
if (@($copies).Count -gt 40) { Add-Line ("  ... ещё {0}" -f (@($copies).Count - 40)) }

# ---------- 7. Архивы (возможно, не распакованы) ----------
Add-Line ""
Add-Line "------------------------------------------------------------------"
Add-Line "7) АРХИВЫ (.rar/.zip/.7z — возможно ещё не распакованы)"
Add-Line "------------------------------------------------------------------"
$arch = $all | Where-Object { $_.Extension.ToLower() -in @('.rar','.zip','.7z') }
Add-Line ("  Найдено архивов: {0}" -f @($arch).Count)
foreach ($a in ($arch | Select-Object -First 40)) {
    Add-Line ("  {0}" -f $a.FullName.Replace($rootItem.FullName, '.'))
}

# ---------- Запись ----------
$enc = New-Object System.Text.UTF8Encoding($true)   # UTF-8 с BOM
[System.IO.File]::WriteAllText($OutFile, ($lines -join "`r`n"), $enc)
Add-Line ""
Add-Line ("Отчёт сохранён: {0}" -f $OutFile)
