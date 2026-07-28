<#
index_files.ps1 — индексирует сканы ваучеров и файлы заявок в два CSV.

Запуск (из папки, где лежат подпапки `vauchers` и `orders`):
    .\index_files.ps1

Или с явными путями:
    .\index_files.ps1 -VauchersDir "D:\data\vauchers" -OrdersDir "D:\data\orders" -OutDir "D:\data"

Результат (рядом, в -OutDir; по умолчанию текущая папка):
    vouchers_index.csv  — путь, имя, номер, суффикс p/k, буксир
    orders_index.csv    — путь, имя, направление, судно, дата, время, причал

Пришли эти два CSV обратно — по ним собирается сверка выгрузка ↔ заявки ↔ ваучеры.
#>

param(
    [string]$VauchersDir = ".\vauchers",
    [string]$OrdersDir   = ".\orders",
    [string]$OutDir      = "."
)

$ErrorActionPreference = "Stop"

function Write-CsvUtf8 {
    param($Rows, [string]$Path)
    if ($null -eq $Rows -or @($Rows).Count -eq 0) {
        # пустой файл с заголовком не создать без данных — предупреждаем
        Write-Host "  ВНИМАНИЕ: файлов не найдено — CSV не создан: $Path" -ForegroundColor Yellow
        return 0
    }
    $Rows | Export-Csv -Path $Path -NoTypeInformation -Encoding UTF8
    return @($Rows).Count
}

# ---------- Ваучеры (папка vauchers) ----------
Write-Host "Индексирую ваучеры: $VauchersDir"
$voucherRows = @()
if (Test-Path $VauchersDir) {
    $voucherRows = Get-ChildItem -Path $VauchersDir -Recurse -File -Include *.pdf,*.jpg,*.jpeg,*.png,*.tif,*.tiff |
        ForEach-Object {
            $name = $_.BaseName                       # имя без расширения, напр. 265p
            $m = [regex]::Match($name, '^(\d+)\s*([pkPK])?')
            $num = if ($m.Success) { $m.Groups[1].Value } else { '' }
            $suf = if ($m.Groups[2].Success) { $m.Groups[2].Value.ToLower() } else { '' }
            $tug = switch ($suf) { 'p' { 'БК Пионер' } 'k' { 'БК Коммунар' } default { '' } }
            [pscustomobject]@{
                Path     = $_.FullName
                File     = $_.Name
                Folder   = $_.Directory.Name
                Number   = $num
                Suffix   = $suf
                Tug      = $tug
            }
        }
} else {
    Write-Host "  ВНИМАНИЕ: папка не найдена: $VauchersDir" -ForegroundColor Yellow
}

# ---------- Заявки (папка orders) ----------
Write-Host "Индексирую заявки: $OrdersDir"
$orderRows = @()
if (Test-Path $OrdersDir) {
    $orderRows = Get-ChildItem -Path $OrdersDir -Recurse -File -Include *.pdf |
        ForEach-Object {
            $name = $_.BaseName

            # направление: Вход / Выход / Перешвартовка (учёт префикса "Re_")
            $direction = ''
            if     ($name -match '(?i)перешвартовк') { $direction = 'Перешвартовка' }
            elseif ($name -match '(?i)\bвход')       { $direction = 'Вход' }
            elseif ($name -match '(?i)\bвыход')      { $direction = 'Выход' }

            # судно: цепочка латинских слов из заглавных букв (напр. "THERESA EMPAT").
            # Начинается с буквенного слова; хвостовой день даты (" 20") отрезаем.
            $vessel = ''
            $mv = [regex]::Match($name, '([A-Z]{2,}(?:\s+[A-Z0-9]+)*)')
            if ($mv.Success) {
                $vessel = ($mv.Value -replace '\s+', ' ').Trim()
                $vessel = ($vessel -replace '\s+\d+$', '').Trim()
            }

            # дата dd.mm  и время hh_mm (в имени время пишут через "_")
            $dateRaw = ''
            $md = [regex]::Match($name, '\b(\d{1,2}\.\d{1,2})\b')
            if ($md.Success) { $dateRaw = $md.Groups[1].Value }

            # время hh_mm: разделитель только "_" или ":" (у даты разделитель ".")
            $timeRaw = ''
            $mt = [regex]::Match($name, '\b(\d{1,2})[_:](\d{2})\b')
            if ($mt.Success) { $timeRaw = "$($mt.Groups[1].Value):$($mt.Groups[2].Value)" }

            # причал: ТСС №N (может быть диапазон "ТСС №7 - ТСС №5")
            $berth = ''
            $mb = [regex]::Matches($name, '(?i)ТСС\s*№?\s*\d+')
            if ($mb.Count -gt 0) { $berth = (($mb | ForEach-Object { $_.Value }) -join ' - ') }

            [pscustomobject]@{
                Path      = $_.FullName
                File      = $_.Name
                Folder    = $_.Directory.Name
                Direction = $direction
                Vessel    = $vessel
                DateRaw   = $dateRaw
                TimeRaw   = $timeRaw
                Berth     = $berth
            }
        }
} else {
    Write-Host "  ВНИМАНИЕ: папка не найдена: $OrdersDir" -ForegroundColor Yellow
}

# ---------- Запись ----------
$vOut = Join-Path $OutDir 'vouchers_index.csv'
$oOut = Join-Path $OutDir 'orders_index.csv'
$vN = Write-CsvUtf8 -Rows $voucherRows -Path $vOut
$oN = Write-CsvUtf8 -Rows $orderRows   -Path $oOut

Write-Host ""
Write-Host "Готово."
Write-Host "  Ваучеры: $vN  -> $vOut"
Write-Host "  Заявки:  $oN  -> $oOut"

# подсказка про архивы: если заявки лежат в .rar/.zip и ещё не распакованы
$archives = @()
if (Test-Path $OrdersDir) {
    $archives = Get-ChildItem -Path $OrdersDir -Recurse -File -Include *.rar,*.zip,*.7z
}
if (@($archives).Count -gt 0) {
    Write-Host ""
    Write-Host ("ВНИМАНИЕ: в '$OrdersDir' найдено архивов: {0}. Распакуй их (PDF-заявки), затем перезапусти скрипт." -f @($archives).Count) -ForegroundColor Yellow
}
