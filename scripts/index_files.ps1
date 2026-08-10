<#
index_files.ps1 — индексирует сканы ваучеров и файлы заявок в два CSV.

Запуск (из папки, где лежат подпапки `vauchers` и `orders`):
    .\index_files.ps1

Или с явными путями:
    .\index_files.ps1 -VauchersDir "D:\data\vauchers" -OrdersDir "D:\data\orders" -OutDir "D:\data"

Заявки — это письма .eml с нейтральными именами, поэтому тема и отправитель
читаются из заголовков письма (Subject/From), а из темы разбираются судно/дата/причал.

Результат (рядом, в -OutDir; по умолчанию текущая папка):
    vouchers_index.csv  — путь, имя, номер, суффикс p/k, буксир
    orders_index.csv    — путь, имя файла, тема письма (Subject), отправитель (From),
                          направление, судно, дата, время, причал

Пришли эти два CSV обратно — по ним собирается сверка выгрузка ↔ заявки ↔ ваучеры
(колонка «Ваучер» ↔ файл скана, колонка «Заявка» ↔ тема письма .eml).
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

# Декодирование MIME encoded-words (=?charset?B/Q?...?=) в заголовках письма.
function ConvertFrom-EncodedWords {
    param([string]$Text)
    if ([string]::IsNullOrEmpty($Text)) { return '' }
    # RFC 2047: пробел между двумя соседними encoded-words не значим — убираем,
    # иначе длинная тема из нескольких кусков склеивается с лишними пробелами.
    $Text = [regex]::Replace($Text, '\?=\s+=\?', '?==?')
    $rx = [regex] '=\?([^?]+)\?([BbQq])\?([^?]*)\?='
    return $rx.Replace($Text, {
        param($m)
        $charset = $m.Groups[1].Value
        $enc     = $m.Groups[2].Value.ToUpper()
        $payload = $m.Groups[3].Value
        try { $e = [System.Text.Encoding]::GetEncoding($charset) } catch { $e = [System.Text.Encoding]::UTF8 }
        try {
            if ($enc -eq 'B') {
                $bytes = [System.Convert]::FromBase64String($payload)
            } else {
                $payload = $payload -replace '_', ' '
                $ms = New-Object System.IO.MemoryStream
                for ($i = 0; $i -lt $payload.Length; $i++) {
                    if ($payload[$i] -eq '=' -and ($i + 2) -lt $payload.Length) {
                        $ms.WriteByte([Convert]::ToByte($payload.Substring($i + 1, 2), 16))
                        $i += 2
                    } else {
                        $ms.WriteByte([byte][char]$payload[$i])
                    }
                }
                $bytes = $ms.ToArray()
            }
            return $e.GetString($bytes)
        } catch { return $m.Value }
    })
}

# Чтение заголовков Subject/From из письма (.eml или файл без расширения).
# Возвращает $null, если файл не похож на письмо (нет заголовков Subject/From).
function Get-EmlHeaders {
    param([string]$Path)
    $bytes  = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -eq 0) { return $null }
    $latin1 = [System.Text.Encoding]::GetEncoding(28591)   # 1:1 байты -> символы
    $text   = $latin1.GetString($bytes)
    $idx = $text.IndexOf("`r`n`r`n")
    if ($idx -lt 0) { $idx = $text.IndexOf("`n`n") }
    $headerText = if ($idx -ge 0) { $text.Substring(0, $idx) } else { $text }
    $lines = $headerText -split "`r?`n"
    $headers = New-Object System.Collections.Generic.List[string]
    foreach ($ln in $lines) {
        if ($ln -match '^[ \t]' -and $headers.Count -gt 0) {
            $headers[$headers.Count - 1] += ' ' + $ln.TrimStart()
        } else {
            $headers.Add($ln)
        }
    }
    $subject = ''; $from = ''
    foreach ($h in $headers) {
        if     (-not $subject -and $h -match '^(?i)Subject:\s*(.*)$') { $subject = $Matches[1] }
        elseif (-not $from    -and $h -match '^(?i)From:\s*(.*)$')    { $from    = $Matches[1] }
    }
    # не письмо: нет ни одного из ключевых заголовков
    if (-not $subject -and -not $from) { return $null }
    return [pscustomobject]@{
        Subject = (ConvertFrom-EncodedWords $subject).Trim()
        From    = (ConvertFrom-EncodedWords $from).Trim()
    }
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
    # Письма часто сохранены БЕЗ расширения, поэтому берём все файлы и определяем
    # письмо по содержимому (наличие заголовков Subject/From внутри файла).
    $skipExt = @('.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.doc', '.docx',
                 '.xls', '.xlsx', '.zip', '.rar', '.7z', '.csv')
    $orderRows = Get-ChildItem -Path $OrdersDir -Recurse -File |
        Where-Object { $skipExt -notcontains $_.Extension.ToLower() } |
        ForEach-Object {
            $filePath = $_.FullName
            # Имена нейтральные — тему и отправителя читаем из заголовков письма.
            $hdr = Get-EmlHeaders -Path $filePath
            if ($null -eq $hdr) { return }   # не письмо — пропускаем
            $subject = $hdr.Subject
            $from    = $hdr.From
            # Разбираем по теме письма (в выгрузке колонка «Заявка» = тема + ".pdf").
            $name = $subject

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
                Path      = $filePath
                File      = $_.Name
                Folder    = $_.Directory.Name
                Subject   = $subject          # тема письма = ключ к колонке «Заявка» в выгрузке
                From      = $from              # отправитель — определяет агента (Транс-Агро и др.)
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

# Диагностика, если письма не распознались: что вообще лежит в orders.
if ($oN -eq 0 -and (Test-Path $OrdersDir)) {
    Write-Host ""
    Write-Host "Диагностика '$OrdersDir' — расширения файлов (пусто = без расширения):" -ForegroundColor Yellow
    Get-ChildItem -Path $OrdersDir -Recurse -File |
        Group-Object { $_.Extension.ToLower() } |
        Sort-Object Count -Descending |
        ForEach-Object { Write-Host ("  {0,6}  '{1}'" -f $_.Count, $_.Name) }
    $sample = Get-ChildItem -Path $OrdersDir -Recurse -File | Select-Object -First 1
    if ($sample) {
        Write-Host ""
        Write-Host ("Первые строки файла-примера ({0}):" -f $sample.Name) -ForegroundColor Yellow
        (Get-Content -LiteralPath $sample.FullName -TotalCount 8 -ErrorAction SilentlyContinue) |
            ForEach-Object { Write-Host ("  | " + $_) }
    }
    Write-Host ""
    Write-Host "Пришли этот вывод — подстрою разбор под реальный формат писем." -ForegroundColor Yellow
}

# подсказка про архивы: если заявки лежат в .rar/.zip и ещё не распакованы
$archives = @()
if (Test-Path $OrdersDir) {
    $archives = Get-ChildItem -Path $OrdersDir -Recurse -File -Include *.rar,*.zip,*.7z
}
if (@($archives).Count -gt 0) {
    Write-Host ""
    Write-Host ("ВНИМАНИЕ: в '$OrdersDir' найдено архивов: {0}. Распакуй их, затем перезапусти скрипт." -f @($archives).Count) -ForegroundColor Yellow
}
