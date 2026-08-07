# Downloads IBM Plex Sans woff2 files into frontend/static/css/fonts/
$dir = "frontend\static\css\fonts"
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
$weights = 400, 500, 600, 700
foreach ($w in $weights) {
    $url = "https://cdn.jsdelivr.net/npm/@fontsource/ibm-plex-sans@5.0.20/files/ibm-plex-sans-latin-$w-normal.woff2"
    $out = Join-Path $dir "ibm-plex-sans-latin-$w-normal.woff2"
    Write-Host "Downloading weight $w..."
    Invoke-WebRequest -Uri $url -OutFile $out
}
Write-Host "Done. IBM Plex Sans installed."
