# Επιβεβαίωση ότι τα σωστά αρχεία εγκαταστάθηκαν
Write-Host "=== Ελεγχος βασικων αρχειων ===" -ForegroundColor Cyan
Write-Host ""

$checks = @(
    @{ File = "frontend\templates\base.html"; Pattern = 'background: var\(--brand\)'; Message = "Navy header background στο base.html" },
    @{ File = "frontend\templates\base.html"; Pattern = 'Human-in-the-Loop Gene Selector'; Message = "Πληρες ονομα στο header" },
    @{ File = "frontend\static\css\styles.css"; Pattern = 'ea580c'; Message = "Coral χρωμα στο CSS" },
    @{ File = "frontend\static\img\hero.svg"; Pattern = 'DNA'; Message = "Hero illustration υπαρχει" },
    @{ File = "frontend\templates\step1.html"; Pattern = 'hero.svg'; Message = "Step 1 χρησιμοποιει το hero" },
    @{ File = "frontend\templates\step2.html"; Pattern = 'dna-mini.svg'; Message = "Step 2 ΔΕΝ πρεπει να εχει mini-DNA"; ShouldMatch = $false }
)

$allPass = $true
foreach ($check in $checks) {
    if (-not (Test-Path $check.File)) {
        Write-Host "[ΑΠΟΥΣΙΑΖΕΙ] $($check.File)" -ForegroundColor Red
        $allPass = $false
        continue
    }
    $found = (Select-String -Path $check.File -Pattern $check.Pattern -Quiet)
    $shouldMatch = if ($check.ContainsKey('ShouldMatch')) { $check.ShouldMatch } else { $true }
    if ($found -eq $shouldMatch) {
        Write-Host "[OK]      $($check.Message)" -ForegroundColor Green
    } else {
        Write-Host "[ΑΠΟΤΥΧΙΑ] $($check.Message)" -ForegroundColor Red
        $allPass = $false
    }
}

Write-Host ""
if ($allPass) {
    Write-Host "Ολα καλα! Ξεκινα την εφαρμογη σε incognito window." -ForegroundColor Green
} else {
    Write-Host "Καποια αρχεια δεν εγκαταστάθηκαν σωστα. Ξαναδοκιμασε το unzip." -ForegroundColor Yellow
}
