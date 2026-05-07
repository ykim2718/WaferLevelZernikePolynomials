$ErrorActionPreference = "Stop"

# 1. Build time string: YYYY-MM-DD_HH-MM-SS
$now = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$commitMsg = "Auto commit at $now"
Write-Host "Commit message: $commitMsg"

# 2. Initialize Git repo if .git does not exist
if (-not (Test-Path .git)) {
    Write-Host "Initializing new git repository..."
    git init | Out-Null
    git branch -M main
}

# 3. Detect current branch
$branch = (& git branch --show-current).Trim()
if ([string]::IsNullOrEmpty($branch)) {
    Write-Host "No branch detected. Creating main branch..."
    git checkout -b main
    $branch = "main"
}
Write-Host "Current branch: $branch"

# 4. Stage all changes
git add .

# 5. Commit only if there are staged changes
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "No changes to commit."
} else {
    git commit -m $commitMsg
}

# 6. Add remote origin only if it does not exist
$remoteUrl = "https://github.com/ykim2718/WaferLevelZernikePolynomials.git"
$remotes = & git remote
if (-not ($remotes -split "`n" | Where-Object { $_.Trim() -eq "origin" })) {
    Write-Host "Adding remote origin..."
    git remote add origin $remoteUrl
}

# 7. Push to remote
Write-Host "Pushing to origin $branch..."
git push -u origin $branch
Write-Host "Done."
