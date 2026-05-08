$ErrorActionPreference = "Stop"

# Local Python interpreter (no trailing space inside the quotes)
$python = "c:/Y/anaconda3/python.exe"

# 1. Clean previous build artifacts (prevents twine from trying to
#    re-upload old wheels/sdists that PyPI would reject as duplicates).
if (Test-Path dist) {
    Write-Host "Cleaning dist\..."
    Remove-Item dist\* -Force -Recurse
}

# 2. Ensure build + twine are up to date
& $python -m pip install --upgrade build twine
if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }

# 3. Build sdist + wheel
& $python -m build
if ($LASTEXITCODE -ne 0) { throw "build failed (exit $LASTEXITCODE)" }

# 4. Upload to PyPI
& $python -m twine upload dist/*
if ($LASTEXITCODE -ne 0) { throw "twine upload failed (exit $LASTEXITCODE)" }

Write-Host "Done."
