# Demo runner. Robust against the caller's current working directory:
# always installs from the project root (where pyproject.toml lives) and
# runs the three stages from this script's own folder so config.json /
# points_13.json resolve correctly.
Clear-Host

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot

# Stage 0: editable install of wlzpoly — skip if already importable.
# (Editable install means edits to src/wlzpoly/*.py are picked up without
#  reinstalling, so we only need to install once per environment.)
python -c "import wlzpoly" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "wlzpoly already installed — skipping install."
} else {
    Write-Host "Installing wlzpoly (editable) from $projectRoot ..."
    python -m pip install -e $projectRoot
}

# Anchor working directory to examples/ so relative paths work
Set-Location $PSScriptRoot

# Stage 1: synthesize wafers      -> examples/samples/
python generate_samples.py `
    --working_folder . `
    --config_json config.json `
    --wafer_points points_13.json `
    --noise_sigma 5.0 `
    --seed 42 `
    --n_drift 30 `
    --output_folder ./samples

# Stage 2: fit coefficients       -> examples/decomposition/
python -m wlzpoly.decompose `
    --working_folder . `
    --wafer_points ./samples/points_13.json `
    --target_file ./samples/target_file.csv `
    --n_terms 9 `
    --output_folder ./decomposition `
    --solver lsq `
    --lam 0.01 `
    --coordinate cartesian

# Stage 3: compare vs truth       -> examples/verification/
python -m wlzpoly.verify `
    --working_folder . `
    --wafer_points ./samples/points_13.json `
    --target_file ./samples/target_file.csv `
    --ground_truth_file ./samples/ground_truth.csv `
    --n_terms 9 `
    --output_folder ./verification `
    --solver lsq ridge `
    --coordinate cartesian
