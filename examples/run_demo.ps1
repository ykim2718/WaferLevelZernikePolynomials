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
    --config_json ./configuration/config.json `
    --wafer_points ./configuration/points_13.json `
    --noise_sigma 5.0 `
    --seed 42 `
    --n_drift 30 `
    --output_folder ./1_samples

# Stage 2a: LSQ fit                -> examples/2_decomposition/decomposed_targets_lsq.csv
python -m wlzpoly.decompose `
    --working_folder . `
    --wafer_points ./1_samples/points_13.json `
    --input_file ./1_samples/target_file.csv `
    --output_folder ./2_decomposition `
    --output_file decomposed_targets_lsq.csv `
    --n_terms 9 `
    --solver lsq `
    --coordinate cartesian

# Stage 2b: Ridge fit with LOOCV   -> examples/2_decomposition/decomposed_targets_ridge.csv
python -m wlzpoly.decompose `
    --working_folder . `
    --wafer_points ./1_samples/points_13.json `
    --input_file ./1_samples/target_file.csv `
    --output_folder ./2_decomposition `
    --output_file decomposed_targets_ridge.csv `
    --n_terms 9 `
    --solver ridge `
    --auto_lam `
    --loocv_ref first_wafer `
    --coordinate cartesian

# Stage 3: compare both vs truth   -> examples/3_verification/
python -m wlzpoly.verify `
    --decomposed_lsq_file ./2_decomposition/decomposed_targets_lsq.csv `
    --decomposed_ridge_file ./2_decomposition/decomposed_targets_ridge.csv `
    --ground_truth_file ./1_samples/ground_truth.csv `
    --n_terms 9 `
    --output_folder ./3_verification

# Stage 4a: LSQ reconstruction      -> examples/4_reconstruction/reconstructed_lsq.csv
python -m wlzpoly.reconstruct `
    --input_folder . `
    --wafer_point_json ./1_samples/points_13.json `
    --decomposed_file ./2_decomposition/decomposed_targets_lsq.csv `
    --output_folder ./4_reconstruction `
    --output_file reconstructed_lsq.csv `
    --n_terms 9 `
    --col_wafer_id id `
    --coordinate cartesian

# Stage 4b: Ridge reconstruction    -> examples/4_reconstruction/reconstructed_ridge.csv
python -m wlzpoly.reconstruct `
    --input_folder . `
    --wafer_point_json ./1_samples/points_13.json `
    --decomposed_file ./2_decomposition/decomposed_targets_ridge.csv `
    --output_folder ./4_reconstruction `
    --output_file reconstructed_ridge.csv `
    --n_terms 9 `
    --col_wafer_id id `
    --coordinate cartesian
