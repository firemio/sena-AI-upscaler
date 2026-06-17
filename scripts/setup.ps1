param(
    [string]$PythonVersion = "3.12",
    [string]$VenvPath = ".venv312",
    # auto | cuda | rocm | cpu  -- which PyTorch backend to install.
    [string]$Backend = "auto",
    # gfx target for the AMD ROCm-on-Windows wheels (Ryzen AI Max+ 395 = gfx1151).
    [string]$Gfx = "gfx1151",
    # Override the ROCm wheel index if AMD moves it. {gfx} is substituted.
    [string]$RocmIndexUrl = "https://rocm.nightlies.amd.com/v2/{gfx}/",
    [string]$CudaIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-GitClone {
    param(
        [string]$RepoUrl,
        [string]$TargetDir
    )
    if (-not (Test-Path $TargetDir)) {
        git clone $RepoUrl $TargetDir
    }
}

function Get-GpuVendor {
    # Detect the GPU vendor before any GPU-enabled torch exists.
    try {
        $names = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ';'
    } catch {
        $names = ''
    }
    if (Test-Command nvidia-smi) { return 'nvidia' }
    if ($names -match '(?i)nvidia|geforce|rtx|quadro|tesla') { return 'nvidia' }
    if ($names -match '(?i)amd|radeon|instinct') { return 'amd' }
    return 'cpu'
}

if (-not (Test-Command ffmpeg)) {
    winget install --id Gyan.FFmpeg --exact --accept-source-agreements --accept-package-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $pyLauncher) {
    winget install --id Python.Python.$($PythonVersion.Replace('.', '')) --exact --accept-source-agreements --accept-package-agreements
}

$pyList = & py -0p 2>$null
if (-not ($pyList -match "3\.12")) {
    winget install --id Python.Python.3.12 --exact --accept-source-agreements --accept-package-agreements
}

if (-not (Test-Path ".\ComfyUI")) {
    git clone https://github.com/comfyanonymous/ComfyUI.git
}

New-Item -ItemType Directory -Force -Path ".\ComfyUI\custom_nodes" | Out-Null

Ensure-GitClone "https://github.com/Comfy-Org/ComfyUI-Manager.git" ".\ComfyUI\custom_nodes\ComfyUI-Manager"
Ensure-GitClone "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" ".\ComfyUI\custom_nodes\ComfyUI-VideoHelperSuite"
Ensure-GitClone "https://github.com/1038lab/ComfyUI-FlashVSR.git" ".\ComfyUI\custom_nodes\ComfyUI-FlashVSR"

# Deploy our custom AMD AMF encoder formats into VideoHelperSuite (the node is a
# fresh clone above, so these tracked files must be copied in after cloning).
$vhsFormats = ".\ComfyUI\custom_nodes\ComfyUI-VideoHelperSuite\video_formats"
if (Test-Path $vhsFormats) {
    Copy-Item ".\scripts\video_formats\*.json" $vhsFormats -Force
}

if (-not (Test-Path $VenvPath)) {
    & py -3.12 -m venv $VenvPath
}

$python = Join-Path $VenvPath "Scripts\python.exe"
$pip = Join-Path $VenvPath "Scripts\pip.exe"

& $python -m pip install --upgrade pip setuptools wheel

# --- PyTorch backend selection (NVIDIA CUDA vs AMD ROCm vs CPU) ---
if ($Backend -eq "auto") {
    $Backend = Get-GpuVendor
    if ($Backend -eq "nvidia") { $Backend = "cuda" }
}
Write-Host "[setup] Installing PyTorch backend: $Backend"

switch ($Backend) {
    "cuda" {
        & $pip install torch torchvision torchaudio --index-url $CudaIndexUrl
    }
    "rocm" {
        # ROCm-on-Windows wheels for AMD (Strix Halo / Radeon 8060S = gfx1151).
        # If this index breaks (e.g. hipsparselt dependency issues), fall back to
        # scottt's self-contained wheels documented in README.md.
        $index = $RocmIndexUrl.Replace("{gfx}", $Gfx)
        Write-Host "[setup] ROCm wheel index: $index"
        & $pip install --pre torch torchvision torchaudio --index-url $index
    }
    "cpu" {
        Write-Host "[setup] WARNING: CPU-only torch -- FlashVSR will be very slow."
        & $pip install torch torchvision torchaudio
    }
    default {
        throw "Unknown -Backend '$Backend' (expected auto|cuda|rocm|cpu)"
    }
}

# Verify the GPU is actually visible to torch.
& $python -c "import torch; print('[setup] torch', torch.__version__, 'cuda/hip available:', torch.cuda.is_available())"

& $pip install -r ".\ComfyUI\requirements.txt"

$reqFiles = @(
    ".\ComfyUI\custom_nodes\ComfyUI-Manager\requirements.txt",
    ".\ComfyUI\custom_nodes\ComfyUI-VideoHelperSuite\requirements.txt",
    ".\ComfyUI\custom_nodes\ComfyUI-FlashVSR\requirements.txt"
)

foreach ($req in $reqFiles) {
    if (Test-Path $req) {
        & $pip install -r $req
    }
}

# Report the resolved hardware profile the run scripts will use.
& $python ".\scripts\hardware.py"
