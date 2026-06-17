param(
    [string]$VenvPath = ".venv312",
    [int]$Port = 8188,
    # auto | cuda | rocm | cpu  -- selects runtime GPU env vars.
    [string]$Backend = "auto",
    [string]$Gfx = ""
)

$ErrorActionPreference = "Stop"
$python = Join-Path $VenvPath "Scripts\python.exe"

function Get-GpuVendor {
    try {
        $names = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ';'
    } catch {
        $names = ''
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return 'nvidia' }
    if ($names -match '(?i)nvidia|geforce|rtx|quadro|tesla') { return 'nvidia' }
    if ($names -match '(?i)amd|radeon|instinct') { return 'amd' }
    return 'cpu'
}

if ($Backend -eq "auto") {
    $Backend = Get-GpuVendor
    if ($Backend -eq "nvidia") { $Backend = "cuda" }
}

if ($Backend -eq "rocm" -or $Backend -eq "amd") {
    # ROCm-on-Windows runtime tuning for AMD (Strix Halo / Radeon 8060S).
    # CUDA_VISIBLE_DEVICES hides the AMD GPU from HIP -- it must be cleared.
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    $env:HIP_VISIBLE_DEVICES = "0"
    $env:GPU_MAX_HEAP_SIZE = "100"
    $env:GPU_MAX_ALLOC_PERCENT = "100"
    $env:AMD_LOG_LEVEL = "0"
    if ($Gfx -ne "") { $env:HSA_OVERRIDE_GFX_VERSION = $Gfx }
    Write-Host "[run] ROCm backend (HIP_VISIBLE_DEVICES=0)"
}

& $python ".\ComfyUI\main.py" --listen 127.0.0.1 --port $Port
