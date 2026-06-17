param(
    # Timestamp-fixed source: analyze_preprocess.py FFV1 output with the broken
    # stray tail frame trimmed off (clean 144,176-frame 29.97fps CFR, A/V synced).
    # ASCII filename on purpose -- PowerShell 5.1 misreads non-ASCII .ps1 literals.
    [string]$Source   = "outputs\kujiramichi_0150.clean.mkv",
    [string]$RunName  = "kujiramichi_0150_fhd_tiny",
    # fast = Tiny (Fast) model (~1.5 s/frame on the evo-x2), balanced = Full (Best).
    [string]$Profile  = "fast",
    # Larger chunks amortize per-chunk overhead and shrink the 32-frame overlap
    # re-processing fraction (~29% at 144 -> ~11% at 288). 107GB unified mem fits it.
    [int]$ChunkFrames = 288,
    [int]$Port        = 9123,
    # Full HD pillarbox geometry for the 4:3 (DAR) source: scale the 2x output to
    # 1440x1080 (restores 4:3 from square-pixel 1440x960) then pad to 1920x1080.
    [int]$FhdBitrateM = 16,
    [string]$FfmpegBin = "C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
)

# Tolerate native-command stderr under output redirection (PowerShell 5.1).
$ErrorActionPreference = "Continue"
$env:Path = "$FfmpegBin;" + $env:Path
$env:COMFYUI_SERVER = "http://127.0.0.1:$Port"

$python = ".\.venv312\Scripts\python.exe"
$runDir = ".\runs\$RunName"
$finalMp4 = "$runDir\$RunName.final.mp4"
$fhdMp4   = "$runDir\$RunName.fullhd.mp4"

function Test-Comfy {
    try { return (Invoke-WebRequest "http://127.0.0.1:$Port/system_stats" -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200 }
    catch { return $false }
}

function Ensure-Comfy {
    if (Test-Comfy) { return $true }
    Write-Host "[supervisor] starting ComfyUI on port $Port ..."
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile","-ExecutionPolicy","Bypass","-Command",
        ". '.\scripts\run-comfyui.ps1' -Port $Port -Backend rocm *>> '.\comfyui-$Port.out.log'"
    ) -WindowStyle Hidden | Out-Null
    $deadline = (Get-Date).AddSeconds(300)
    while ((Get-Date) -lt $deadline) {
        if (Test-Comfy) { Write-Host "[supervisor] ComfyUI is ready."; return $true }
        Start-Sleep -Seconds 5
    }
    Write-Host "[supervisor] ComfyUI did not become ready in time."
    return $false
}

Write-Host "[supervisor] === FlashVSR full Full-HD run ==="
Write-Host "[supervisor] source=$Source run=$RunName profile=$Profile"

# 1+2. Drive the resumable chunk upscaler, restarting it on any transient failure.
#      Completed chunks are skipped (raw_chunks/*.mp4 already on disk), so each
#      retry resumes instead of starting over.
$maxAttempts = 500
for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    if (-not (Ensure-Comfy)) { Start-Sleep -Seconds 30; continue }
    Write-Host "[supervisor] chunk run attempt $attempt ..."
    & $python ".\scripts\run_flashvsr_chunks.py" $Source --run-name $RunName --profile $Profile --chunk-frames $ChunkFrames
    if ($LASTEXITCODE -eq 0 -and (Test-Path $finalMp4)) {
        Write-Host "[supervisor] chunk run complete -> $finalMp4"
        break
    }
    Write-Host "[supervisor] chunk run exited $LASTEXITCODE; retrying in 30s (attempt $attempt/$maxAttempts)"
    Start-Sleep -Seconds 30
}

if (-not (Test-Path $finalMp4)) {
    Write-Host "[supervisor] ERROR: $finalMp4 was never produced. Aborting before final framing."
    exit 1
}

# 3. Final Full HD framing: 1440x960 -> 1440x1080 (4:3) -> pad to 1920x1080.
Write-Host "[supervisor] final Full-HD framing -> $fhdMp4"
& "$FfmpegBin\ffmpeg.exe" -y -hide_banner -loglevel error -i $finalMp4 `
    -vf "scale=1440:1080:flags=lanczos,setsar=1,pad=1920:1080:240:0:black" `
    -c:v h264_amf -quality quality -b:v "$($FhdBitrateM)M" -pix_fmt yuv420p `
    -c:a copy $fhdMp4

if (Test-Path $fhdMp4) {
    Write-Host "[supervisor] DONE -> $fhdMp4"
} else {
    Write-Host "[supervisor] ERROR: final framing failed."
    exit 1
}
