# Sourced by build.sh and run.sh: headless, RADV only, fresh compiles (so disassembly is captured), low priority,
# every process under a 3 GiB memory cap with no swap (a runaway is killed instead of taking the desktop's memory).
unset DISPLAY WAYLAND_DISPLAY
export SDL_VIDEO_DRIVER=offscreen
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
export MESA_SHADER_CACHE_DISABLE=true
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCK="$HERE/../gpu-bench.lock"
capped() { systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 -- nice -n 10 "$@"; }
