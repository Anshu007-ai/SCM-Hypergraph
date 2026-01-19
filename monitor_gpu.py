"""
GPU Memory Monitor - Real-time monitoring during training
Tracks memory usage, utilization, and thermal info
"""

import torch
import subprocess
import time
import sys
from datetime import datetime

def get_gpu_stats():
    """Get GPU stats using nvidia-smi"""
    try:
        # Get comprehensive GPU info
        cmd = [
            'nvidia-smi',
            '--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return None
        
        line = result.stdout.strip()
        parts = [x.strip() for x in line.split(',')]
        
        return {
            'timestamp': parts[0],
            'name': parts[1],
            'memory_total': int(parts[2]),
            'memory_used': int(parts[3]),
            'memory_free': int(parts[4]),
            'utilization': int(parts[5]),
            'temp': int(parts[6])
        }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None

def print_header():
    """Print monitor header"""
    print("\n" + "="*100)
    print("GPU MEMORY MONITOR - Real-time GPU Usage Tracking")
    print("="*100)
    print(f"{'Time':<12} {'Memory Used':<15} {'Memory Free':<15} {'Usage %':<12} {'GPU %':<10} {'Temp':<10}")
    print("-"*100)

def print_stats(stats):
    """Print formatted GPU stats"""
    if stats is None:
        print("[ERROR] Could not retrieve GPU stats")
        return
    
    used_gb = stats['memory_used'] / 1024
    free_gb = stats['memory_free'] / 1024
    total_gb = stats['memory_total'] / 1024
    usage_pct = (stats['memory_used'] / stats['memory_total']) * 100
    
    # Color coding based on usage
    if usage_pct > 90:
        status = "[CRITICAL]"
        color = "\033[91m"  # Red
    elif usage_pct > 75:
        status = "[WARNING]"
        color = "\033[93m"  # Yellow
    else:
        status = "[OK]"
        color = "\033[92m"  # Green
    
    reset_color = "\033[0m"
    
    time_str = datetime.now().strftime("%H:%M:%S")
    
    print(
        f"{color}{time_str:<12} "
        f"{used_gb:>6.2f}GB/{total_gb:>5.2f}GB    "
        f"{free_gb:>6.2f}GB         "
        f"{usage_pct:>6.1f}%       "
        f"{stats['utilization']:>5d}%     "
        f"{stats['temp']:>3d}C{reset_color}"
    )

def main():
    """Main monitoring loop"""
    print_header()
    
    try:
        # Initial stats
        stats = get_gpu_stats()
        if stats:
            print(f"GPU: {stats['name']}")
            print(f"Total Memory: {stats['memory_total']/1024:.2f} GB\n")
        
        print("Monitoring GPU (press Ctrl+C to stop)...\n")
        
        while True:
            stats = get_gpu_stats()
            if stats:
                print_stats(stats)
            
            time.sleep(1)  # Update every second
    
    except KeyboardInterrupt:
        print("\n" + "="*100)
        print("Monitoring stopped")
        print("="*100 + "\n")

if __name__ == "__main__":
    # Check if nvidia-smi is available
    try:
        subprocess.run(['nvidia-smi', '--version'], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("ERROR: nvidia-smi not found!")
        print("Please install NVIDIA drivers: https://www.nvidia.com/drivers")
        sys.exit(1)
    
    main()
