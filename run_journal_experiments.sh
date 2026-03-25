#!/bin/bash

# HT-HGNN Journal Experiments Runner
#
# This script runs the complete experimental suite needed for journal submission.
# It executes all experiments in sequence with proper logging and organization.
#
# Usage: ./run_journal_experiments.sh
#        or: bash run_journal_experiments.sh

set -e  # Exit on any error

# Configuration
PYTHON_CMD="python"
MAIN_SCRIPT="train.py"
RESULTS_DIR="journal_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="$RESULTS_DIR/journal_experiments_$TIMESTAMP"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to estimate time remaining
print_time_estimate() {
    echo -e "${YELLOW}[TIME ESTIMATE]${NC} This experiment may take $1"
}

# Setup
print_header "HT-HGNN JOURNAL EXPERIMENTS SUITE"
echo "Starting comprehensive experimental evaluation for journal submission"
echo "Timestamp: $(date)"
echo "Results directory: $EXPERIMENT_DIR"
echo ""

# Check prerequisites
print_info "Checking prerequisites..."

if ! command_exists python; then
    print_error "Python not found. Please install Python 3.7+ and ensure it's in PATH."
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    print_error "Main training script ($MAIN_SCRIPT) not found in current directory."
    exit 1
fi

# Check for required Python packages
print_info "Verifying Python dependencies..."
python -c "import torch, numpy, pandas, sklearn, matplotlib, seaborn" 2>/dev/null || {
    print_warning "Some Python dependencies may be missing. The experiments will continue but may fail."
    echo "Required packages: torch, numpy, pandas, scikit-learn, matplotlib, seaborn"
    echo ""
}

# Create results directory
mkdir -p "$EXPERIMENT_DIR"
print_success "Created experiment directory: $EXPERIMENT_DIR"

# Save system information
print_info "Collecting system information..."
{
    echo "HT-HGNN Journal Experiments"
    echo "=========================="
    echo "Start time: $(date)"
    echo "Python version: $(python --version)"
    echo "System: $(uname -a)"
    echo "Working directory: $(pwd)"
    echo ""
} > "$EXPERIMENT_DIR/system_info.txt"

# Define experiment configurations
declare -a experiments=(
    "1:Core Training and Evaluation:40-50 minutes:--split_mode temporal --gap_hours 72 --moo_mode full --ssl_temperature 0.1 --attention_type structural"
    "2:MOO Ablation Study:60-80 minutes:--split_mode temporal --gap_hours 72 --moo_mode full --ssl_temperature 0.1 --attention_type structural --run_ablations"
    "3:SSL Temperature Sensitivity:30-40 minutes:--split_mode temporal --gap_hours 72 --moo_mode full --attention_type structural --run_ssl_sweep"
    "4:Attention Mechanism Comparison:45-60 minutes:--split_mode temporal --gap_hours 72 --moo_mode full --ssl_temperature 0.1 --run_attention_cmp"
    "5:Transfer Learning Experiments:50-70 minutes:--split_mode temporal --gap_hours 72 --moo_mode full --ssl_temperature 0.1 --attention_type structural --run_transfer_exp"
)

# Initialize experiment tracking
total_experiments=${#experiments[@]}
completed_experiments=0
failed_experiments=0
start_time=$(date +%s)

print_info "Planning to run $total_experiments experiments"
echo ""

# Function to run individual experiment
run_experiment() {
    local exp_info="$1"
    local exp_num=$(echo "$exp_info" | cut -d: -f1)
    local exp_name=$(echo "$exp_info" | cut -d: -f2)
    local exp_time=$(echo "$exp_info" | cut -d: -f3)
    local exp_args=$(echo "$exp_info" | cut -d: -f4)

    local exp_dir="$EXPERIMENT_DIR/exp_$exp_num"
    mkdir -p "$exp_dir"

    print_header "EXPERIMENT $exp_num: $exp_name"
    print_time_estimate "$exp_time"
    echo "Arguments: $exp_args"
    echo "Output directory: $exp_dir"
    echo ""

    # Run experiment with timeout (3 hours max)
    local cmd="$PYTHON_CMD $MAIN_SCRIPT $exp_args --output_dir $exp_dir"
    local log_file="$exp_dir/experiment.log"

    print_info "Executing: $cmd"
    print_info "Log file: $log_file"

    if timeout 10800 $cmd > "$log_file" 2>&1; then
        print_success "Experiment $exp_num completed successfully"
        ((completed_experiments++))

        # Save experiment metadata
        {
            echo "Experiment: $exp_name"
            echo "Number: $exp_num"
            echo "Status: SUCCESS"
            echo "Arguments: $exp_args"
            echo "Completion time: $(date)"
            echo "Duration: $exp_time"
        } > "$exp_dir/metadata.txt"

    else
        local exit_code=$?
        print_error "Experiment $exp_num failed (exit code: $exit_code)"
        ((failed_experiments++))

        # Save failure metadata
        {
            echo "Experiment: $exp_name"
            echo "Number: $exp_num"
            echo "Status: FAILED"
            echo "Exit code: $exit_code"
            echo "Arguments: $exp_args"
            echo "Failure time: $(date)"
        } > "$exp_dir/metadata.txt"

        # Show last few lines of log for debugging
        print_error "Last 10 lines of log file:"
        if [ -f "$log_file" ]; then
            tail -n 10 "$log_file" | sed 's/^/  /'
        fi
    fi

    echo ""
}

# Run all experiments
for exp_info in "${experiments[@]}"; do
    run_experiment "$exp_info"
done

# Final summary
end_time=$(date +%s)
total_duration=$((end_time - start_time))
hours=$((total_duration / 3600))
minutes=$(((total_duration % 3600) / 60))

print_header "JOURNAL EXPERIMENTS COMPLETE"
echo "Total experiments: $total_experiments"
echo "Completed successfully: $completed_experiments"
echo "Failed: $failed_experiments"
echo "Total duration: ${hours}h ${minutes}m"
echo ""

# Generate comprehensive summary
summary_file="$EXPERIMENT_DIR/journal_experiments_summary.txt"
{
    echo "HT-HGNN Journal Experiments Summary"
    echo "=================================="
    echo "Execution date: $(date)"
    echo "Total experiments: $total_experiments"
    echo "Completed: $completed_experiments"
    echo "Failed: $failed_experiments"
    echo "Total duration: ${hours}h ${minutes}m"
    echo "Results directory: $EXPERIMENT_DIR"
    echo ""
    echo "Individual Experiments:"
    echo "======================"

    for exp_info in "${experiments[@]}"; do
        local exp_num=$(echo "$exp_info" | cut -d: -f1)
        local exp_name=$(echo "$exp_info" | cut -d: -f2)
        local exp_dir="$EXPERIMENT_DIR/exp_$exp_num"

        if [ -f "$exp_dir/metadata.txt" ]; then
            echo "Experiment $exp_num: $exp_name"
            grep "Status:" "$exp_dir/metadata.txt" | sed 's/^/  /'
            echo ""
        fi
    done

    echo "Generated Files:"
    echo "==============="
    find "$EXPERIMENT_DIR" -name "*.png" -o -name "*.json" -o -name "*.csv" | head -20 | sed 's/^/  /'

} > "$summary_file"

print_success "Summary saved to: $summary_file"

# Check if all experiments succeeded
if [ $failed_experiments -eq 0 ]; then
    print_success "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
    echo ""
    print_info "Your journal submission package is ready in: $EXPERIMENT_DIR"
    print_info "Key results files:"
    echo "  - Experiment logs: $EXPERIMENT_DIR/exp_*/experiment.log"
    echo "  - Model checkpoints: $EXPERIMENT_DIR/exp_*/best_model.pth"
    echo "  - Visualizations: $EXPERIMENT_DIR/exp_*/*.png"
    echo "  - Data tables: $EXPERIMENT_DIR/exp_*/*.csv"
    echo "  - Summary: $summary_file"
    echo ""
    exit 0
else
    print_warning "Some experiments failed ($failed_experiments/$total_experiments)"
    print_info "Check individual experiment logs for details:"

    for exp_info in "${experiments[@]}"; do
        local exp_num=$(echo "$exp_info" | cut -d: -f1)
        local exp_dir="$EXPERIMENT_DIR/exp_$exp_num"
        if [ -f "$exp_dir/metadata.txt" ] && grep -q "FAILED" "$exp_dir/metadata.txt"; then
            local exp_name=$(echo "$exp_info" | cut -d: -f2)
            echo "  - Failed: Experiment $exp_num ($exp_name) - Log: $exp_dir/experiment.log"
        fi
    done
    echo ""
    exit 1
fi