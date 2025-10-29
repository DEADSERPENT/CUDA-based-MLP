#!/usr/bin/env python3
"""
Training Visualization Script for CUDA MLP Project
Generates professional graphs for M.Tech project documentation
"""

import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 10

def parse_training_output(output_text):
    """
    Parse training output to extract epoch, accuracy, and timing data.

    Expected format:
    Epoch 0: Train 0.10450, Val 0.09733, Test 0.10170
    Epoch 1: Train 0.14170, Val 0.13525, Test 0.13380  Average time per epoch: 0.560 sec
    """
    epochs = []
    train_acc = []
    val_acc = []
    test_acc = []
    time_per_epoch = []
    learning_rates = []

    # Pattern for accuracy lines
    pattern = r'Epoch (\d+):\s+Train ([\d.]+),\s+Val ([\d.]+),\s+Test ([\d.]+)'
    # Pattern for time
    time_pattern = r'Average time per epoch:\s+([\d.]+)\s+sec'
    # Pattern for learning rate (if present)
    lr_pattern = r'LR:\s+([\d.]+)'

    for line in output_text.split('\n'):
        match = re.search(pattern, line)
        if match:
            epochs.append(int(match.group(1)))
            train_acc.append(float(match.group(2)) * 100)  # Convert to percentage
            val_acc.append(float(match.group(3)) * 100)
            test_acc.append(float(match.group(4)) * 100)

        time_match = re.search(time_pattern, line)
        if time_match:
            time_per_epoch.append(float(time_match.group(1)))

        lr_match = re.search(lr_pattern, line)
        if lr_match:
            learning_rates.append(float(lr_match.group(1)))

    return {
        'epochs': epochs,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'time_per_epoch': time_per_epoch,
        'learning_rates': learning_rates
    }

def plot_accuracy_curves(data_dict, output_file='training_accuracy.png'):
    """
    Plot training/validation/test accuracy curves for multiple optimizers.

    data_dict: {'SGD': data, 'Momentum': data, 'Adam': data}
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    colors = {'SGD': '#2E86AB', 'Momentum': '#A23B72', 'Adam': '#F18F01',
              'SGD+LR': '#06A77D', 'BatchNorm': '#D00000'}
    linestyles = {'train': '-', 'val': '--', 'test': '-.'}

    for name, data in data_dict.items():
        if not data['epochs']:
            continue

        color = colors.get(name, '#333333')

        # Plot test accuracy (main metric)
        ax.plot(data['epochs'], data['test_acc'],
               label=f'{name} (Test)', color=color, linewidth=2.5,
               linestyle='-', marker='o', markersize=4, markevery=2)

        # Plot train accuracy (lighter, dashed)
        ax.plot(data['epochs'], data['train_acc'],
               label=f'{name} (Train)', color=color, linewidth=1.5,
               linestyle='--', alpha=0.6)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Training Accuracy Comparison: SGD vs Advanced Optimizers\n' +
                 'CUDA MLP on MNIST (2 layers × 128 neurons, batch size 2048)',
                 fontweight='bold', pad=20)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=100)

    # Add annotations for final accuracy
    for name, data in data_dict.items():
        if data['test_acc'] and data['test_acc'][-1] > 5:  # Only annotate if converged
            final_acc = data['test_acc'][-1]
            final_epoch = data['epochs'][-1]
            ax.annotate(f'{final_acc:.1f}%',
                       xy=(final_epoch, final_acc),
                       xytext=(10, 0), textcoords='offset points',
                       fontsize=9, color=colors.get(name, '#333333'),
                       fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✓ Saved accuracy plot to {output_file}')
    plt.close()

def plot_training_time(data_dict, output_file='training_time.png'):
    """
    Bar chart comparing training time per epoch across optimizers.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = []
    times = []
    colors_list = []

    colors = {'SGD': '#2E86AB', 'Momentum': '#A23B72', 'Adam': '#F18F01',
              'SGD+LR': '#06A77D', 'BatchNorm': '#D00000'}

    for name, data in data_dict.items():
        if data['time_per_epoch']:
            names.append(name)
            # Use median time (more stable than last value)
            times.append(np.median(data['time_per_epoch']))
            colors_list.append(colors.get(name, '#333333'))

    if not names:
        print('⚠ No timing data available')
        return

    bars = ax.bar(names, times, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Time per Epoch (seconds)', fontweight='bold')
    ax.set_title('Training Speed Comparison\nCUDA MLP on MNIST (20 epochs, batch size 2048)',
                 fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(top=max(times) * 1.2)  # Add 20% headroom

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✓ Saved timing plot to {output_file}')
    plt.close()

def plot_optimizer_convergence(data_dict, output_file='optimizer_convergence.png'):
    """
    Plot showing optimizer convergence behavior - all working correctly now!
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {'SGD': '#2E86AB', 'Momentum': '#A23B72', 'Adam': '#F18F01'}

    # Left plot: First 5 epochs (zoomed in to show early convergence)
    for name, data in data_dict.items():
        if not data['epochs']:
            continue
        color = colors.get(name, '#333333')
        epochs_subset = [e for e in data['epochs'] if e <= 5]
        test_acc_subset = data['test_acc'][:len(epochs_subset)]

        ax1.plot(epochs_subset, test_acc_subset,
                label=name, color=color, linewidth=3,
                marker='o', markersize=8)

    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax1.set_title('Early Training Behavior (Epochs 0-5)', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 85)

    # Add annotation showing Adam's faster convergence
    ax1.annotate('Adam converges\nfastest (78% by epoch 5)',
                xy=(5, 78.7), xytext=(3, 55),
                arrowprops=dict(arrowstyle='->', color='#F18F01', lw=2),
                fontsize=10, color='#F18F01', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Right plot: Full training (all epochs)
    for name, data in data_dict.items():
        if not data['epochs']:
            continue
        color = colors.get(name, '#333333')
        ax2.plot(data['epochs'], data['test_acc'],
                label=name, color=color, linewidth=2.5,
                marker='o', markersize=4, markevery=3)

    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax2.set_title('Full Training Run (All Epochs)', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 90)

    plt.suptitle('Optimizer Convergence Analysis: All Optimizers Working Correctly!',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✓ Saved convergence analysis to {output_file}')
    plt.close()

def create_summary_table(data_dict):
    """
    Print a formatted summary table of results.
    """
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"{'Optimizer':<15} {'Final Test Acc':<18} {'Avg Time/Epoch':<18} {'Status':<15}")
    print("-"*80)

    for name, data in data_dict.items():
        if not data['epochs']:
            continue

        final_acc = data['test_acc'][-1] if data['test_acc'] else 0.0
        avg_time = np.mean(data['time_per_epoch']) if data['time_per_epoch'] else 0.0

        # Determine status
        if final_acc < 1.0:
            status = '❌ Diverged'
        elif final_acc < 50.0:
            status = '⚠️  Poor'
        elif final_acc < 70.0:
            status = '✓ Fair'
        else:
            status = '✓ Good'

        print(f"{name:<15} {final_acc:>6.2f}%            {avg_time:>6.3f}s            {status:<15}")

    print("="*80 + "\n")

def main():
    """
    Main function to generate all visualizations.
    """
    print("="*80)
    print("CUDA MLP Training Visualization")
    print("="*80 + "\n")

    # Real data from actual CUDA training runs (October 2025)
    # Configuration: 2 layers × 128 neurons, batch size 2048, 20 epochs

    sgd_output = """
Epoch 0: Train 0.10450, Val 0.09733, Test 0.10170
Epoch 1: Train 0.14170, Val 0.13525, Test 0.13380  Average time per epoch: 0.000000 sec
Epoch 2: Train 0.19530, Val 0.18592, Test 0.17930  Average time per epoch: inf sec
Epoch 3: Train 0.26640, Val 0.26433, Test 0.25450  Average time per epoch: 0.835749 sec
Epoch 4: Train 0.35490, Val 0.34892, Test 0.34360  Average time per epoch: 0.626687 sec
Epoch 5: Train 0.41510, Val 0.41517, Test 0.40090  Average time per epoch: 0.557106 sec
Epoch 6: Train 0.47600, Val 0.47492, Test 0.46470  Average time per epoch: 0.522303 sec
Epoch 7: Train 0.51830, Val 0.51608, Test 0.50480  Average time per epoch: 0.500890 sec
Epoch 8: Train 0.56010, Val 0.55750, Test 0.54670  Average time per epoch: 0.486574 sec
Epoch 9: Train 0.59680, Val 0.59825, Test 0.58620  Average time per epoch: 0.476406 sec
Epoch 10: Train 0.61580, Val 0.62067, Test 0.60730  Average time per epoch: 0.468790 sec
Epoch 11: Train 0.63830, Val 0.64567, Test 0.63420  Average time per epoch: 0.462847 sec
Epoch 12: Train 0.65480, Val 0.66233, Test 0.65680  Average time per epoch: 0.458052 sec
Epoch 13: Train 0.66830, Val 0.67708, Test 0.67280  Average time per epoch: 0.454150 sec
Epoch 14: Train 0.68860, Val 0.69742, Test 0.69460  Average time per epoch: 0.450907 sec
Epoch 15: Train 0.69380, Val 0.70592, Test 0.69680  Average time per epoch: 0.448161 sec
Epoch 16: Train 0.71040, Val 0.71600, Test 0.71570  Average time per epoch: 0.445801 sec
Epoch 17: Train 0.72000, Val 0.72983, Test 0.72300  Average time per epoch: 0.443778 sec
Epoch 18: Train 0.72720, Val 0.73258, Test 0.73220  Average time per epoch: 0.441997 sec
Epoch 19: Train 0.73450, Val 0.74508, Test 0.74010  Average time per epoch: 0.440426 sec
Epoch 20: Train 0.74470, Val 0.75392, Test 0.74880  Average time per epoch: 0.439015 sec
    """

    momentum_output = """
Epoch 0: Train 0.10450, Val 0.09733, Test 0.10170
Epoch 1: Train 0.14170, Val 0.13525, Test 0.13380  Average time per epoch: 0.000000 sec
Epoch 2: Train 0.24750, Val 0.24842, Test 0.23390  Average time per epoch: inf sec
Epoch 3: Train 0.39370, Val 0.40833, Test 0.38870  Average time per epoch: 0.824435 sec
Epoch 4: Train 0.53880, Val 0.54950, Test 0.53380  Average time per epoch: 0.618314 sec
Epoch 5: Train 0.62260, Val 0.63133, Test 0.61350  Average time per epoch: 0.549619 sec
Epoch 6: Train 0.66340, Val 0.66558, Test 0.66450  Average time per epoch: 0.515542 sec
Epoch 7: Train 0.69490, Val 0.69258, Test 0.69690  Average time per epoch: 0.495129 sec
Epoch 8: Train 0.72020, Val 0.71975, Test 0.72100  Average time per epoch: 0.481559 sec
Epoch 9: Train 0.73920, Val 0.74558, Test 0.74410  Average time per epoch: 0.471850 sec
Epoch 10: Train 0.74320, Val 0.75175, Test 0.74820  Average time per epoch: 0.464567 sec
Epoch 11: Train 0.73360, Val 0.74625, Test 0.73770  Average time per epoch: 0.458918 sec
Epoch 12: Train 0.71930, Val 0.73008, Test 0.72230  Average time per epoch: 0.454379 sec
Epoch 13: Train 0.72060, Val 0.73567, Test 0.73410  Average time per epoch: 0.450671 sec
Epoch 14: Train 0.74450, Val 0.75692, Test 0.75370  Average time per epoch: 0.447591 sec
Epoch 15: Train 0.73330, Val 0.74367, Test 0.74090  Average time per epoch: 0.444988 sec
Epoch 16: Train 0.69400, Val 0.70867, Test 0.70570  Average time per epoch: 0.442740 sec
Epoch 17: Train 0.74630, Val 0.75700, Test 0.75550  Average time per epoch: 0.440797 sec
Epoch 18: Train 0.75120, Val 0.75950, Test 0.75860  Average time per epoch: 0.439109 sec
Epoch 19: Train 0.70840, Val 0.72258, Test 0.72050  Average time per epoch: 0.437609 sec
Epoch 20: Train 0.73690, Val 0.74617, Test 0.74650  Average time per epoch: 0.436282 sec
    """

    adam_output = """
Epoch 0: Train 0.10450, Val 0.09733, Test 0.10170
Epoch 1: Train 0.27120, Val 0.28783, Test 0.28130  Average time per epoch: 0.000000 sec
Epoch 2: Train 0.41630, Val 0.42833, Test 0.41050  Average time per epoch: inf sec
Epoch 3: Train 0.55930, Val 0.56700, Test 0.55620  Average time per epoch: 0.808938 sec
Epoch 4: Train 0.70280, Val 0.70983, Test 0.70360  Average time per epoch: 0.606963 sec
Epoch 5: Train 0.78380, Val 0.78950, Test 0.78700  Average time per epoch: 0.539356 sec
Epoch 6: Train 0.74500, Val 0.74475, Test 0.73850  Average time per epoch: 0.505695 sec
Epoch 7: Train 0.74460, Val 0.74833, Test 0.74190  Average time per epoch: 0.485420 sec
Epoch 8: Train 0.80170, Val 0.80567, Test 0.80500  Average time per epoch: 0.471880 sec
Epoch 9: Train 0.79290, Val 0.79650, Test 0.79310  Average time per epoch: 0.462216 sec
Epoch 10: Train 0.75290, Val 0.75775, Test 0.75480  Average time per epoch: 0.455129 sec
Epoch 11: Train 0.72550, Val 0.73117, Test 0.72920  Average time per epoch: 0.449661 sec
Epoch 12: Train 0.75810, Val 0.76017, Test 0.75520  Average time per epoch: 0.445268 sec
Epoch 13: Train 0.78600, Val 0.79167, Test 0.78750  Average time per epoch: 0.441672 sec
Epoch 14: Train 0.79100, Val 0.79817, Test 0.79090  Average time per epoch: 0.438666 sec
Epoch 15: Train 0.79030, Val 0.79442, Test 0.78840  Average time per epoch: 0.436148 sec
Epoch 16: Train 0.81700, Val 0.81433, Test 0.80710  Average time per epoch: 0.433987 sec
Epoch 17: Train 0.84500, Val 0.84142, Test 0.83890  Average time per epoch: 0.432127 sec
Epoch 18: Train 0.83750, Val 0.83917, Test 0.82960  Average time per epoch: 0.430489 sec
Epoch 19: Train 0.82110, Val 0.82758, Test 0.81820  Average time per epoch: 0.429044 sec
Epoch 20: Train 0.81050, Val 0.81942, Test 0.80800  Average time per epoch: 0.427752 sec
    """

    # Parse data
    data_dict = {
        'SGD': parse_training_output(sgd_output),
        'Momentum': parse_training_output(momentum_output),
        'Adam': parse_training_output(adam_output)
    }

    # Generate plots
    print("Generating visualizations...")
    plot_accuracy_curves(data_dict, 'training_accuracy_comparison.png')
    plot_training_time(data_dict, 'training_time_comparison.png')
    plot_optimizer_convergence(data_dict, 'optimizer_convergence_analysis.png')

    # Print summary
    create_summary_table(data_dict)

    print("✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - training_accuracy_comparison.png")
    print("  - training_time_comparison.png")
    print("  - optimizer_convergence_analysis.png")
    print("\nUse these graphs in your M.Tech project report!")

if __name__ == '__main__':
    main()
