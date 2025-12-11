#!/usr/bin/env python3
"""
Analyze sequence span distribution by chemical bond type
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_sequence_span_distribution(csv_path: str, output_dir: str = None):
    """
    Analyze sequence span distribution by chemical bond type
    
    Args:
        csv_path: Path to links.csv file
        output_dir: Output directory, if None uses the directory of CSV file
    """
    print(f"Reading file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total records: {len(df)}")
    print(f"Chemical types: {df['chem_type'].unique()}")
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by chem_type and calculate statistics for sequence_span
    stats_list = []
    
    for chem_type in sorted(df['chem_type'].unique()):
        subset = df[df['chem_type'] == chem_type]['sequence_span']
        
        if len(subset) == 0:
            continue
        
        stats = {
            'chem_type': chem_type,
            'count': len(subset),
            'mean': subset.mean(),
            'median': subset.median(),
            'std': subset.std(),
            'min': subset.min(),
            'max': subset.max(),
            'q25': subset.quantile(0.25),
            'q75': subset.quantile(0.75),
            'q90': subset.quantile(0.90),
            'q95': subset.quantile(0.95),
            'q99': subset.quantile(0.99),
        }
        stats_list.append(stats)
        
        print(f"\n{chem_type}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.1f}")
        print(f"  Std: {stats['std']:.2f}")
        print(f"  Range: [{stats['min']}, {stats['max']}]")
        print(f"  Quantiles: Q25={stats['q25']:.1f}, Q75={stats['q75']:.1f}")
    
    # Save statistics
    stats_df = pd.DataFrame(stats_list)
    stats_output_path = os.path.join(output_dir, 'sequence_span_by_chem_type.csv')
    stats_df.to_csv(stats_output_path, index=False)
    print(f"\nStatistics saved to: {stats_output_path}")
    
    # Create visualizations
    try:
        plt.rcParams['font.size'] = 10
        
        # 1. Boxplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Boxplot
        chem_types = df['chem_type'].unique()
        data_to_plot = [df[df['chem_type'] == ct]['sequence_span'].values 
                       for ct in sorted(chem_types)]
        
        bp = axes[0, 0].boxplot(data_to_plot, labels=sorted(chem_types), 
                                patch_artist=True, showfliers=False)
        axes[0, 0].set_title('Sequence Span Distribution by Chemical Type (Boxplot)', fontsize=12)
        axes[0, 0].set_xlabel('Chemical Type', fontsize=10)
        axes[0, 0].set_ylabel('Sequence Span', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        # Violin plot
        if len(df) < 100000:  # Skip violin plot if data is too large
            sns.violinplot(data=df, x='chem_type', y='sequence_span', ax=axes[0, 1])
            axes[0, 1].set_title('Sequence Span Distribution by Chemical Type (Violin Plot)', fontsize=12)
            axes[0, 1].set_xlabel('Chemical Type', fontsize=10)
            axes[0, 1].set_ylabel('Sequence Span', fontsize=10)
        else:
            axes[0, 1].text(0.5, 0.5, 'Data too large, skipped violin plot', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Sequence Span Distribution (Violin Plot - Skipped)', fontsize=12)
        
        # Overlaid histograms
        for chem_type in sorted(chem_types):
            subset = df[df['chem_type'] == chem_type]['sequence_span']
            axes[1, 0].hist(subset, bins=50, alpha=0.5, label=chem_type, density=True)
        axes[1, 0].set_title('Sequence Span Distribution by Chemical Type (Histogram)', fontsize=12)
        axes[1, 0].set_xlabel('Sequence Span', fontsize=10)
        axes[1, 0].set_ylabel('Density', fontsize=10)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution function
        for chem_type in sorted(chem_types):
            subset = df[df['chem_type'] == chem_type]['sequence_span'].sort_values()
            y = np.arange(1, len(subset) + 1) / len(subset)
            axes[1, 1].plot(subset, y, label=chem_type, linewidth=2)
        axes[1, 1].set_title('Cumulative Distribution Function', fontsize=12)
        axes[1, 1].set_xlabel('Sequence Span', fontsize=10)
        axes[1, 1].set_ylabel('Cumulative Probability', fontsize=10)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_output_path = os.path.join(output_dir, 'sequence_span_distribution.png')
        plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plots saved to: {plot_output_path}")
        plt.close()
        
        # 2. Quantile comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        x_pos = np.arange(len(stats_df))
        width = 0.15
        
        ax.bar(x_pos - 2*width, stats_df['q25'], width, label='Q25', alpha=0.8)
        ax.bar(x_pos - width, stats_df['median'], width, label='Median', alpha=0.8)
        ax.bar(x_pos, stats_df['mean'], width, label='Mean', alpha=0.8)
        ax.bar(x_pos + width, stats_df['q75'], width, label='Q75', alpha=0.8)
        ax.bar(x_pos + 2*width, stats_df['q90'], width, label='Q90', alpha=0.8)
        
        ax.set_xlabel('Chemical Type', fontsize=10)
        ax.set_ylabel('Sequence Span', fontsize=10)
        ax.set_title('Sequence Span Statistics by Chemical Type', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats_df['chem_type'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        quantile_plot_path = os.path.join(output_dir, 'sequence_span_quantiles.png')
        plt.savefig(quantile_plot_path, dpi=300, bbox_inches='tight')
        print(f"Quantile comparison plot saved to: {quantile_plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    return stats_df


def analyze_sequence_span_1_20(csv_path: str, output_dir: str = None):
    """
    Detailed analysis of sequence span in the 1-20 range by chemical bond type
    
    Args:
        csv_path: Path to links.csv file
        output_dir: Output directory, if None uses the directory of CSV file
    """
    print(f"Reading file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total records: {len(df)}")
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for sequence span 1-20
    df_1_20 = df[(df['sequence_span'] >= 1) & (df['sequence_span'] <= 20)].copy()
    print(f"Records with sequence_span 1-20: {len(df_1_20)} ({100.0*len(df_1_20)/len(df):.2f}%)")
    
    # Statistics by chemical type for 1-20 range
    stats_list = []
    detailed_counts_list = []
    
    for chem_type in sorted(df['chem_type'].unique()):
        all_subset = df[df['chem_type'] == chem_type]['sequence_span']
        subset_1_20 = df_1_20[df_1_20['chem_type'] == chem_type]['sequence_span']
        
        if len(all_subset) == 0:
            continue
        
        # Overall statistics for 1-20 range
        stats = {
            'chem_type': chem_type,
            'total_count': len(all_subset),
            'count_1_20': len(subset_1_20),
            'pct_1_20': 100.0 * len(subset_1_20) / len(all_subset),
            'mean_1_20': subset_1_20.mean() if len(subset_1_20) > 0 else 0,
            'median_1_20': subset_1_20.median() if len(subset_1_20) > 0 else 0,
            'std_1_20': subset_1_20.std() if len(subset_1_20) > 0 else 0,
        }
        stats_list.append(stats)
        
        print(f"\n{chem_type}:")
        print(f"  Total count: {stats['total_count']}")
        print(f"  Count in 1-20: {stats['count_1_20']} ({stats['pct_1_20']:.2f}%)")
        if len(subset_1_20) > 0:
            print(f"  Mean (1-20): {stats['mean_1_20']:.2f}")
            print(f"  Median (1-20): {stats['median_1_20']:.1f}")
            print(f"  Std (1-20): {stats['std_1_20']:.2f}")
        
        # Detailed count for each value 1-20
        for span_val in range(1, 21):
            count = len(subset_1_20[subset_1_20 == span_val])
            pct_in_1_20 = 100.0 * count / len(subset_1_20) if len(subset_1_20) > 0 else 0
            pct_of_total = 100.0 * count / len(all_subset) if len(all_subset) > 0 else 0
            
            detailed_counts_list.append({
                'chem_type': chem_type,
                'sequence_span': span_val,
                'count': count,
                'pct_in_1_20': pct_in_1_20,
                'pct_of_total': pct_of_total,
            })
    
    # Save statistics
    stats_df = pd.DataFrame(stats_list)
    stats_output_path = os.path.join(output_dir, 'sequence_span_1_20_stats.csv')
    stats_df.to_csv(stats_output_path, index=False)
    print(f"\nStatistics saved to: {stats_output_path}")
    
    # Save detailed counts
    detailed_df = pd.DataFrame(detailed_counts_list)
    detailed_output_path = os.path.join(output_dir, 'sequence_span_1_20_detailed.csv')
    detailed_df.to_csv(detailed_output_path, index=False)
    print(f"Detailed counts saved to: {detailed_output_path}")
    
    # Create visualizations
    try:
        plt.rcParams['font.size'] = 10
        sns.set_style("whitegrid")
        
        # 1. Bar plot showing counts for each span value (1-20)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        chem_types = sorted(df['chem_type'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(chem_types)))
        color_map = {ct: colors[i] for i, ct in enumerate(chem_types)}
        
        # Plot 1: Count by sequence span (1-20) for each chemical type
        x_pos = np.arange(1, 21)
        width = 0.8 / len(chem_types)
        
        for i, chem_type in enumerate(chem_types):
            subset = df_1_20[df_1_20['chem_type'] == chem_type]
            counts = [len(subset[subset['sequence_span'] == val]) for val in range(1, 21)]
            axes[0, 0].bar(x_pos + i*width - width*(len(chem_types)-1)/2, counts, 
                          width, label=chem_type, alpha=0.8, color=color_map[chem_type])
        
        axes[0, 0].set_xlabel('Sequence Span', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].set_title('Count Distribution: Sequence Span 1-20 by Chemical Type', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(range(1, 21))
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Percentage within 1-20 range (normalized)
        for i, chem_type in enumerate(chem_types):
            subset = df_1_20[df_1_20['chem_type'] == chem_type]
            if len(subset) > 0:
                percentages = [100.0 * len(subset[subset['sequence_span'] == val]) / len(subset) 
                              for val in range(1, 21)]
                axes[0, 1].plot(x_pos, percentages, marker='o', label=chem_type, 
                               linewidth=2, markersize=4, color=color_map[chem_type])
        
        axes[0, 1].set_xlabel('Sequence Span', fontsize=11)
        axes[0, 1].set_ylabel('Percentage (%) within 1-20 range', fontsize=11)
        axes[0, 1].set_title('Percentage Distribution: Sequence Span 1-20 (Normalized)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(range(1, 21))
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Stacked bar chart showing proportion
        bottom = np.zeros(20)
        for chem_type in chem_types:
            subset = df_1_20[df_1_20['chem_type'] == chem_type]
            counts = [len(subset[subset['sequence_span'] == val]) for val in range(1, 21)]
            axes[1, 0].bar(x_pos, counts, label=chem_type, bottom=bottom, 
                             alpha=0.8, color=color_map[chem_type])
            bottom += counts
        
        axes[1, 0].set_xlabel('Sequence Span', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title('Stacked Count: Sequence Span 1-20 by Chemical Type', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(range(1, 21))
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Heatmap of counts
        heatmap_data = []
        for chem_type in chem_types:
            subset = df_1_20[df_1_20['chem_type'] == chem_type]
            counts = [len(subset[subset['sequence_span'] == val]) for val in range(1, 21)]
            heatmap_data.append(counts)
        
        heatmap_df = pd.DataFrame(heatmap_data, index=chem_types, columns=range(1, 21))
        sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1], 
                   cbar_kws={'label': 'Count'})
        axes[1, 1].set_xlabel('Sequence Span', fontsize=11)
        axes[1, 1].set_ylabel('Chemical Type', fontsize=11)
        axes[1, 1].set_title('Heatmap: Count by Sequence Span and Chemical Type', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot_output_path = os.path.join(output_dir, 'sequence_span_1_20_detailed.png')
        plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
        print(f"Detailed plots saved to: {plot_output_path}")
        plt.close()
        
        # 2. Separate subplots for each chemical type (better visibility)
        n_types = len(chem_types)
        fig, axes = plt.subplots(n_types, 1, figsize=(14, 4*n_types))
        if n_types == 1:
            axes = [axes]
        
        for idx, chem_type in enumerate(chem_types):
            subset = df_1_20[df_1_20['chem_type'] == chem_type]
            counts = [len(subset[subset['sequence_span'] == val]) for val in range(1, 21)]
            
            axes[idx].bar(x_pos, counts, alpha=0.8, color=color_map[chem_type], edgecolor='black', linewidth=0.5)
            axes[idx].set_xlabel('Sequence Span', fontsize=11)
            axes[idx].set_ylabel('Count', fontsize=11)
            axes[idx].set_title(f'{chem_type.upper()}: Sequence Span 1-20 Distribution (Total: {len(subset)})', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xticks(range(1, 21))
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, count in enumerate(counts):
                if count > 0:
                    axes[idx].text(i+1, count + max(counts)*0.01, str(count), 
                                 ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        individual_plot_path = os.path.join(output_dir, 'sequence_span_1_20_by_type.png')
        plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
        print(f"Individual plots saved to: {individual_plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    return stats_df, detailed_df


def main():
    parser = argparse.ArgumentParser(description='Analyze sequence span distribution by chemical bond type')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to links.csv file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as CSV file directory)')
    parser.add_argument('--focus_1_20', action='store_true',
                       help='Focus on detailed analysis of sequence span 1-20 range')
    
    args = parser.parse_args()
    
    if args.focus_1_20:
        analyze_sequence_span_1_20(args.csv, args.output_dir)
    else:
        analyze_sequence_span_distribution(args.csv, args.output_dir)


if __name__ == '__main__':
    main()

