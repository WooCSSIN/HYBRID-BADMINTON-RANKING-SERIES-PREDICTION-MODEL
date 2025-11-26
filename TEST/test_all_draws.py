"""
Test ML Pipeline with All Draws (MS, WS, MD, WD, XD)
==================================================

This script tests the complete ML pipeline across all badminton draws
and creates comprehensive visualizations for comparison.

Author: ML Improvements Team
Date: 2024-11-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from importlib import import_module

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

# Import modules using importlib
fe_module = import_module('1_feature_engineering')
val_module = import_module('2_validation_framework')
ci_module = import_module('4_confidence_intervals')


def test_single_draw(draw_name='MS'):
    """
    Test pipeline for a single draw using existing demo functions
    
    Args:
        draw_name: One of 'MS', 'WS', 'MD', 'WD', 'XD'
    
    Returns:
        dict: Results including metrics and predictions
    """
    print(f"\n{'='*80}")
    print(f" TESTING: {draw_name} Draw")
    print(f"{'='*80}\n")
    
    try:
        # Check if validation results exist
        val_file = Path(f'outputs/validation_results_{draw_name}.csv')
        ci_file = Path(f'outputs/predictions_with_ci_{draw_name}.csv')
        
        if not val_file.exists() or not ci_file.exists():
            print(f"     WARNING: Results for {draw_name} not found. Skipping.")
            print(f"      Run demo_pipeline.py for {draw_name} first.")
            return None
        
        # Load results
        validation_df = pd.read_csv(val_file)
        predictions_df = pd.read_csv(ci_file)
        
        # Calculate metrics
        avg_metrics = {
            'mae': validation_df['mae'].mean(),
            'rmse': validation_df['rmse'].mean(),
            'mape': validation_df['mape'].mean(),
            'spearman': validation_df['spearman_corr'].mean()
        }
        
        ci_width = predictions_df['confidence_width'].mean()
        within_ci = ((predictions_df['actual_points'] >= predictions_df['lower_bound']) & 
                    (predictions_df['actual_points'] <= predictions_df['upper_bound'])).mean() * 100
        
        print(f"\n Results for {draw_name}:")
        print(f"   MAE:        {avg_metrics['mae']:.2f}")
        print(f"   RMSE:       {avg_metrics['rmse']:.2f}")
        print(f"   Spearman:   {avg_metrics['spearman']:.4f}")
        print(f"   CI Width:   {ci_width:.2f} points")
        print(f"   CI Coverage: {within_ci:.1f}%")
        
        # Get dataset info from enhanced file
        enhanced_file = Path('../bwf_official_enhanced.csv')
        if enhanced_file.exists():
            df = pd.read_csv(enhanced_file)
            draw_df = df[df['draw'] == draw_name]
            n_samples = len(draw_df)
            n_features = len([c for c in df.columns if c not in ['draw', 'date', 'player_id', 'player_name', 'points']])
        else:
            n_samples = len(predictions_df)
            n_features = 21  # Estimated
        
        return {
            'draw': draw_name,
            'n_samples': n_samples,
            'n_features': n_features,
            'validation_metrics': avg_metrics,
            'ci_width': ci_width,
            'ci_coverage': within_ci,
            'validation_results': validation_df,
            'predictions': predictions_df
        }
        
    except Exception as e:
        print(f"    ERROR testing {draw_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_visualizations(all_results):
    """
    Create comprehensive visualizations comparing all draws
    
    Args:
        all_results: List of result dictionaries from test_single_draw()
    """
    print(f"\n{'='*80}")
    print(f" CREATING COMPARISON VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # Filter out None results
    all_results = [r for r in all_results if r is not None]
    
    if len(all_results) == 0:
        print("    No results to visualize")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Validation Metrics Comparison (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    draws = [r['draw'] for r in all_results]
    mae_values = [r['validation_metrics']['mae'] for r in all_results]
    
    bars = ax1.bar(draws, mae_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(draws)])
    ax1.set_title('Mean Absolute Error by Draw', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MAE (points)')
    ax1.set_xlabel('Draw')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. RMSE Comparison
    ax2 = plt.subplot(2, 3, 2)
    rmse_values = [r['validation_metrics']['rmse'] for r in all_results]
    bars = ax2.bar(draws, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(draws)])
    ax2.set_title('Root Mean Square Error by Draw', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE (points)')
    ax2.set_xlabel('Draw')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    # 3. Spearman Correlation
    ax3 = plt.subplot(2, 3, 3)
    spearman_values = [r['validation_metrics']['spearman'] for r in all_results]
    bars = ax3.bar(draws, spearman_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(draws)])
    ax3.set_title('Spearman Correlation by Draw', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Correlation')
    ax3.set_xlabel('Draw')
    ax3.set_ylim([min(spearman_values) - 0.001, 1.0])
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # 4. Confidence Interval Coverage
    ax4 = plt.subplot(2, 3, 4)
    ci_coverage = [r['ci_coverage'] for r in all_results]
    bars = ax4.bar(draws, ci_coverage, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(draws)])
    ax4.axhline(y=80, color='r', linestyle='--', label='Target 80%', linewidth=2)
    ax4.set_title('Confidence Interval Coverage', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Coverage (%)')
    ax4.set_xlabel('Draw')
    ax4.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    # 5. CI Width Comparison
    ax5 = plt.subplot(2, 3, 5)
    ci_widths = [r['ci_width'] for r in all_results]
    bars = ax5.bar(draws, ci_widths, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(draws)])
    ax5.set_title('Average Confidence Width', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Width (points)')
    ax5.set_xlabel('Draw')
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10)
    
    # 6. Dataset Size Comparison
    ax6 = plt.subplot(2, 3, 6)
    n_samples = [r['n_samples'] for r in all_results]
    bars = ax6.bar(draws, n_samples, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(draws)])
    ax6.set_title('Dataset Size by Draw', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Number of Records')
    ax6.set_xlabel('Draw')
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/draw_comparison.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: outputs/draw_comparison.png")
    
    # Create summary report
    create_summary_report(all_results)


def create_summary_report(all_results):
    """Create a summary report in JSON and markdown format"""
    
    # JSON report
    summary = {
        'test_date': datetime.now().isoformat(),
        'draws_tested': [r['draw'] for r in all_results],
        'results': []
    }
    
    for r in all_results:
        summary['results'].append({
            'draw': r['draw'],
            'n_samples': int(r['n_samples']),
            'n_features': int(r['n_features']),
            'mae': float(r['validation_metrics']['mae']),
            'rmse': float(r['validation_metrics']['rmse']),
            'mape': float(r['validation_metrics']['mape']),
            'spearman_correlation': float(r['validation_metrics']['spearman']),
            'ci_width': float(r['ci_width']),
            'ci_coverage_percent': float(r['ci_coverage'])
        })
    
    with open('outputs/test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved: outputs/test_summary.json")
    
    # Markdown report
    md_lines = [
        "# ML Pipeline Test Results - All Draws",
        f"\n**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Draws Tested:** {', '.join([r['draw'] for r in all_results])}",
        "\n## Summary Table\n",
        "| Draw | Samples | Features | MAE | RMSE | Spearman | CI Coverage | CI Width |",
        "|------|---------|----------|-----|------|----------|-------------|----------|"
    ]
    
    for r in all_results:
        md_lines.append(
            f"| {r['draw']} | {r['n_samples']:,} | {r['n_features']} | "
            f"{r['validation_metrics']['mae']:.1f} | {r['validation_metrics']['rmse']:.1f} | "
            f"{r['validation_metrics']['spearman']:.4f} | {r['ci_coverage']:.1f}% | "
            f"{r['ci_width']:.0f} |"
        )
    
    md_lines.extend([
        "\n## Key Insights\n",
        f"- **Best MAE:** {min(all_results, key=lambda x: x['validation_metrics']['mae'])['draw']}",
        f"- **Best Correlation:** {max(all_results, key=lambda x: x['validation_metrics']['spearman'])['draw']}",
        f"- **Best CI Coverage:** {max(all_results, key=lambda x: x['ci_coverage'])['draw']}",
        "\n## Files Generated\n",
        "- `outputs/draw_comparison.png` - Visualization comparing all draws",
        "- `outputs/test_summary.json` - Detailed results in JSON format",
        "- `outputs/test_summary.md` - This summary report"
    ])
    
    with open('outputs/test_summary.md', 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"    Saved: outputs/test_summary.md")


def main():
    """Main testing function - analyzes existing results"""
    print("\n" + "="*80)
    print(" ML PIPELINE TEST - ANALYZING EXISTING RESULTS")
    print("="*80)
    print("\nThis will analyze existing validation results from:")
    print("  - outputs/validation_results_*.csv")
    print("  -outputs/predictions_with_ci_*.csv")
    print("\n To generate results for other draws, edit demo_pipeline.py")
    print("   and change DRAW = 'MS' to 'WS', 'MD', 'WD', or 'XD'\n")
    
    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Check which draws have results
    draws_to_test = ['MS', 'WS', 'MD', 'WD', 'XD']
    all_results = []
    
    for draw in draws_to_test:
        result = test_single_draw(draw)
        if result:
            all_results.append(result)
    
    # Create visualizations
    if all_results:
        create_comparison_visualizations(all_results)
        
        print(f"\n{'='*80}")
        print(f" ANALYSIS COMPLETED!")
        print(f"{'='*80}\n")
        print(f" Results saved in: TEST/outputs/")
        print(f"   - draw_comparison.png")
        print(f"   - test_summary.json")
        print(f"   - test_summary.md\n")
    else:
        print(f"\n  No results found to analyze.")
        print(f"   Run demo_pipeline.py first to generate results for MS draw.")
        print(f"   Then edit DRAW variable in demo files to test other draws.\n")


if __name__ == "__main__":
    main()



def create_summary_report(all_results):
    """Create a summary report in JSON and markdown format"""
    
    # JSON report
    summary = {
        'test_date': datetime.now().isoformat(),
        'draws_tested': [r['draw'] for r in all_results],
        'results': []
    }
    
    for r in all_results:
        summary['results'].append({
            'draw': r['draw'],
            'n_samples': int(r['n_samples']),
            'n_features': int(r['n_features']),
            'mae': float(r['validation_metrics']['mae']),
            'rmse': float(r['validation_metrics']['rmse']),
            'mape': float(r['validation_metrics']['mape']),
            'spearman_correlation': float(r['validation_metrics']['spearman']),
            'ci_width': float(r['ci_width']),
            'ci_coverage_percent': float(r['ci_coverage'])
        })
    
    with open('outputs/test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved: outputs/test_summary.json")
    
    # Markdown report
    md_lines = [
        "# ML Pipeline Test Results - All Draws",
        f"\n**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Draws Tested:** {', '.join([r['draw'] for r in all_results])}",
        "\n## Summary Table\n",
        "| Draw | Samples | Features | MAE | RMSE | Spearman | CI Coverage | CI Width |",
        "|------|---------|----------|-----|------|----------|-------------|----------|"
    ]
    
    for r in all_results:
        md_lines.append(
            f"| {r['draw']} | {r['n_samples']:,} | {r['n_features']} | "
            f"{r['validation_metrics']['mae']:.1f} | {r['validation_metrics']['rmse']:.1f} | "
            f"{r['validation_metrics']['spearman']:.4f} | {r['ci_coverage']:.1f}% | "
            f"{r['ci_width']:.0f} |"
        )
    
    md_lines.extend([
        "\n## Key Insights\n",
        f"- **Best MAE:** {min(all_results, key=lambda x: x['validation_metrics']['mae'])['draw']}",
        f"- **Best Correlation:** {max(all_results, key=lambda x: x['validation_metrics']['spearman'])['draw']}",
        f"- **Best CI Coverage:** {max(all_results, key=lambda x: x['ci_coverage'])['draw']}",
        "\n## Files Generated\n",
        "- `outputs/draw_comparison.png` - Visualization comparing all draws",
        "- `outputs/test_summary.json` - Detailed results in JSON format"
    ])
    
    for r in all_results:
        md_lines.append(f"- `outputs/predictions_with_ci_{r['draw']}.csv` - Predictions for {r['draw']}")
    
    with open('outputs/test_summary.md', 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"    Saved: outputs/test_summary.md")


def main():
    """Main testing function"""
    print("\n" + "="*80)
    print(" TESTING ML PIPELINE - ALL DRAWS")
    print("="*80)
    
    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Test all draws
    draws_to_test = ['MS', 'WS', 'MD', 'WD', 'XD']
    all_results = []
    
    for draw in draws_to_test:
        result = test_single_draw(draw)
        if result:
            all_results.append(result)
    
    # Create visualizations
    if all_results:
        create_comparison_visualizations(all_results)
    
    print(f"\n{'='*80}")
    print(f" TESTING COMPLETED!")
    print(f"{'='*80}\n")
    print(f" Results saved in: TEST/outputs/")
    print(f"   - draw_comparison.png")
    print(f"   - test_summary.json")
    print(f"   - test_summary.md")
    print(f"   - predictions_with_ci_*.csv (for each draw)\n")


if __name__ == "__main__":
    main()
