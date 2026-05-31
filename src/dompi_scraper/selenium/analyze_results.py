#!/usr/bin/env python3
"""
Analysis and report generation for benchmark results.

Reads CSV results, computes statistics, generates plots, and produces
a comprehensive comparison report in Markdown format.

Usage:
    python analyze_results.py bench_results/results_<timestamp>.csv
    python analyze_results.py bench_results/results_*.csv --output report.md
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib and/or pandas not available. Plots will be skipped.")


class BenchmarkAnalyzer:
    """Analyze benchmark results and generate reports."""
    
    def __init__(self):
        self.data = []
        self.scenarios = {}
    
    def load_csv(self, csv_file: str):
        """Load results from CSV file."""
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
        
        print(f"Loaded {len(self.data)} records from {csv_file}")
    
    def organize_by_scenario(self):
        """Organize results by scenario (approach + concurrency)."""
        for record in self.data:
            scenario = f"{record['approach']}_conc{record['concurrency']}"
            if scenario not in self.scenarios:
                self.scenarios[scenario] = []
            self.scenarios[scenario].append(record)
    
    def compute_statistics(self) -> Dict:
        """Compute statistics for each scenario."""
        stats = {}
        
        for scenario, records in self.scenarios.items():
            if not records:
                continue
            
            successful_records = [r for r in records if r['success'].lower() == 'true']
            
            if not successful_records:
                stats[scenario] = {
                    'total': len(records),
                    'successful': 0,
                    'success_rate': 0.0,
                }
                continue
            
            wall_times = [float(r['wall_time_s']) for r in successful_records]
            bytes_list = [int(r['bytes_downloaded']) for r in successful_records]
            cpu_times = [float(r['cpu_user_s']) + float(r['cpu_sys_s']) for r in successful_records]
            memory_rss = [float(r['memory_rss_mb']) for r in successful_records]
            
            stats[scenario] = {
                'total': len(records),
                'successful': len(successful_records),
                'success_rate': len(successful_records) / len(records),
                'wall_time_mean': statistics.mean(wall_times),
                'wall_time_median': statistics.median(wall_times),
                'wall_time_stdev': statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0,
                'wall_time_min': min(wall_times),
                'wall_time_max': max(wall_times),
                'bytes_total': sum(bytes_list),
                'bytes_mean': statistics.mean(bytes_list),
                'cpu_time_mean': statistics.mean(cpu_times),
                'cpu_time_total': sum(cpu_times),
                'memory_rss_mean': statistics.mean(memory_rss),
                'memory_rss_max': max(memory_rss),
                'throughput_docs_per_sec': len(successful_records) / sum(wall_times) if sum(wall_times) > 0 else 0.0,
            }
        
        return stats
    
    def generate_text_report(self, stats: Dict) -> str:
        """Generate text report in Markdown format."""
        report = []
        
        report.append("# Benchmark Comparison Report\n")
        report.append(f"Total records: {len(self.data)}\n")
        report.append(f"Scenarios: {len(self.scenarios)}\n\n")
        
        report.append("## Summary Statistics by Scenario\n\n")
        
        # Sort by approach and concurrency for readability
        sorted_scenarios = sorted(
            stats.items(),
            key=lambda x: (x[0].split('_conc')[0], int(x[0].split('_conc')[1]))
        )
        
        for scenario, scenario_stats in sorted_scenarios:
            approach, conc = scenario.rsplit('_conc', 1)
            
            report.append(f"### {approach} (concurrency={conc})\n\n")
            report.append(f"- **Success rate**: {scenario_stats['success_rate']*100:.1f}% ({scenario_stats['successful']}/{scenario_stats['total']})\n")
            
            if scenario_stats['successful'] > 0:
                report.append(f"- **Wall time per document**:\n")
                report.append(f"  - Mean: {scenario_stats['wall_time_mean']:.3f}s\n")
                report.append(f"  - Median: {scenario_stats['wall_time_median']:.3f}s\n")
                report.append(f"  - Stdev: {scenario_stats['wall_time_stdev']:.3f}s\n")
                report.append(f"  - Min/Max: {scenario_stats['wall_time_min']:.3f}s / {scenario_stats['wall_time_max']:.3f}s\n")
                
                report.append(f"- **Throughput**: {scenario_stats['throughput_docs_per_sec']:.2f} docs/sec\n")
                
                report.append(f"- **Data transferred**: {scenario_stats['bytes_total'] / (1024**2):.2f} MB ")
                report.append(f"({scenario_stats['bytes_mean'] / 1024:.1f} KB/doc avg)\n")
                
                report.append(f"- **CPU time**: {scenario_stats['cpu_time_mean']:.3f}s mean, {scenario_stats['cpu_time_total']:.1f}s total\n")
                report.append(f"- **Memory**: {scenario_stats['memory_rss_mean']:.2f} MB mean, {scenario_stats['memory_rss_max']:.2f} MB max\n")
            
            report.append("\n")
        
        # Comparison section
        report.append("## Comparison by Approach\n\n")
        
        approaches_data = defaultdict(list)
        for scenario, scenario_stats in stats.items():
            approach = scenario.split('_conc')[0]
            approaches_data[approach].append(scenario_stats)
        
        for approach, approach_stats_list in sorted(approaches_data.items()):
            if not approach_stats_list:
                continue
            
            avg_wall_times = [s.get('wall_time_mean', 0) for s in approach_stats_list if s['successful'] > 0]
            avg_throughput = [s.get('throughput_docs_per_sec', 0) for s in approach_stats_list if s['successful'] > 0]
            avg_memory = [s.get('memory_rss_mean', 0) for s in approach_stats_list if s['successful'] > 0]
            
            report.append(f"### {approach}\n\n")
            if avg_wall_times:
                report.append(f"- **Avg wall time/doc**: {statistics.mean(avg_wall_times):.3f}s\n")
            if avg_throughput:
                report.append(f"- **Avg throughput**: {statistics.mean(avg_throughput):.2f} docs/sec\n")
            if avg_memory:
                report.append(f"- **Avg memory**: {statistics.mean(avg_memory):.2f} MB\n")
            report.append("\n")
        
        return "".join(report)
    
    def generate_plots(self, output_dir: str = "."):
        """Generate comparison plots."""
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping plots (matplotlib/pandas not available)")
            return
        
        df = pd.DataFrame(self.data)
        
        # Convert numeric columns
        for col in ['wall_time_s', 'bytes_downloaded', 'cpu_user_s', 'cpu_sys_s', 'memory_rss_mb']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['success'] = df['success'].str.lower() == 'true'
        
        # Filter successful results only
        df_success = df[df['success']]
        
        if df_success.empty:
            print("No successful results to plot")
            return
        
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Wall time by approach and concurrency
        fig, ax = plt.subplots(figsize=(12, 6))
        scenarios = df_success.groupby(['approach', 'concurrency'])['wall_time_s'].apply(list)
        scenario_names = [f"{a}\n(conc={c})" for a, c in scenarios.index]
        scenario_data = [times for times in scenarios.values]
        
        ax.boxplot(scenario_data, labels=scenario_names)
        ax.set_ylabel('Wall time (seconds)')
        ax.set_title('Download time distribution by approach and concurrency')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir_path / 'wall_time_boxplot.png', dpi=100)
        plt.close()
        print(f"Saved: {output_dir_path / 'wall_time_boxplot.png'}")
        
        # Plot 2: Throughput by approach and concurrency
        fig, ax = plt.subplots(figsize=(10, 6))
        throughput_by_scenario = {}
        for approach in df_success['approach'].unique():
            for conc in sorted(df_success['concurrency'].unique()):
                mask = (df_success['approach'] == approach) & (df_success['concurrency'] == conc)
                subset = df_success[mask]
                if not subset.empty:
                    total_time = subset['wall_time_s'].sum()
                    throughput = len(subset) / total_time if total_time > 0 else 0
                    scenario_key = f"{approach}\n(conc={conc})"
                    throughput_by_scenario[scenario_key] = throughput
        
        scenario_names = list(throughput_by_scenario.keys())
        throughput_values = list(throughput_by_scenario.values())
        
        bars = ax.bar(range(len(scenario_names)), throughput_values, color='steelblue')
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax.set_ylabel('Throughput (docs/sec)')
        ax.set_title('Throughput by approach and concurrency')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir_path / 'throughput_bar.png', dpi=100)
        plt.close()
        print(f"Saved: {output_dir_path / 'throughput_bar.png'}")
        
        # Plot 3: Memory usage by approach and concurrency
        fig, ax = plt.subplots(figsize=(10, 6))
        memory_by_scenario = {}
        for approach in df_success['approach'].unique():
            for conc in sorted(df_success['concurrency'].unique()):
                mask = (df_success['approach'] == approach) & (df_success['concurrency'] == conc)
                subset = df_success[mask]
                if not subset.empty:
                    avg_memory = subset['memory_rss_mb'].mean()
                    scenario_key = f"{approach}\n(conc={conc})"
                    memory_by_scenario[scenario_key] = avg_memory
        
        scenario_names = list(memory_by_scenario.keys())
        memory_values = list(memory_by_scenario.values())
        
        bars = ax.bar(range(len(scenario_names)), memory_values, color='coral')
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Average memory usage by approach and concurrency')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir_path / 'memory_bar.png', dpi=100)
        plt.close()
        print(f"Saved: {output_dir_path / 'memory_bar.png'}")
        
        # Plot 4: Success rate by approach
        fig, ax = plt.subplots(figsize=(10, 6))
        success_by_approach = defaultdict(list)
        for approach in df['approach'].unique():
            for conc in sorted(df['concurrency'].unique()):
                mask = (df['approach'] == approach) & (df['concurrency'] == conc)
                subset = df[mask]
                if not subset.empty:
                    success_rate = subset['success'].sum() / len(subset)
                    success_by_approach[f"{approach}\n(conc={conc})"].append(success_rate)
        
        scenario_names = list(success_by_approach.keys())
        success_rates = [rates[0] * 100 for rates in success_by_approach.values()]
        
        bars = ax.bar(range(len(scenario_names)), success_rates, color='green', alpha=0.7)
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax.set_ylabel('Success rate (%)')
        ax.set_ylim([0, 105])
        ax.set_title('Success rate by approach and concurrency')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir_path / 'success_rate.png', dpi=100)
        plt.close()
        print(f"Saved: {output_dir_path / 'success_rate.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and generate reports"
    )
    parser.add_argument(
        'results',
        type=str,
        nargs='+',
        help='CSV file(s) with benchmark results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_report.md',
        help='Output Markdown report file'
    )
    parser.add_argument(
        '--plots-dir',
        type=str,
        default='plots',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = BenchmarkAnalyzer()
    
    # Load all CSV files
    for csv_file in args.results:
        csv_path = Path(csv_file)
        if csv_path.exists():
            analyzer.load_csv(str(csv_path))
        else:
            print(f"Warning: File not found: {csv_file}")
    
    if not analyzer.data:
        print("Error: No data loaded")
        sys.exit(1)
    
    # Organize and analyze
    analyzer.organize_by_scenario()
    stats = analyzer.compute_statistics()
    
    # Generate report
    report_text = analyzer.generate_text_report(stats)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Report saved to: {args.output}")
    
    # Generate plots
    if not args.no_plots:
        analyzer.generate_plots(output_dir=args.plots_dir)


if __name__ == '__main__':
    main()
