import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PerformanceVisualizer:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    def generate_comparison_tearsheet(
        self,
        bt_df: pd.DataFrame,
        importance_df: pd.DataFrame,
        output_name: str = "performance_comparison_tearsheet.png"
    ):
        """Generates a 3-panel tearsheet comparing Long-Only vs Long/Short vs Benchmark."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 1, 1.2]})
        fig.suptitle("Weather-Alpha: Systematic Carbon Trading Strategy Analysis (2023 Out-of-Sample)", fontsize=13, fontweight='bold', y=0.94)

        # 1. CUMULATIVE EQUITY (Portfolio value across all 3 variants)
        ax1 = axes[0]
        ax1.plot(bt_df.index, bt_df['equity_long_only'], label='Weather-Alpha (High-Conviction Long-Only > 0.60)', color='#1f77b4', linewidth=2.2)
        ax1.plot(bt_df.index, bt_df['equity_long_short'], label='Weather-Alpha (Long/Short Asymmetric)', color='#9467bd', linewidth=1.8, linestyle='-.')
        ax1.plot(bt_df.index, bt_df['equity_benchmark'], label='EUA Futures Benchmark (Buy & Hold)', color='#d62728', linestyle='--', linewidth=1.5)
        ax1.set_ylabel("Portfolio Value [EUR]", fontsize=10, fontweight='bold')
        ax1.set_title("Cumulative Equity: Long-Only vs. Long/Short vs. Market", fontsize=11, fontweight='bold', loc='left')
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.3)

        # 2. UNDERWATER DRAWDOWN (Capital drawdown and exposure risk)
        ax2 = axes[1]
        peak_lo = bt_df['equity_long_only'].cummax()
        dd_lo = (bt_df['equity_long_only'] - peak_lo) / peak_lo * 100

        peak_ls = bt_df['equity_long_short'].cummax()
        dd_ls = (bt_df['equity_long_short'] - peak_ls) / peak_ls * 100

        peak_bench = bt_df['equity_benchmark'].cummax()
        dd_bench = (bt_df['equity_benchmark'] - peak_bench) / peak_bench * 100

        ax2.plot(bt_df.index, dd_lo, label='Drawdown: Long-Only', color='#1f77b4', linewidth=1.8)
        ax2.plot(bt_df.index, dd_ls, label='Drawdown: Long/Short', color='#9467bd', linewidth=1.5, linestyle='-.')
        ax2.plot(bt_df.index, dd_bench, label='Drawdown: Benchmark', color='#d62728', linestyle=':', linewidth=1.2)
        ax2.fill_between(bt_df.index, dd_lo, 0, color='#1f77b4', alpha=0.15)
        ax2.set_ylabel("Drawdown [%]", fontsize=10, fontweight='bold')
        ax2.set_title("Underwater Chart (Capital Preservation & Asymmetry Analysis)", fontsize=11, fontweight='bold', loc='left')
        ax2.legend(loc='lower left', frameon=True)
        ax2.grid(True, alpha=0.3)

        # 3. FEATURE IMPORTANCE (Permutation feature ranking)
        ax3 = axes[2]
        sorted_imp = importance_df.sort_values('Importance', ascending=True)
        ax3.barh(sorted_imp['Feature'], sorted_imp['Importance'], color='#2ca02c', alpha=0.85)
        ax3.set_xlabel("Permutation Importance Score", fontsize=10, fontweight='bold')
        ax3.set_title("Predictive Drivers: Proof of Dunkelflaute Lead Time (Lag-3 Dominance)", fontsize=11, fontweight='bold', loc='left')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        save_path = os.path.join(self.output_dir, output_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"\n[INFO] Saved performance tearsheet to: {save_path}")