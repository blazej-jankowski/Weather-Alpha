import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BacktestEngine:
    def __init__(self, initial_capital: float = 100_000.0, transaction_cost_bps: float = 5.0):
        self.initial_capital = initial_capital
        self.cost_pct = transaction_cost_bps / 10_000.0  # 5 bps = 0.0005

    def run_comparison_backtest(
            self,
            df_results: pd.DataFrame,
            long_threshold: float = 0.60,
            short_threshold: float = 0.42
    ) -> pd.DataFrame:
        """
        Simulates three distinct strategies concurrently:
        1. Benchmark (Buy & Hold)
        2. Weather-Alpha (High-Conviction Long-Only)
        3. Weather-Alpha (Long/Short Market-Neutral)
        """
        bt = df_results.copy()

        # --- 1. LONG-ONLY STRATEGY ---
        bt['pos_long_only'] = (bt['signal_prob'] >= long_threshold).astype(int)
        bt['trades_long_only'] = bt['pos_long_only'].diff().abs().fillna(bt['pos_long_only'])
        bt['cost_long_only'] = bt['trades_long_only'] * self.cost_pct
        bt['ret_long_only'] = (bt['pos_long_only'] * bt['target_return_t1']) - bt['cost_long_only']
        bt['equity_long_only'] = self.initial_capital * (1 + bt['ret_long_only']).cumprod()

        # --- 2. LONG/SHORT STRATEGY ---
        pos_ls = np.zeros(len(bt), dtype=int)
        for i in range(len(bt)):
            prob = bt['signal_prob'].iloc[i]
            if prob >= long_threshold:
                pos_ls[i] = 1
            elif prob <= short_threshold:
                pos_ls[i] = -1
            else:
                pos_ls[i] = 0

        bt['pos_long_short'] = pos_ls
        bt['trades_long_short'] = bt['pos_long_short'].diff().abs().fillna(bt['pos_long_short'].abs())
        bt['cost_long_short'] = bt['trades_long_short'] * self.cost_pct
        bt['ret_long_short'] = (bt['pos_long_short'] * bt['target_return_t1']) - bt['cost_long_short']
        bt['equity_long_short'] = self.initial_capital * (1 + bt['ret_long_short']).cumprod()

        # --- 3. BENCHMARK (BUY & HOLD) ---
        bt['equity_benchmark'] = self.initial_capital * (1 + bt['target_return_t1']).cumprod()

        return bt

    def calculate_metrics_summary(self, bt: pd.DataFrame) -> pd.DataFrame:
        """Computes comparative performance metrics across all three regimes."""
        ann_factor = 252
        total_days = len(bt)

        def _calc(returns, equity, positions):
            cagr = (equity.iloc[-1] / self.initial_capital) ** (ann_factor / total_days) - 1
            vol = returns.std() * np.sqrt(ann_factor)
            sharpe = cagr / vol if vol > 0 else 0.0

            peak = equity.cummax()
            mdd = ((equity - peak) / peak).min()

            active = positions != 0
            win_rate = (returns[active] > 0).sum() / active.sum() if active.sum() > 0 else 0.0

            return {
                "Total Return": f"{(equity.iloc[-1] / self.initial_capital - 1) * 100:.2f}%",
                "Annualized Return": f"{cagr * 100:.2f}%",
                "Annualized Volatility": f"{vol * 100:.2f}%",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Max Drawdown": f"{mdd * 100:.2f}%",
                "Days in Market": f"{int(active.sum())} / {total_days}",
                "Win Rate": f"{win_rate * 100:.2f}%" if active.sum() > 0 else "N/A"
            }

        summary = {
            "Benchmark (Buy & Hold)": _calc(bt['target_return_t1'], bt['equity_benchmark'],
                                            pd.Series(1, index=bt.index)),
            "Weather-Alpha (Long-Only)": _calc(bt['ret_long_only'], bt['equity_long_only'], bt['pos_long_only']),
            "Weather-Alpha (Long/Short)": _calc(bt['ret_long_short'], bt['equity_long_short'], bt['pos_long_short'])
        }

        return pd.DataFrame(summary)