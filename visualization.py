"""
可视化模块
提供回测结果的可视化展示功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
from backtest_engine import BacktestResult
from strategy import SignalType
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    可视化工具类
    """

    def plot_equity_curve(self, results: dict, title: str = "策略权益曲线"):
        """
        绘制策略权益曲线
        
        Parameters:
        results: 回测结果字典 {策略名称: BacktestResult}
        title: 图表标题
        """
        plt.figure(figsize=(12, 6))
        
        for name, result in results.items():
            if result.equity_curve is not None and not result.equity_curve.empty:
                plt.plot(result.equity_curve.index, result.equity_curve.values, label=name, linewidth=2)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('账户价值', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_drawdown_curve(self, results: dict, title: str = "策略回撤曲线"):
        """
        绘制策略回撤曲线
        
        Parameters:
        results: 回测结果字典 {策略名称: BacktestResult}
        title: 图表标题
        """
        plt.figure(figsize=(12, 6))
        
        for name, result in results.items():
            if result.drawdown_curve is not None and not result.drawdown_curve.empty:
                plt.plot(result.drawdown_curve.index, result.drawdown_curve.values, label=name, linewidth=2)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('回撤 (%)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_trades(self, result: BacktestResult, price_data: pd.DataFrame, title: str = "买卖点标记"):
        """
        在价格图表上标记买卖点
        
        Parameters:
        result: 回测结果
        price_data: 价格数据
        title: 图表标题
        """
        if result.trades is None or len(result.trades) == 0:
            print("没有交易记录可绘制")
            return
            
        plt.figure(figsize=(12, 8))
        
        # 绘制收盘价
        plt.plot(price_data.index, price_data['close'], label='收盘价', linewidth=2)
        
        # 标记买卖点
        buy_signals = [(trade.timestamp, trade.price) for trade in result.trades if trade.trade_type == SignalType.BUY]
        sell_signals = [(trade.timestamp, trade.price) for trade in result.trades if trade.trade_type == SignalType.SELL]
        
        if buy_signals:
            buy_times, buy_prices = zip(*buy_signals)
            plt.scatter(buy_times, buy_prices, color='red', marker='^', s=100, label='买入信号', zorder=5)
            
        if sell_signals:
            sell_times, sell_prices = zip(*sell_signals)
            plt.scatter(sell_times, sell_prices, color='green', marker='v', s=100, label='卖出信号', zorder=5)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('价格', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def print_summary(self, results: dict):
        """
        打印回测结果摘要
        
        Parameters:
        results: 回测结果字典 {策略名称: BacktestResult}
        """
        print("=" * 80)
        print("回测结果摘要".center(80))
        print("=" * 80)
        
        for name, result in results.items():
            print(f"\n策略名称: {name}")
            print("-" * 50)
            print(f"总收益率: {result.total_return:.2f}%")
            print(f"年化收益率: {result.annual_return:.2f}%")
            print(f"最大回撤: {result.max_drawdown:.2f}%")
            print(f"夏普比率: {result.sharpe_ratio:.2f}")
            print(f"胜率: {result.win_rate:.2f}")
            print(f"盈利因子: {result.profit_factor:.2f}")
            print(f"交易次数: {len(result.trades)}")

    def print_trade_details_table(self, result: BacktestResult, strategy_name: str):
        """
        以表格形式打印详细交易记录
        
        Parameters:
        result: 回测结果
        strategy_name: 策略名称
        """
        if not result.trades:
            print("没有交易记录")
            return
            
        print(f"\n策略 '{strategy_name}' 的详细交易记录:")
        print("-" * 110)
        print(f"{'时间':<20} {'类型':<8} {'股票代码':<12} {'价格':<12} {'数量':<12} {'手续费':<12} {'金额':<12}")
        print("-" * 110)
        
        buy_count = 0
        sell_count = 0
        total_commission = 0
        
        for trade in result.trades:
            trade_type = "买入" if trade.trade_type == SignalType.BUY else "卖出"
            if trade.trade_type == SignalType.BUY:
                buy_count += 1
            else:
                sell_count += 1
                
            total_commission += trade.commission
            # 显示正确的交易金额
            amount = abs(trade.total_cost)  # 使用绝对值显示交易金额
            # 修复时间显示问题，统一时间格式
            timestamp_str = trade.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(trade.timestamp, 'strftime') else str(trade.timestamp)
            print(f"{timestamp_str:<20} {trade_type:<8} {trade.symbol:<12} "
                  f"{trade.price:<12.2f} {abs(trade.quantity):<12.0f} {trade.commission:<12.2f} {amount:<12.2f}")
        
        print("-" * 110)
        print(f"总买入次数: {buy_count}, 总卖出次数: {sell_count}, 总交易次数: {len(result.trades)}, 总手续费: {total_commission:.2f}")

    def plot_performance_comparison(self, results: Dict[str, BacktestResult]):
        """
        绘制策略性能对比图
        
        Parameters:
        results: 回测结果字典 {策略名称: BacktestResult}
        """
        if len(results) < 2:
            print("至少需要两个策略才能进行比较")
            return
            
        metrics = ['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio']
        metric_names = ['总收益率(%)', '年化收益率(%)', '最大回撤(%)', '夏普比率']
        
        # 准备数据
        data = {}
        for strategy_name, result in results.items():
            data[strategy_name] = [
                result.total_return,
                result.annual_return,
                abs(result.max_drawdown),  # 使用绝对值显示
                result.sharpe_ratio
            ]
        
        df = pd.DataFrame(data, index=metric_names)
        
        # 绘制分组柱状图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [getattr(result, metric) for result in results.values()]
            names = list(results.keys())
            
            # 对于回撤，使用绝对值以便更好地可视化
            if metric == 'max_drawdown':
                values = [abs(v) for v in values]
            
            bars = axes[i].bar(names, values, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
            axes[i].set_title(metric_name)
            axes[i].set_ylabel(metric_name)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{value:.2f}', ha='center', va='bottom')
            
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()