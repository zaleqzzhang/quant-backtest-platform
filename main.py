"""
量化策略回测平台主程序
整合各模块功能，提供完整回测流程
"""

import pandas as pd
import argparse
import sys
import os
import matplotlib.pyplot as plt
from data_manager import DataManager
from data_source import DataSource
from strategy import (
    MovingAverageCrossoverStrategy,
    RSIStategy,
    BollingerBandsStrategy,
    SentimentStrategy
)
from backtest_engine import BacktestEngine
from visualization import Visualizer


# 创建全局数据管理器实例
data_manager = DataManager()


def get_data_source(source_type: str) -> DataSource:
    """
    工厂函数：根据类型创建数据源实例
    
    Parameters:
    source_type: 数据源类型 ('tushare' 或 'longport')
    
    Returns:
    DataSource: 数据源实例
    """
    if source_type.lower() == 'tushare':
        from data_source import TushareDataSource
        return TushareDataSource()
    elif source_type.lower() == 'longport':
        from data_source import LongportDataSource
        return LongportDataSource()
    else:
        raise ValueError(f"不支持的数据源类型: {source_type}")


def create_strategy(strategy_name: str, **kwargs) -> object:
    """创建策略实例"""
    strategies = {
        'ma': MovingAverageCrossoverStrategy(short_window=kwargs.get('short_window', 5), 
                                          long_window=kwargs.get('long_window', 20)),
        'rsi': RSIStategy(period=kwargs.get('rsi_period', 14),
                         oversold=kwargs.get('rsi_oversold', 30),
                         overbought=kwargs.get('rsi_overbought', 70)),
        'bollinger': BollingerBandsStrategy(period=kwargs.get('bb_period', 20),
                                          std_dev=kwargs.get('bb_std_dev', 2)),
        'sentiment': SentimentStrategy(index_code=kwargs.get('index_code'))
    }
    
    return strategies.get(strategy_name)


def run_single_backtest(strategy_name, symbol, start_date, end_date, data_source_type, **kwargs):
    """
    运行单个策略回测
    
    Parameters:
    strategy_name: 策略名称
    symbol: 股票代码
    start_date: 开始日期
    end_date: 结束日期
    data_source_type: 数据源类型
    kwargs: 其他参数
    """
    force_update = kwargs.get('force_update', False)
    adj_mode = kwargs.get('adj_mode', 'none')
    index_code = kwargs.get('index_code', None)
    short_window = kwargs.get('short_window', 5)
    long_window = kwargs.get('long_window', 20)
    rsi_period = kwargs.get('rsi_period', 14)
    rsi_oversold = kwargs.get('rsi_oversold', 30)
    rsi_overbought = kwargs.get('rsi_overbought', 70)
    bb_period = kwargs.get('bb_period', 20)
    bb_std_dev = kwargs.get('bb_std_dev', 2)
    initial_capital = kwargs.get('initial_capital', 100000.0)
    commission_rate = kwargs.get('commission_rate', 0.001)
    warmup_period = kwargs.get('warmup_period', 100)
    fixed_quantity = kwargs.get('fixed_quantity', None)
    max_position = kwargs.get('max_position', None)
    fixed_buy_quantity = kwargs.get('fixed_buy_quantity', None)
    fixed_sell_quantity = kwargs.get('fixed_sell_quantity', None)
    # 可视化选项
    disable_plots = kwargs.get('disable_plots', False)
    disable_summary = kwargs.get('disable_summary', False)
    disable_trade_details = kwargs.get('disable_trade_details', False)
    
    print("=" * 50)
    print("量化策略回测平台 - 单策略回测")
    print("=" * 50)
    print(f"策略: {strategy_name}")
    print(f"股票代码: {symbol}")
    print(f"数据源: {data_source_type}")
    print(f"回测期间: {start_date} 至 {end_date}")
    print("=" * 50)
    
    # 获取数据
    price_data = data_manager.get_data(
        symbol, start_date, end_date, data_source_type, force_update=force_update, 
        adj_mode=adj_mode, index_symbol=index_code
    )
    
    if price_data is None or price_data.empty:
        print(f"未能获取到 {symbol} 的数据")
        return
    
    # 创建策略实例
    strategy = None
    if strategy_name == "ma":
        strategy = MovingAverageCrossoverStrategy(short_window=short_window, long_window=long_window)
    elif strategy_name == "rsi":
        strategy = RSIStategy(period=rsi_period, oversold=rsi_oversold, overbought=rsi_overbought)
    elif strategy_name == "bollinger":
        strategy = BollingerBandsStrategy(period=bb_period, std_dev=bb_std_dev)
    elif strategy_name == "sentiment":
        strategy = SentimentStrategy(index_code=index_code)
    
    if strategy is None:
        print(f"未知的策略: {strategy_name}")
        return
    
    # 创建回测引擎实例（传递固定数量交易和持仓限制参数）
    engine = BacktestEngine(
        initial_capital=initial_capital, 
        commission_rate=commission_rate, 
        warmup_period=warmup_period,
        fixed_quantity=fixed_quantity,
        max_position=max_position,
        fixed_buy_quantity=fixed_buy_quantity,
        fixed_sell_quantity=fixed_sell_quantity
    )
    
    # 运行回测
    result = engine.run_backtest(strategy, price_data, symbol, start_date, end_date)
    
    # 打印结果并可视化
    print(f"\n{strategy.name}回测完成")
    visualizer = Visualizer(
        disable_plots=disable_plots,
        disable_summary=disable_summary,
        disable_trade_details=disable_trade_details
    )
    
    # 默认显示所有可视化内容，除非通过命令行参数禁用
    visualizer.plot_backtest_results(result, price_data, strategy.name)


def download_data(symbol, start_date, end_date, data_source_type, **kwargs):
    """
    下载并缓存数据功能
    
    Parameters:
    symbol: 股票代码
    start_date: 开始日期
    end_date: 结束日期
    data_source_type: 数据源类型
    kwargs: 其他参数
    """
    index_code = kwargs.get('index_code', None)
    adj_mode = kwargs.get('adj_mode', 'none')
    
    print(f"数据下载功能")
    print("=" * 50)
    print(f"股票代码: {symbol}")
    if index_code:
        print(f"指数代码: {index_code}")
    print(f"复权模式: {adj_mode}")
    print(f"数据源: {data_source_type}")
    print(f"下载期间: {start_date} 至 {end_date}")
    print("=" * 50)
    
    # 1. 初始化数据管理器
    data_manager = DataManager()
    
    # 2. 获取数据配置
    data_source_config = {}
    if data_source_type == 'tushare':
        data_source_config['token'] = os.getenv('TUSHARE_TOKEN')
    elif data_source_type == 'longport':
        data_source_config['app_key'] = os.getenv('LONGPORT_APP_KEY')
        data_source_config['app_secret'] = os.getenv('LONGPORT_APP_SECRET')
        data_source_config['access_token'] = os.getenv('LONGPORT_ACCESS_TOKEN')
    
    # 3. 强制获取并保存数据
    print("正在下载数据...")
    try:
        price_data = data_manager.fetch_and_save_data(
            symbol, start_date, end_date, data_source_type,
            index_symbol=index_code,
            adj_mode=adj_mode,
            **data_source_config
        )
        
        if price_data.empty:
            print(f"未获取到股票 {symbol} 在指定日期范围内的数据")
            return False
        
        print(f"数据下载完成!")
        print(f"数据条目数: {len(price_data)}")
        print(f"数据时间范围: {price_data.index.min()} 至 {price_data.index.max()}")
        
        # 显示数据文件信息
        data_file = data_manager._get_data_filename(symbol, start_date, end_date, data_source_type, index_code, adj_mode)
        if os.path.exists(data_file):
            file_size = os.path.getsize(data_file) / 1024  # KB
            print(f"数据已保存至: {data_file}")
            print(f"文件大小: {file_size:.2f} KB")
        
        return True
    except Exception as e:
        print(f"数据下载失败: {e}")
        return False


def view_data(symbol, start_date, end_date, data_source_type, **kwargs):
    """
    查看数据功能
    
    Parameters:
    symbol: 股票代码
    start_date: 开始日期
    end_date: 结束日期
    data_source_type: 数据源类型
    kwargs: 其他参数
    """
    index_code = kwargs.get('index_code', None)
    adj_mode = kwargs.get('adj_mode', 'none')
    
    print(f"数据查看功能")
    print("=" * 50)
    print(f"股票代码: {symbol}")
    if index_code:
        print(f"指数代码: {index_code}")
    print(f"复权模式: {adj_mode}")
    print(f"数据源: {data_source_type}")
    print(f"查看期间: {start_date} 至 {end_date}")
    print("=" * 50)
    
    # 1. 初始化数据管理器
    data_manager = DataManager()
    
    # 2. 获取数据配置
    data_source_config = {}
    if data_source_type == 'tushare':
        data_source_config['token'] = os.getenv('TUSHARE_TOKEN')
    elif data_source_type == 'longport':
        data_source_config['app_key'] = os.getenv('LONGPORT_APP_KEY')
        data_source_config['app_secret'] = os.getenv('LONGPORT_APP_SECRET')
        data_source_config['access_token'] = os.getenv('LONGPORT_ACCESS_TOKEN')
    
    # 3. 获取数据（优先从本地加载）
    price_data = data_manager.get_data(
        symbol, start_date, end_date, data_source_type,
        force_update=kwargs.get('force_update', False),
        index_symbol=index_code,
        adj_mode=adj_mode,
        **data_source_config
    )
    
    if price_data.empty:
        print(f"未获取到股票 {symbol} 在指定日期范围内的数据")
        return
    
    # 4. 显示数据基本信息
    print(f"\n数据基本信息:")
    print(f"数据条目数: {len(price_data)}")
    print(f"数据时间范围: {price_data.index.min()} 至 {price_data.index.max()}")
    print(f"股票代码: {price_data['symbol'].iloc[0] if 'symbol' in price_data.columns else 'N/A'}")
    
    # 5. 显示数据统计信息
    print(f"\n价格统计信息:")
    if 'open' in price_data.columns:
        print(f"开盘价范围: {price_data['open'].min():.2f} - {price_data['open'].max():.2f}")
    if 'high' in price_data.columns:
        print(f"最高价范围: {price_data['high'].min():.2f} - {price_data['high'].max():.2f}")
    if 'low' in price_data.columns:
        print(f"最低价范围: {price_data['low'].min():.2f} - {price_data['low'].max():.2f}")
    if 'close' in price_data.columns:
        print(f"收盘价范围: {price_data['close'].min():.2f} - {price_data['close'].max():.2f}")
    if 'volume' in price_data.columns:
        print(f"成交量范围: {price_data['volume'].min():.0f} - {price_data['volume'].max():.0f}")
    
    # 检查是否有指数数据
    has_index_data = any(col.endswith('_index') for col in price_data.columns)
    if has_index_data:
        print(f"\n指数数据信息:")
        index_cols = [col for col in price_data.columns if col.endswith('_index')]
        for col in index_cols:
            base_col = col.replace('_index', '')
            print(f"{base_col.upper()}范围: {price_data[col].min():.2f} - {price_data[col].max():.2f}")
    
    # 6. 显示前几行数据
    print(f"\n前5行数据:")
    print(price_data.head())
    
    # 显示数据文件信息
    data_file = data_manager._get_data_filename(symbol, start_date, end_date, data_source_type, index_code, adj_mode)
    if os.path.exists(data_file):
        file_size = os.path.getsize(data_file) / 1024  # KB
        print(f"\n数据文件信息:")
        print(f"文件路径: {data_file}")
        print(f"文件大小: {file_size:.2f} KB")
    
    # 7. 绘制价格和成交量图表
    plot_data_charts(price_data, symbol)


def plot_data_charts(data, symbol):
    """
    绘制数据图表
    
    Parameters:
    data: 数据
    symbol: 股票代码
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 绘制价格图表
    ax1.plot(data.index, data['close'], label='收盘价', linewidth=1.5, color='black')
    if 'open' in data.columns:
        ax1.plot(data.index, data['open'], label='开盘价', linewidth=1.5, color='blue', alpha=0.7)
    if 'high' in data.columns:
        ax1.plot(data.index, data['high'], label='最高价', linewidth=1.0, color='green', alpha=0.7)
    if 'low' in data.columns:
        ax1.plot(data.index, data['low'], label='最低价', linewidth=1.0, color='red', alpha=0.7)
    
    ax1.set_title(f'{symbol} 价格走势', fontsize=16)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 绘制成交量图表
    if 'volume' in data.columns:
        ax2.bar(data.index, data['volume'], width=1.0, color='orange', alpha=0.6, label='成交量')
        ax2.set_title(f'{symbol} 成交量', fontsize=16)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.set_ylabel('成交量', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数 - 演示量化策略回测平台的使用
    """
    parser = argparse.ArgumentParser(description='量化策略回测平台')
    parser.add_argument('--mode', choices=['demo', 'single', 'view', 'download'], default='demo', 
                       help='运行模式: demo(演示多策略), single(单策略), view(查看数据), download(下载数据)')
    parser.add_argument('--strategy', choices=['ma', 'rsi', 'bollinger', 'sentiment'],
                       help='策略名称: ma(均线交叉), rsi(RSI策略), bollinger(布林带), sentiment(情绪指标)')
    parser.add_argument('--symbol', help='股票代码')
    parser.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--data-source', choices=['tushare', 'longport'], default='tushare',
                       help='数据源类型: tushare 或 longport')
    parser.add_argument('--force-update', action='store_true', 
                       help='强制更新数据（忽略缓存）')
    parser.add_argument('--adj-mode', choices=['none', 'qfq'], default='none',
                       help='复权模式: none(不复权), qfq(前复权)')
    
    # 策略参数
    parser.add_argument('--short-window', type=int, default=5, help='均线交叉策略短周期')
    parser.add_argument('--long-window', type=int, default=20, help='均线交叉策略长周期')
    parser.add_argument('--rsi-period', type=int, default=14, help='RSI计算周期')
    parser.add_argument('--rsi-oversold', type=int, default=30, help='RSI超卖阈值')
    parser.add_argument('--rsi-overbought', type=int, default=70, help='RSI超买阈值')
    parser.add_argument('--bb-period', type=int, default=20, help='布林带计算周期')
    parser.add_argument('--bb-std-dev', type=float, default=2, help='布林带标准差倍数')
    parser.add_argument('--index-code', type=str, default=None, help='指数代码，用于需要指数数据的策略')
    
    # 回测参数
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='初始资金')
    parser.add_argument('--commission-rate', type=float, default=0.001, help='手续费率')
    parser.add_argument('--warmup-period', type=int, default=100, help='预热期长度（交易日数量）')
    parser.add_argument('--fixed-quantity', type=int, default=None, 
                       help='固定交易数量（如果不指定，则使用动态仓位）')
    parser.add_argument('--max-position', type=int, default=None, 
                       help='最大持仓数量（如果不指定，则无限制）')
    parser.add_argument('--fixed-buy-quantity', type=int, default=None, 
                       help='固定买入数量（如果不指定，则使用fixed-quantity或动态仓位）')
    parser.add_argument('--fixed-sell-quantity', type=int, default=None, 
                       help='固定卖出数量（如果不指定，则使用fixed-quantity或动态仓位）')
    
    # 可视化选项
    parser.add_argument('--disable-plots', action='store_true',
                       help='禁用图表显示')
    parser.add_argument('--disable-summary', action='store_true',
                       help='禁用摘要信息打印')
    parser.add_argument('--disable-trade-details', action='store_true',
                       help='禁用详细交易记录打印')
    parser.add_argument('--plot-equity-curve', action='store_true',
                       help='启用权益曲线图')
    parser.add_argument('--plot-drawdown-curve', action='store_true',
                       help='启用回撤曲线图')
    parser.add_argument('--plot-trades', action='store_true',
                       help='启用买卖点标记图')
    parser.add_argument('--plot-performance-comparison', action='store_true',
                       help='启用策略性能对比图')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not all([args.strategy, args.symbol, args.start_date, args.end_date]):
            print("单策略模式需要指定 --strategy, --symbol, --start-date, --end-date 参数")
            parser.print_help()
            sys.exit(1)
            
        run_single_backtest(
            strategy_name=args.strategy,
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            data_source_type=args.data_source,
            force_update=args.force_update,
            adj_mode=args.adj_mode,
            short_window=args.short_window,
            long_window=args.long_window,
            rsi_period=args.rsi_period,
            rsi_oversold=args.rsi_oversold,
            rsi_overbought=args.rsi_overbought,
            bb_period=args.bb_period,
            bb_std_dev=args.bb_std_dev,
            index_code=args.index_code,
            initial_capital=args.initial_capital,
            commission_rate=args.commission_rate,
            warmup_period=args.warmup_period,
            fixed_quantity=args.fixed_quantity,
            max_position=args.max_position,
            fixed_buy_quantity=args.fixed_buy_quantity,
            fixed_sell_quantity=args.fixed_sell_quantity,
            disable_plots=args.disable_plots,
            disable_summary=args.disable_summary,
            disable_trade_details=args.disable_trade_details
        )
    elif args.mode == 'view':
        if not all([args.symbol, args.start_date, args.end_date]):
            print("查看数据模式需要指定 --symbol, --start-date, --end-date 参数")
            parser.print_help()
            sys.exit(1)
            
        view_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            data_source_type=args.data_source,
            force_update=args.force_update,
            adj_mode=args.adj_mode,
            index_code=args.index_code
        )
    elif args.mode == 'download':
        if not all([args.symbol, args.start_date, args.end_date]):
            print("下载数据模式需要指定 --symbol, --start-date, --end-date 参数")
            parser.print_help()
            sys.exit(1)
            
        success = download_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            data_source_type=args.data_source,
            adj_mode=args.adj_mode,
            index_code=args.index_code
        )
        
        if not success:
            sys.exit(1)
    else:
        # 原有演示模式
        print("量化策略回测平台")
        print("=" * 50)
        
        # 1. 初始化数据管理器
        data_manager = DataManager()
        
        # 2. 获取数据（优先从本地加载）
        symbol = "000001.SZ"  # 平安银行
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        price_data = data_manager.get_data(symbol, start_date, end_date, 'tushare')
        
        # 3. 创建多个策略实例
        strategies = {
            "均线交叉策略": MovingAverageCrossoverStrategy(short_window=5, long_window=20),
            "RSI策略": RSIStategy(period=14, oversold=30, overbought=70),
            "布林带策略": BollingerBandsStrategy(period=20, std_dev=2)
        }
        
        # 4. 初始化回测引擎
        engine = BacktestEngine(initial_capital=100000.0, commission_rate=0.001)
        
        # 5. 运行各策略回测
        results = {}
        for name, strategy in strategies.items():
            print(f"正在运行{name}...")
            result = engine.run_backtest(strategy, price_data, symbol, start_date, end_date)
            results[name] = result
            print(f"{name}回测完成")
        
        # 6. 可视化结果
        visualizer = Visualizer(
            disable_plots=args.disable_plots,
            disable_summary=args.disable_summary,
            disable_trade_details=args.disable_trade_details
        )
        
        # 打印摘要报告
        if not args.disable_summary:
            visualizer.print_summary(results)
        
        # 打印详细交易记录
        if not args.disable_trade_details:
            print("\n" + "=" * 80)
            print("详细交易记录".center(80))
            print("=" * 80)
            for name, result in results.items():
                visualizer.print_trade_details_table(result, name)
        
        # 绘制权益曲线
        if args.plot_equity_curve and not args.disable_plots:
            visualizer.plot_equity_curve(results, "策略权益曲线对比")
        
        # 绘制回撤曲线
        if args.plot_drawdown_curve and not args.disable_plots:
            visualizer.plot_drawdown_curve(results, "策略回撤曲线对比")
        
        # 绘制策略性能对比
        if args.plot_performance_comparison and not args.disable_plots:
            visualizer.plot_performance_comparison(results)
        
        # 选择一个策略绘制买卖点
        if args.plot_trades and not args.disable_plots:
            first_strategy_name = list(results.keys())[0]
            visualizer.plot_trades(results[first_strategy_name], price_data, 
                                  f"{first_strategy_name}买卖点标记")


if __name__ == "__main__":
    main()