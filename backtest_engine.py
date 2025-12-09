"""
回测引擎模块
执行策略回测的核心逻辑
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dataclasses import dataclass, field
from strategy import Strategy, Signal, SignalType
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    trade_type: SignalType
    price: float
    quantity: float
    timestamp: pd.Timestamp
    commission: float = 0.0
    total_cost: float = 0.0
    reason: str = ""  # 交易原因


@dataclass
class Portfolio:
    """投资组合"""
    cash: float = 100000.0  # 初始资金
    positions: Dict[str, float] = field(default_factory=dict)  # 持仓 {symbol: quantity}
    portfolio_value: float = 100000.0  # 总资产价值
    history: List[Dict] = field(default_factory=list)  # 价值历史记录


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = None
    drawdown_curve: pd.Series = None
    
    def print_trade_details(self):
        """
        输出详细的交易点信息
        """
        if not self.trades:
            print("没有交易记录")
            return
            
        print("\n详细交易记录:")
        print("-" * 100)
        print(f"{'时间':<20} {'类型':<6} {'股票代码':<10} {'价格':<10} {'数量':<10} {'手续费':<10} {'成本':<12}")
        print("-" * 100)
        
        for trade in self.trades:
            trade_type = "买入" if trade.trade_type == SignalType.BUY else "卖出"
            print(f"{trade.timestamp:<20} {trade_type:<6} {trade.symbol:<10} "
                  f"{trade.price:<10.2f} {trade.quantity:<10.0f} {trade.commission:<10.2f} {trade.total_cost:<12.2f}")
        
        print("-" * 100)
        print(f"总交易次数: {len(self.trades)}")
    
    def print_summary(self):
        """
        打印回测结果摘要
        """
        print("=" * 80)
        print("回测结果摘要".center(80))
        print("=" * 80)
        print(f"总收益率: {self.total_return:.2f}%")
        print(f"年化收益率: {self.annual_return:.2f}%")
        print(f"最大回撤: {self.max_drawdown:.2f}%")
        print(f"夏普比率: {self.sharpe_ratio:.2f}")
        print(f"胜率: {self.win_rate:.2f}")
        print(f"盈利因子: {self.profit_factor:.2f}")
        print(f"交易次数: {len(self.trades) if self.trades else 0}")


class BacktestEngine:
    """
    回测引擎
    执行策略回测的核心逻辑
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission_rate: float = 0.001, 
                 warmup_period: int = 100, fixed_quantity: int = None, max_position: int = None,
                 fixed_buy_quantity: int = None, fixed_sell_quantity: int = None):
        """
        初始化回测引擎
        
        Parameters:
        initial_capital: 初始资金
        commission_rate: 手续费率
        warmup_period: 预热期长度（交易日数量）
        fixed_quantity: 固定交易数量（如果为None，则使用动态仓位）
        max_position: 最大持仓数量（如果为None，则无限制）
        fixed_buy_quantity: 固定买入数量（如果为None，则使用fixed_quantity或动态仓位）
        fixed_sell_quantity: 固定卖出数量（如果为None，则使用fixed_quantity或动态仓位）
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.warmup_period = warmup_period
        self.fixed_quantity = fixed_quantity
        self.max_position = max_position
        self.fixed_buy_quantity = fixed_buy_quantity
        self.fixed_sell_quantity = fixed_sell_quantity
        self.portfolio = Portfolio(cash=initial_capital)
        self.results = {}  # 存储回测结果
    
    def run_backtest(self, strategy: Strategy, data: pd.DataFrame, symbol: str, 
                     start_date: str, end_date: str) -> BacktestResult:
        """
        运行回测
        
        Parameters:
        strategy: 策略实例
        data: 历史价格数据
        symbol: 股票代码
        start_date: 回测开始日期
        end_date: 回测结束日期
        
        Returns:
        BacktestResult: 回测结果
        """
        # 筛选指定日期范围内的数据
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        if data.empty:
            print(f"在指定日期范围内没有找到数据: {start_date} 至 {end_date}")
            return BacktestResult()
        
        # 确定预热期后的日期
        if len(data) <= self.warmup_period:
            print(f"数据不足{self.warmup_period}个交易日，无法进行回测")
            return BacktestResult()
        
        warmup_end_date = data.index[self.warmup_period - 1]  # 预热期结束的日期
        print(f"策略将在 {warmup_end_date.strftime('%Y-%m-%d')} 之后开始交易 (预热期: {self.warmup_period} 个交易日)")
        
        # 向策略传递完整数据集（如果策略支持）
        if hasattr(strategy, 'set_full_data'):
            strategy.set_full_data(data)
        
        # 初始化变量
        trades = []
        equity_curve = []
        timestamps = []
        
        # 按日遍历数据
        for i in range(1, len(data)):
            # 当前日期的数据
            current_data = data.iloc[:i+1].copy()  # 使用.copy()避免SettingWithCopyWarning
            
            # 只有在超过预热期之后才生成交易信号
            if i >= self.warmup_period:  # 从预热期后第一个交易日开始交易
                # 生成交易信号
                signal = strategy.generate_signal(current_data)
                
                # 执行交易
                trade = self._execute_trade(signal, data.iloc[i])
                if trade:
                    trades.append(trade)
            
            # 计算当前资产总值
            portfolio_value = self._calculate_portfolio_value(data.iloc[i])
            equity_curve.append(portfolio_value)
            timestamps.append(data.index[i])
            
            # 记录到历史
            self.portfolio.history.append({
                'timestamp': data.index[i],
                'portfolio_value': portfolio_value,
                'cash': self.portfolio.cash,
                'positions': self.portfolio.positions.copy()
            })
        
        # 在回测结束时强制平仓
        final_trade = self._force_liquidate(data.iloc[-1])
        if final_trade:
            trades.append(final_trade)
            portfolio_value = self._calculate_portfolio_value(data.iloc[-1])
            equity_curve.append(portfolio_value)
            timestamps.append(data.index[-1])
            
        # 构建权益曲线
        equity_series = pd.Series(equity_curve, index=timestamps)
        
        # 计算回测指标
        result = self._calculate_metrics(equity_series, trades)
        result.trades = trades
        result.equity_curve = equity_series
        result.price_data = data  # 添加价格数据到回测结果中
        
        self.results[strategy.name] = result
        return result

    def _execute_trade(self, signal: Signal, current_bar: pd.Series) -> Trade:
        """
        执行交易
        
        Parameters:
        signal: 交易信号
        current_bar: 当前K线数据
        
        Returns:
        Trade: 交易记录（如果没有交易则返回None）
        """
        if signal.signal_type == SignalType.HOLD:
            return None
            
        symbol = signal.symbol
        price = signal.price
        timestamp = signal.timestamp
        
        # 计算可交易数量（基于可用现金的一定比例）
        position_size = 0
        if signal.signal_type == SignalType.BUY:
            # 检查是否超过最大持仓限制
            current_position = self.portfolio.positions.get(symbol, 0)
            if self.max_position is not None and current_position >= self.max_position:
                return None  # 超过最大持仓限制，不买入
            
            # 确定使用的买入数量
            if self.fixed_buy_quantity is not None:
                # 使用固定的买入数量
                position_size = self.fixed_buy_quantity
            elif self.fixed_quantity is not None:
                # 使用通用的固定交易数量
                position_size = self.fixed_quantity
            else:
                # 动态调整仓位大小，最多使用可用现金的一部分
                available_cash = self.portfolio.cash * 0.1  # 使用更多可用资金
                position_size = int(available_cash / price / 100) * 100  # 以100股为单位
            
            if position_size > 0:
                cost = position_size * price
                commission = cost * self.commission_rate
                total_cost = cost + commission
                
                # 检查是否有足够现金
                if total_cost <= self.portfolio.cash:
                    self.portfolio.cash -= total_cost
                    self.portfolio.positions[symbol] = current_position + position_size
                    
                    trade = Trade(
                        symbol=symbol,
                        trade_type=signal.signal_type,
                        price=price,
                        quantity=position_size,
                        timestamp=timestamp,
                        commission=commission,
                        total_cost=-total_cost,  # 改为负值表示现金流出
                        reason=signal.reason     # 添加交易原因
                    )
                    return trade
                    
        elif signal.signal_type == SignalType.SELL:
            # 检查是否有持仓可以卖出
            if symbol in self.portfolio.positions and self.portfolio.positions[symbol] > 0:
                # 确定使用的卖出数量
                if self.fixed_sell_quantity is not None:
                    # 使用固定的卖出数量
                    position_size = min(self.fixed_sell_quantity, self.portfolio.positions[symbol])
                elif self.fixed_quantity is not None:
                    # 使用通用的固定交易数量
                    position_size = min(self.fixed_quantity, self.portfolio.positions[symbol])
                else:
                    # 可以只卖出部分仓位
                    available_shares = self.portfolio.positions[symbol]
                    position_size = int(available_shares / 2) if available_shares > 100 else available_shares  # 卖出一半或全部
                
                if position_size <= 0:
                    return None
                    
                proceeds = position_size * price
                commission = proceeds * self.commission_rate
                net_proceeds = proceeds - commission
                
                self.portfolio.cash += net_proceeds
                self.portfolio.positions[symbol] -= position_size
                
                trade = Trade(
                    symbol=symbol,
                    trade_type=signal.signal_type,
                    price=price,
                    quantity=position_size,  # 正的数量表示卖出数量
                    timestamp=timestamp,
                    commission=commission,
                    total_cost=net_proceeds,  # 正值表示现金流入
                    reason=signal.reason      # 添加交易原因
                )
                return trade
                
        return None

    def _force_liquidate(self, current_bar: pd.Series) -> Trade:
        """
        强制平仓所有持仓
        
        Parameters:
        current_bar: 当前K线数据
        
        Returns:
        Trade: 交易记录（如果没有持仓则返回None）
        """
        symbol = current_bar['symbol'] if 'symbol' in current_bar else 'unknown'
        price = current_bar['close']
        timestamp = current_bar.name
        
        if symbol in self.portfolio.positions and self.portfolio.positions[symbol] > 0:
            # 强制平仓所有剩余股份
            position_size = self.portfolio.positions[symbol]
            proceeds = position_size * price
            commission = proceeds * self.commission_rate
            net_proceeds = proceeds - commission
            
            self.portfolio.cash += net_proceeds
            self.portfolio.positions[symbol] = 0
            
            trade = Trade(
                symbol=symbol,
                trade_type=SignalType.SELL,
                price=price,
                quantity=position_size,
                timestamp=timestamp,
                commission=commission,
                total_cost=net_proceeds,
                reason="强制平仓"  # 添加交易原因
            )
            return trade
            
        return None

    def _calculate_portfolio_value(self, current_bar: pd.Series) -> float:
        """
        计算当前投资组合总价值
        
        Parameters:
        current_bar: 当前K线数据
        
        Returns:
        float: 投资组合总价值
        """
        total_value = self.portfolio.cash
        
        for symbol, quantity in self.portfolio.positions.items():
            if quantity > 0:
                # 这里简化处理，假设只有一个股票
                total_value += quantity * current_bar['close']
                
        self.portfolio.portfolio_value = total_value
        return total_value

    def _calculate_metrics(self, equity_curve: pd.Series, trades: List[Trade]) -> BacktestResult:
        """
        计算回测指标
        
        Parameters:
        equity_curve: 权益曲线
        trades: 交易记录列表
        
        Returns:
        BacktestResult: 包含各项指标的回测结果
        """
        if len(equity_curve) == 0:
            return BacktestResult()
            
        # 总收益率
        total_return = (equity_curve.iloc[-1] / self.initial_capital - 1) * 100
        
        # 年化收益率
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = total_return * 365 / days if days > 0 else 0
        
        # 最大回撤
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # 夏普比率 (简化计算，无风险利率设为0)
        returns = equity_curve.pct_change().dropna()
        if returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
            
        # 胜率和盈利因子
        profitable_trades = 0
        total_won = 0.0
        total_lost = 0.0
        
        # 创建交易配对来计算收益
        buy_trades = []
        sell_trades = []
        
        for trade in trades:
            if trade.trade_type == SignalType.BUY:
                buy_trades.append(trade)
            elif trade.trade_type == SignalType.SELL and buy_trades:
                # 将卖出交易与最早的未匹配买入交易配对
                buy_trade = buy_trades.pop(0)  # FIFO原则
                sell_trades.append((buy_trade, trade))
        
        # 计算每一对交易的盈亏
        for buy_trade, sell_trade in sell_trades:
            # 盈利 = 卖出金额 - 买入金额 - 手续费
            profit = (sell_trade.price - buy_trade.price) * buy_trade.quantity - \
                     sell_trade.commission - buy_trade.commission
            
            if profit > 0:
                profitable_trades += 1
                total_won += profit
            else:
                total_lost += abs(profit)
                
        total_pairs = len(sell_trades)
        win_rate = (profitable_trades / total_pairs) if total_pairs > 0 else 0
        # 修复盈利因子计算逻辑，避免出现inf值
        if total_lost > 0:
            profit_factor = total_won / total_lost
        elif total_won > 0:
            # 当只有盈利没有亏损时，盈利因子设为total_won（或者一个大的有限值）
            profit_factor = total_won
        else:
            profit_factor = 0
        
        result = BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            equity_curve=equity_curve,
            drawdown_curve=drawdown
        )
        
        return result