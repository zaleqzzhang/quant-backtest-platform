"""
策略模块
定义量化策略的基类和各种具体策略实现
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """信号类型枚举"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    price: float
    timestamp: pd.Timestamp
    strength: float = 1.0  # 信号强度 0-1
    reason: str = ""       # 信号产生原因


class Strategy(ABC):
    """抽象策略基类"""

    def __init__(self, name: str):
        """
        初始化策略
        
        Parameters:
        name: 策略名称
        """
        self.name = name
        self.position = {}  # 当前持仓 {symbol: quantity}

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        根据数据生成交易信号
        
        Parameters:
        data: 历史价格数据
        
        Returns:
        Signal: 交易信号
        """
        pass

    def update_position(self, symbol: str, quantity: float):
        """
        更新持仓
        
        Parameters:
        symbol: 股票代码
        quantity: 数量变化（正数表示买入，负数表示卖出）
        """
        if symbol not in self.position:
            self.position[symbol] = 0
        self.position[symbol] += quantity


class MovingAverageCrossoverStrategy(Strategy):
    """
    均线交叉策略
    当短期均线上穿长期均线时买入，下穿时卖出
    """

    def __init__(self, name: str = "MovingAverageCrossover", short_window: int = 5, long_window: int = 20):
        """
        初始化均线交叉策略
        
        Parameters:
        name: 策略名称
        short_window: 短期均线窗口
        long_window: 长期均线窗口
        """
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        生成交易信号
        """
        if len(data) < self.long_window:
            return Signal(
                symbol=data.iloc[-1]['symbol'] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.HOLD,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                reason="数据不足"
            )

        # 计算移动平均线
        short_ma = data['close'].rolling(window=self.short_window, min_periods=1).mean()
        long_ma = data['close'].rolling(window=self.long_window, min_periods=1).mean()

        # 获取最新两个时间点的均线值
        short_current = short_ma.iloc[-1]
        short_prev = short_ma.iloc[-2] if len(short_ma) > 1 else short_current
        long_current = long_ma.iloc[-1]
        long_prev = long_ma.iloc[-2] if len(long_ma) > 1 else long_current

        symbol = data.iloc[-1]['symbol'] if 'symbol' in data.columns else 'unknown'
        price = data['close'].iloc[-1]
        timestamp = data.index[-1]

        # 判断交叉情况
        if short_prev <= long_prev and short_current > long_current:
            # 短期均线上穿长期均线，买入信号
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=price,
                timestamp=timestamp,
                strength=min(1.0, (short_current - long_current) / long_current * 10),
                reason=f"短期均线({short_current:.2f})上穿长期均线({long_current:.2f})"
            )
        elif short_prev >= long_prev and short_current < long_current:
            # 短期均线下穿长期均线，卖出信号
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=price,
                timestamp=timestamp,
                strength=min(1.0, (long_current - short_current) / long_current * 10),
                reason=f"短期均线({short_current:.2f})下穿长期均线({long_current:.2f})"
            )
        else:
            # 无交叉，保持持有
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=price,
                timestamp=timestamp,
                reason="无交叉信号"
            )


class RSIStategy(Strategy):
    """
    RSI相对强弱指数策略
    当RSI低于超卖区时买入，高于超买区时卖出
    """

    def __init__(self, name: str = "RSIStrategy", period: int = 14, oversold: int = 30, overbought: int = 70):
        """
        初始化RSI策略
        
        Parameters:
        name: 策略名称
        period: RSI计算周期
        oversold: 超卖阈值
        overbought: 超买阈值
        """
        super().__init__(name)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _calculate_rsi(self, prices: pd.Series) -> float:
        """
        计算RSI值
        
        Parameters:
        prices: 价格序列
        
        Returns:
        float: RSI值
        """
        if len(prices) < self.period + 1:
            return 50.0  # 数据不足时返回中性值

        deltas = prices.diff().dropna()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)

        # 使用简单移动平均而不是滚动平均
        avg_gain = gains.tail(self.period).mean()
        avg_loss = losses.tail(self.period).mean()

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        生成交易信号
        """
        symbol = data.iloc[-1]['symbol'] if 'symbol' in data.columns else 'unknown'
        price = data['close'].iloc[-1]
        timestamp = data.index[-1]

        rsi = self._calculate_rsi(data['close'])

        if rsi < self.oversold:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=price,
                timestamp=timestamp,
                strength=(self.oversold - rsi) / self.oversold,
                reason=f"RSI({rsi:.2f})进入超卖区"
            )
        elif rsi > self.overbought:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=price,
                timestamp=timestamp,
                strength=(rsi - self.overbought) / (100 - self.overbought),
                reason=f"RSI({rsi:.2f})进入超买区"
            )
        else:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=price,
                timestamp=timestamp,
                reason=f"RSI({rsi:.2f})处于正常区间"
            )


class BollingerBandsStrategy(Strategy):
    """
    布林带策略
    当价格突破布林带上轨时卖出，突破下轨时买入
    """

    def __init__(self, name: str = "BollingerBandsStrategy", period: int = 20, std_dev: int = 2):
        """
        初始化布林带策略
        
        Parameters:
        name: 策略名称
        period: 布林带计算周期
        std_dev: 标准差倍数
        """
        super().__init__(name)
        self.period = period
        self.std_dev = std_dev

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        生成交易信号
        """
        if len(data) < self.period:
            return Signal(
                symbol=data.iloc[-1]['symbol'] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.HOLD,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                reason="数据不足"
            )

        symbol = data.iloc[-1]['symbol'] if 'symbol' in data.columns else 'unknown'
        price = data['close'].iloc[-1]
        timestamp = data.index[-1]

        # 计算布林带
        rolling_mean = data['close'].rolling(window=self.period, min_periods=1).mean()
        rolling_std = data['close'].rolling(window=self.period, min_periods=1).std()
        
        # 处理标准差为空的情况
        if pd.isna(rolling_std.iloc[-1]):
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=price,
                timestamp=timestamp,
                reason="标准差计算异常"
            )

        middle_band = rolling_mean.iloc[-1]
        upper_band = middle_band + (rolling_std.iloc[-1] * self.std_dev)
        lower_band = middle_band - (rolling_std.iloc[-1] * self.std_dev)
        
        # 如果带宽太小，则不交易
        if upper_band - lower_band < price * 0.001:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=price,
                timestamp=timestamp,
                reason="布林带宽度不足"
            )

        prev_price = data['close'].iloc[-2] if len(data) > 1 else price

        if prev_price <= lower_band and price > lower_band:
            # 价格从下方突破下轨，买入信号
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=price,
                timestamp=timestamp,
                strength=min(1.0, (price - lower_band) / (middle_band - lower_band)) if middle_band != lower_band else 1.0,
                reason=f"价格({price:.2f})突破布林带下轨({lower_band:.2f})"
            )
        elif prev_price >= upper_band and price < upper_band:
            # 价格从上方突破上轨，卖出信号
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=price,
                timestamp=timestamp,
                strength=min(1.0, (upper_band - price) / (upper_band - middle_band)) if upper_band != middle_band else 1.0,
                reason=f"价格({price:.2f})突破布林带上轨({upper_band:.2f})"
            )
        else:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=price,
                timestamp=timestamp,
                reason="无布林带突破信号"
            )