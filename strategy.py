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
from scipy.stats import linregress


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
    """均线交叉策略"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20, name: str = "均线交叉策略"):
        """
        初始化均线交叉策略
        
        Parameters:
        short_window: 短期均线窗口
        long_window: 长期均线窗口
        name: 策略名称
        """
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        生成交易信号
        
        Parameters:
        data: 历史价格数据
        
        Returns:
        Signal: 交易信号
        """
        if len(data) < self.long_window:
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.HOLD,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1]
            )
        
        # 计算移动平均线
        short_ma = data['close'].rolling(window=self.short_window, min_periods=1).mean()
        long_ma = data['close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # 生成信号
        if short_ma.iloc[-2] <= long_ma.iloc[-2] and short_ma.iloc[-1] > long_ma.iloc[-1]:
            # 金叉：短期均线上穿长期均线
            reason = f"金叉:MA{self.short_window}上穿MA{self.long_window}"
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.BUY,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strength=min(1.0, abs(short_ma.iloc[-1] - long_ma.iloc[-1]) / data['close'].iloc[-1]),
                reason=reason
            )
        elif short_ma.iloc[-2] >= long_ma.iloc[-2] and short_ma.iloc[-1] < long_ma.iloc[-1]:
            # 死叉：短期均线下穿长期均线
            reason = f"死叉:MA{self.short_window}下穿MA{self.long_window}"
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.SELL,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strength=min(1.0, abs(short_ma.iloc[-1] - long_ma.iloc[-1]) / data['close'].iloc[-1]),
                reason=reason
            )
        else:
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.HOLD,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1]
            )


class RSIStategy(Strategy):
    """RSI策略"""
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70, name: str = "RSI策略"):
        """
        初始化RSI策略
        
        Parameters:
        period: RSI计算周期
        oversold: 超卖阈值
        overbought: 超买阈值
        name: 策略名称
        """
        super().__init__(name)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        生成交易信号
        
        Parameters:
        data: 历史价格数据
        
        Returns:
        Signal: 交易信号
        """
        if len(data) < self.period + 1:
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.HOLD,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1]
            )
        
        # 计算RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # 生成信号
        if prev_rsi <= self.oversold and current_rsi > self.oversold:
            # RSI从超卖区回升，买入信号
            reason = f"RSI({current_rsi:.2f})超卖回升"
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.BUY,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strength=min(1.0, (self.oversold - current_rsi) / self.oversold),
                reason=reason
            )
        elif prev_rsi >= self.overbought and current_rsi < self.overbought:
            # RSI从超买区回落，卖出信号
            reason = f"RSI({current_rsi:.2f})超买回落"
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.SELL,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1],
                strength=min(1.0, (current_rsi - self.overbought) / (100 - self.overbought)),
                reason=reason
            )
        else:
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.HOLD,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1]
            )


class BollingerBandsStrategy(Strategy):
    """布林带策略"""
    
    def __init__(self, period: int = 20, std_dev: float = 2, name: str = "布林带策略"):
        """
        初始化布林带策略
        
        Parameters:
        period: 布林带计算周期
        std_dev: 标准差倍数
        name: 策略名称
        """
        super().__init__(name)
        self.period = period
        self.std_dev = std_dev
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        生成交易信号
        
        Parameters:
        data: 历史价格数据
        
        Returns:
        Signal: 交易信号
        """
        if len(data) < self.period:
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.HOLD,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1]
            )
        
        # 计算布林带
        rolling_mean = data['close'].rolling(window=self.period).mean()
        rolling_std = data['close'].rolling(window=self.period).std()
        upper_band = rolling_mean + (rolling_std * self.std_dev)
        lower_band = rolling_mean - (rolling_std * self.std_dev)
        
        current_price = data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = rolling_mean.iloc[-1]
        
        prev_price = data['close'].iloc[-2]
        prev_upper = upper_band.iloc[-2]
        prev_lower = lower_band.iloc[-2]
        
        # 生成信号
        if prev_price <= prev_lower and current_price > current_lower:
            # 价格从下轨反弹，买入信号
            reason = f"价格({current_price:.2f})突破下轨({current_lower:.2f})"
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.BUY,
                price=current_price,
                timestamp=data.index[-1],
                strength=min(1.0, (current_lower - current_price) / current_lower),
                reason=reason
            )
        elif prev_price >= prev_upper and current_price < current_upper:
            # 价格从上轨回落，卖出信号
            reason = f"价格({current_price:.2f})跌破上轨({current_upper:.2f})"
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.SELL,
                price=current_price,
                timestamp=data.index[-1],
                strength=min(1.0, (current_price - current_upper) / current_upper),
                reason=reason
            )
        elif current_price > current_upper:
            # 价格突破上轨，追高买入（较强势）
            reason = f"价格({current_price:.2f})创新高"
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.BUY,
                price=current_price,
                timestamp=data.index[-1],
                strength=min(1.0, (current_price - current_upper) / current_upper),
                reason=reason
            )
        else:
            return Signal(
                symbol=data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown',
                signal_type=SignalType.HOLD,
                price=data['close'].iloc[-1],
                timestamp=data.index[-1]
            )


class MomentumStrategy(Strategy):
    """动量策略"""
    
    def __init__(self, period: int = 90, atr_period: int = 20, name: str = "动量策略"):
        """
        初始化动量策略
        
        Parameters:
        period: 动量计算周期
        atr_period: ATR计算周期
        name: 策略名称
        """
        super().__init__(name)
        self.period = period
        self.atr_period = atr_period
        self.full_data = None
    
    def set_full_data(self, data: pd.DataFrame):
        """
        设置完整数据集
        
        Parameters:
        data: 完整的历史价格数据
        """
        self.full_data = data.copy()
    
    def calculate_momentum(self, closes: pd.Series) -> Tuple[float, float, float]:
        """
        计算动量值及相关统计指标
        
        Parameters:
        closes: 收盘价序列
        
        Returns:
        Tuple[float, float, float]: (动量值, R平方, 斜率)
        """
        if len(closes) < self.period:
            return 0.0, 0.0, 0.0
        
        # 计算对数收益率
        log_prices = np.log(closes.values)
        x = np.arange(len(log_prices))
        
        # 使用线性回归拟合趋势线
        slope, intercept, r_value, p_value, std_err = linregress(x, log_prices)
        
        # 计算决定系数R^2
        r_squared = r_value ** 2
        
        # 年化斜率并乘以R平方作为动量值
        # 年化因子为252个交易日
        annualized_return = np.power(np.exp(slope), 252)
        momentum_value = annualized_return * r_squared
        
        return momentum_value, r_squared, slope
    
    def calculate_atr(self, data: pd.DataFrame) -> float:
        """
        计算ATR指标
        
        Parameters:
        data: 价格数据
        
        Returns:
        float: ATR值
        """
        if len(data) < self.atr_period:
            return 0.0
        
        # 使用pandas向量化操作提高效率
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算真实波幅TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR（简单移动平均）
        atr = tr.rolling(window=self.atr_period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        生成交易信号
        
        Parameters:
        data: 历史价格数据
        
        Returns:
        Signal: 交易信号
        """
        symbol = data['symbol'].iloc[-1] if 'symbol' in data.columns else 'unknown'
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        
        # 检查数据长度是否足够
        required_length = max(self.period, self.atr_period) + 1
        if len(data) < required_length:
            debug_info = f"数据不足: 当前{len(data)}条, 需要{required_length}条"
            print(f"[{self.name}] {debug_info}")
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=current_price,
                timestamp=timestamp,
                reason="数据不足"
            )
        
        # 计算动量指标
        momentum_value, r_squared, slope = self.calculate_momentum(data['close'].tail(self.period))
        
        # 计算波动率指标
        atr_value = self.calculate_atr(data.tail(self.atr_period + 1))
        
        # 计算100日移动平均线
        ma100_series = data['close'].rolling(window=100, min_periods=50).mean()
        ma100 = ma100_series.iloc[-1] if len(ma100_series) >= 100 else data['close'].mean()
        
        # 获取当前持仓
        current_position = self.position.get(symbol, 0)
        
        # 调试信息输出
        debug_msg = (
            f"[{self.name}] "
            f"价格:{current_price:.2f}, "
            f"动量值:{momentum_value:.4f}, "
            f"R²:{r_squared:.3f}, "
            f"斜率:{slope:.6f}, "
            f"ATR:{atr_value:.2f}, "
            f"MA100:{ma100:.2f}, "
            f"持仓:{current_position}"
        )
        print(debug_msg)
        
        # 生成交易信号逻辑
        # 买入条件：动量向上且价格在均线之上
        if current_price > ma100 and momentum_value > 1.02:
            reason = f"动量强劲 (值:{momentum_value:.4f}, R²:{r_squared:.3f}), 价格高于100日均线 ({current_price:.2f}>{ma100:.2f})"
            strength = min(1.0, (momentum_value - 1) * 10)  # 增强信号强度
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                timestamp=timestamp,
                strength=strength,
                reason=reason
            )
        
        # 卖出条件：多种情况都可能导致卖出
        elif (current_price < ma100 or 
              momentum_value < 0.98 or 
              (current_position > 0 and momentum_value < 1.0)):
            
            # 只有当有持仓时才产生卖出信号
            if current_position > 0:
                if current_price < ma100:
                    reason = f"价格跌破100日均线 ({current_price:.2f}<{ma100:.2f}), 触发止损"
                elif momentum_value < 0.98:
                    reason = f"动量转弱 (值:{momentum_value:.4f}), 趋势反转"
                else:
                    reason = f"动量减弱 (值:{momentum_value:.4f}), 获利了结"
                
                strength = min(1.0, (1 - momentum_value) * 10)
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    timestamp=timestamp,
                    strength=strength,
                    reason=reason
                )
            else:
                reason = f"满足卖出条件但无持仓"
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    price=current_price,
                    timestamp=timestamp,
                    reason=reason
                )
        
        # 持有状态
        else:
            holding_reason = f"动量稳定 (值:{momentum_value:.4f})"
            if current_position > 0:
                holding_reason += f", 当前持仓:{current_position}"
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=current_price,
                timestamp=timestamp,
                reason=holding_reason
            )


class SentimentStrategy(Strategy):
    """
    情绪指标策略
    通过综合多个维度的情绪指标，识别市场情绪的极端情况，
    在市场极度悲观时买入，极度乐观时卖出
    """
    
    def __init__(self, name: str = "情绪指标策略", index_code: str = "000001.SH"):
        """
        初始化情绪指标策略
        
        Parameters:
        name: 策略名称
        index_code: 对应指数代码，默认为上证指数
        """
        super().__init__(name)
        self.index_code = index_code
        self.full_data = None  # 存储完整数据集
        self.indicators_saved = False  # 标记是否已保存指标

    def set_full_data(self, data: pd.DataFrame):
        """
        设置完整数据集
        
        Parameters:
        data: 完整的历史价格数据
        """
        self.full_data = data.copy()

    @staticmethod
    # 定义通达信 SMA 函数（放在类外或工具模块中）
    def _sma(src: pd.Series, n: int, m: int) -> pd.Series:
        result = np.zeros_like(src, dtype=np.float64)
        # 处理 NaN：找到第一个非 NaN 位置
        first_valid = src.first_valid_index()
        if first_valid is None:
            return pd.Series(np.nan, index=src.index)
    
        idx_pos = src.index.get_loc(first_valid)
        result[:idx_pos] = np.nan
        result[idx_pos] = src.iloc[idx_pos]
    
        for i in range(idx_pos + 1, len(src)):
            if pd.isna(src.iloc[i]):
                result[i] = np.nan
            else:
                prev_sma = result[i - 1]
                if pd.isna(prev_sma):
                    result[i] = src.iloc[i]  # 退化为当前值
                else:
                    result[i] = (m * src.iloc[i] + (n - m) * prev_sma) / n
        return pd.Series(result, index=src.index)

    @staticmethod
    def _filter_signal(condition: pd.Series, n: int) -> pd.Series:
        """
        模拟通达信 FILTER(COND, N)：
        当 COND 为 True 时，未来 N 个周期内不再触发，第 N+1 周期可再次触发。
    
        Parameters:
            condition: 布尔 Series（信号候选）
            n: 过滤周期数
    
        Returns:
            filtered: 布尔 Series（实际触发信号）
        """
        condition = condition.copy()
        filtered = pd.Series(False, index=condition.index)
        last_triggered = -n  # 上次触发位置（索引偏移）

        for i in range(len(condition)):
            if condition.iloc[i] and (i - last_triggered) >= n:
                filtered.iloc[i] = True
                last_triggered = i

        return filtered

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算情绪指标所需的所有指标
        
        Parameters:
        data: 包含股票和指数数据的DataFrame
        
        Returns:
        DataFrame: 添加了所有指标的DataFrame
        """
        df = data.copy()
        
        # 确保数值类型
        cols_to_check = ['open', 'high', 'low', 'close', 'pre_close', 'change', 
                        'pct_chg', 'vol', 'amount']
        # 检查是否有指数数据列
        has_index_data = 'high_index' in df.columns and 'low_index' in df.columns and 'close_index' in df.columns
        if has_index_data:
            cols_to_check.extend(['high_index', 'low_index', 'close_index'])
            
        for col in cols_to_check:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # === 核心指标计算 ===
        # X_1: (CLOSE*2+HIGH+LOW)/4*10
        df['X_1'] = (df['close'] * 2 + df['high'] + df['low']) / 4 * 10
        
        # X_2: EMA(X_1,13)-EMA(X_1,34)
        df['X_2'] = df['X_1'].ewm(span=13, adjust=False).mean() - df['X_1'].ewm(span=34, adjust=False).mean()
        
        # X_3: EMA(X_2,5)
        df['X_3'] = df['X_2'].ewm(span=5, adjust=False).mean()
        
        # X_4: 2*(X_2-X_3)*5.5
        df['X_4'] = 2 * (df['X_2'] - df['X_3']) * 5.5
        
        # X_5/X_6: 正负分离
        df['X_5'] = np.where(df['X_4'] <= 0, df['X_4'], 0)
        df['X_6'] = np.where(df['X_4'] >= 0, df['X_4'], 0)
        
        # 如果没有指数数据，则使用默认值填充相关指标
        if not has_index_data:
            # 使用默认值填充指数相关指标
            df['high_index'] = df['high']
            df['low_index'] = df['low']
            df['close_index'] = df['close']
            # 只在调试模式下打印警告
            # print("警告: 缺少指数数据，使用个股数据代替")
        
        # X_7: 指数相对位置
        df['HHV_indexH_8'] = df['high_index'].rolling(8, min_periods=1).max()
        df['LLV_indexL_8'] = df['low_index'].rolling(8, min_periods=1).min()
        denominator = df['HHV_indexH_8'] - df['LLV_indexL_8']
        df['X_7'] = (df['HHV_indexH_8'] - df['close_index']) / denominator * 8
        df['X_7'] = df['X_7'].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        
        # X_8: 指数动量
        df['SMA_X7_18'] = self._sma(df['X_7'], n=18, m=1)
        df['J_like'] = 3 * df['X_7'] - 2 * df['SMA_X7_18']
        df['X_8'] = df['J_like'].ewm(span=5, adjust=False).mean()

        
        # X_9: 指数超买超卖
        df['X_9'] = np.where((denominator != 0) & (~np.isnan(denominator)),
                            (df['close_index'] - df['LLV_indexL_8']) / denominator * 10,
                            0)
        
        # X_10-X_15: 指数趋势分析
        df['X_10'] = (df['close_index'] * 2 + df['high_index'] + df['low_index']) / 4
        df['X_11'] = df['X_10'].ewm(span=13, adjust=False).mean() - df['X_10'].ewm(span=34, adjust=False).mean()
        df['X_12'] = df['X_11'].ewm(span=3, adjust=False).mean()
        df['X_13'] = (df['X_11'] - df['X_12']) / 2
        df['X_14'] = np.where(df['X_13'] >= 0, df['X_13'], 0)
        df['X_15'] = np.where(df['X_13'] <= 0, df['X_13'], 0)
        
        # 情绪数值核心计算
        df['LLV_LOW_55'] = df['low'].rolling(55, min_periods=1).min()
        df['HHV_HIGH_55'] = df['high'].rolling(55, min_periods=1).max()
        denominator_55 = df['HHV_HIGH_55'] - df['LLV_LOW_55']
        # 更健壮的 RSV 计算（保留 NaN 比填 0 更合理）
        df['RSV'] = (df['close'] - df['LLV_LOW_55']) / denominator_55 * 100
        df['RSV'] = df['RSV'].replace([np.inf, -np.inf], np.nan)
        df['RSV'] = df['RSV'].clip(lower=0, upper=100)  # 理论范围 [0,100]
        
        # 三重平滑处理
        df['SMA1'] = self._sma(df['RSV'], n=5, m=1)      # = SMA(RSV,5,1)
        df['SMA2'] = self._sma(df['SMA1'], n=3, m=1)     # = SMA(SMA1,3,1)
        df['X_16'] = 3 * df['SMA1'] - 2 * df['SMA2']
        
        # 情绪数值 (核心指标)
        df['ZZZ'] = df['X_16'].ewm(span=3, adjust=False).mean()
        df['X_17'] = df['ZZZ'].pct_change() * 100 # 百分比变化

        # === 信号生成 ===
        # === 即将反弹（用于绘图，非信号）===
        df['即将反弹_plot'] = np.where(df['ZZZ'] <= 13, 20, np.nan)  # 高度20，通达信用 STICKLINE(0,20)

        # === X_18: ZZZ <= 13 且 FILTER(..., 15) ===
        cond_x18 = df['ZZZ'] <= 13
        df['X_18'] = self._filter_signal(cond_x18, n=15) # 即将反弹

        # === 开始反弹（用于绘图）===
        cond_start_rebound = (df['ZZZ'] <= 13) & (df['X_17'] > 13)
        df['开始反弹_plot'] = np.where(cond_start_rebound, 50, np.nan)  # 高度50

        # === X_19: ZZZ<=13 AND X_17>13 AND FILTER(..., 10) ===
        df['X_19'] = self._filter_signal(cond_start_rebound, n=10) #开始反弹

        # === 卖临界（用于绘图）===
        cond_sell_critical = (df['ZZZ'] > 90) & (df['ZZZ'] > df['ZZZ'].shift(1))
        df['卖临界_plot'] = np.where(cond_sell_critical, 95, np.nan)  # 通达信画在 95~100 区间

        # === 风险信号: FILTER(ZZZ>90 AND ZZZ<REF(ZZZ,1) AND X_6<REF(X_6,1), 8) 持续下降 ===
        cond_risk = (
            (df['ZZZ'] > 90) &
            (df['ZZZ'] < df['ZZZ'].shift(1)) &   
            (df['X_6'] < df['X_6'].shift(1))
        )
        df['风险信号'] = self._filter_signal(cond_risk, n=8) #风险信号

        # === X_20: ZZZ>=90 AND X_17 AND FILTER(..., 10) ===
        # 注意：原公式 "X_17" 应理解为 "X_17 存在且非零"，但通常指 "X_17 有定义"
        # 更合理解释：ZZZ >= 90 且 ZZZ 开始下降（即 X_17 < 0），但原式写法模糊
        # 根据上下文，推测应为：ZZZ >= 90 且出现拐点（类似风险信号）
        # 但按字面：X_17 是数值，不能直接做布尔。此处按常见用法修正为：
        cond_x20 = (df['ZZZ'] >= 90) & (df['X_17'].notna())  # 或更严格：& (df['X_17'] < 0)
        # ⚠️ 建议确认原意！这里保守处理为只要 ZZZ>=90 且 X_17 有值就满足
        df['X_20'] = self._filter_signal(cond_x20, n=10)

        # 最终买卖信号
        df['buy_signal'] = (df['X_19'])
        df['sell_signal'] = (df['风险信号']) 

        return df

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        根据情绪指标生成交易信号
        
        Parameters:
        data: 包含股票和指数数据的DataFrame
        
        Returns:
        Signal: 交易信号
        """
        symbol = data.iloc[-1]['symbol'] if 'symbol' in data.columns else 'unknown'
        price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        
        # 检查是否拥有完整数据集且尚未保存指标
        if (self.full_data is not None and 
            len(self.full_data) > 360 and  # 接近完整数据长度
            not self.indicators_saved):
            try:
                # 使用完整数据集计算指标
                df_with_indicators = self._calculate_indicators(self.full_data)
                
                # 创建indicators目录（如果不存在）
                import os
                if not os.path.exists('indicators'):
                    os.makedirs('indicators')
                
                # 生成文件名
                symbol = df_with_indicators['symbol'].iloc[0] if 'symbol' in df_with_indicators.columns else 'unknown'
                start_date = df_with_indicators.index[0].strftime('%Y%m%d') if len(df_with_indicators) > 0 else 'unknown'
                end_date = df_with_indicators.index[-1].strftime('%Y%m%d') if len(df_with_indicators) > 0 else 'unknown'
                filename = f"indicators/{symbol}_{start_date}_{end_date}_indicators.csv"
                
                # 保存指标数据
                df_with_indicators.to_csv(filename, encoding='utf-8-sig')
                print(f"指标数据已保存至: {filename}")
                print(f"保存的指标数据行数: {len(df_with_indicators)}")
                self.indicators_saved = True
            except Exception as e:
                print(f"保存指标数据时出错: {e}")
        
        # 为当前数据计算指标（用于生成信号）
        df_with_indicators = self._calculate_indicators(data)
        
        # 获取最新的信号
        latest_row = df_with_indicators.iloc[-1]
        prev_row = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else None
        
        # 判断买入信号
        if latest_row['buy_signal']:
            reason = f"满足情绪指标买入条件，当前情绪值:{latest_row['ZZZ']:.2f}"
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=price,
                timestamp=timestamp,
                reason=reason
            )
        
        # 判断卖出信号
        if latest_row['sell_signal']:
            reason = f"满足情绪指标卖出条件，当前情绪值:{latest_row['ZZZ']:.2f}"
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=price,
                timestamp=timestamp,
                reason=reason
            )
        
        # 默认持有信号
        return Signal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            price=price,
            timestamp=timestamp,
            reason="未满足交易条件"
        )
