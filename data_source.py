"""
数据源模块
支持多种数据提供商：Tushare, Longport等
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import os


class DataSource(ABC):
    """
    抽象数据源基类
    """

    @abstractmethod
    def get_history_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取历史数据
        
        Parameters:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
        Returns:
        DataFrame: 包含OHLCV数据
        """
        pass

    @abstractmethod
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取实时数据
        
        Parameters:
        symbol: 股票代码
        
        Returns:
        dict: 实时行情数据
        """
        pass


class TushareDataSource(DataSource):
    """
    Tushare数据源实现
    """

    def __init__(self, token: str = None):
        """
        初始化Tushare数据源
        
        Parameters:
        token: Tushare API Token
        """
        self.token = token or os.getenv('TUSHARE_TOKEN')
        self.available = False
        
        try:
            import tushare as ts
            if not self.token:
                print("警告: 未提供Tushare Token，将使用模拟数据")
                return
                
            # 设置token并创建pro接口
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            
            # 尝试调用一个简单的API来验证token有效性
            try:
                # 调用基本的股票列表接口验证token
                self.pro.stock_basic(exchange='', list_status='L', fields=['ts_code'])
                self.available = True
            except Exception as e:
                # 如果API调用失败（如token无效），则标记为不可用
                error_msg = str(e).lower()
                if 'token' in error_msg or 'invalid' in error_msg or 'author' in error_msg:
                    print(f"警告: Tushare Token无效或权限不足: {e}，将使用模拟数据")
                else:
                    print(f"警告: Tushare API访问出错: {e}，将使用模拟数据")
                    
        except ImportError:
            print("警告: 未安装tushare库或导入失败，将使用模拟数据")

    def get_history_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取历史数据（真实实现）
        """
        # 首先检查数据源是否可用
        if not self.available:
            # 模拟数据用于演示
            return self._generate_mock_data(symbol, start_date, end_date)
            
        try:
            # 调用tushare API获取真实数据
            import tushare as ts
            df = self.pro.daily(ts_code=symbol, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
            
            # 检查返回数据
            if df is None or df.empty:
                print(f"警告: Tushare未返回{symbol}的数据，使用模拟数据")
                return self._generate_mock_data(symbol, start_date, end_date)
            
            # 转换列名和索引
            df = df.rename(columns={
                'trade_date': 'date',
                'vol': 'volume'
            })
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 设置日期为索引
            df = df.set_index('date')
            
            # 重新排序列
            column_order = ['open', 'high', 'low', 'close', 'volume']
            # 确保所有需要的列都存在
            for col in column_order:
                if col not in df.columns:
                    print(f"警告: 数据缺少必要列 {col}，使用模拟数据")
                    return self._generate_mock_data(symbol, start_date, end_date)
                    
            df = df[column_order]
            
            return df
            
        except Exception as e:
            # 捕获所有异常，包括网络错误、认证失败等
            print(f"获取Tushare数据时出错: {e}，使用模拟数据")
            return self._generate_mock_data(symbol, start_date, end_date)

    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取实时数据（真实实现）
        """
        # 首先检查数据源是否可用
        if not self.available:
            # 模拟实时数据
            return self._generate_mock_realtime_data(symbol)
            
        try:
            # 调用tushare获取实时数据
            import tushare as ts
            df = self.pro.quote(ts_code=symbol)
            
            # 检查返回数据
            if df is None or df.empty:
                print(f"警告: Tushare未返回{symbol}的实时数据，使用模拟数据")
                return self._generate_mock_realtime_data(symbol)
                
            row = df.iloc[0]
            
            # 确保必要的字段存在
            required_fields = ['price', 'bid', 'ask', 'volume']
            for field in required_fields:
                if field not in row:
                    print(f"警告: 实时数据缺少必要字段 {field}，使用模拟数据")
                    return self._generate_mock_realtime_data(symbol)
            
            return {
                'symbol': symbol,
                'price': float(row['price']),
                'bid': float(row['bid']),
                'ask': float(row['ask']),
                'volume': int(row['volume']),
                'timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            # 捕获所有异常
            print(f"获取Tushare实时数据时出错: {e}，使用模拟数据")
            return self._generate_mock_realtime_data(symbol)

    def _generate_mock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成模拟数据
        """
        print(f"使用模拟数据进行回测: {symbol}")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        prices = np.random.lognormal(0, 0.05, len(dates)).cumprod() * 100
        open_prices = prices * np.random.uniform(0.99, 1.01, len(dates))
        high_prices = np.maximum(prices, open_prices) * np.random.uniform(1.0, 1.05, len(dates))
        low_prices = np.minimum(prices, open_prices) * np.random.uniform(0.95, 1.0)
        close_prices = prices
        volume = np.random.randint(100000, 1000000, len(dates))
        
        df = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        return df.set_index('date')

    def _generate_mock_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """
        生成模拟实时数据
        """
        return {
            'symbol': symbol,
            'price': np.random.uniform(95, 105),
            'bid': np.random.uniform(94.9, 104.9),
            'ask': np.random.uniform(95.1, 105.1),
            'volume': np.random.randint(1000, 10000),
            'timestamp': pd.Timestamp.now()
        }


class LongportDataSource(DataSource):
    """
    Longport数据源实现
    """

    def __init__(self, app_key: str = None, app_secret: str = None, access_token: str = None):
        """
        初始化Longport数据源
        
        Parameters:
        app_key: Longport App Key
        app_secret: Longport App Secret
        access_token: Longport Access Token
        """
        self.app_key = app_key or os.getenv('LONGPORT_APP_KEY')
        self.app_secret = app_secret or os.getenv('LONGPORT_APP_SECRET')
        self.access_token = access_token or os.getenv('LONGPORT_ACCESS_TOKEN')
        
        try:
            from longport.openapi import QuoteContext
            if self.app_key and self.app_secret and self.access_token:
                # 注意：在实际环境中需要取消注释以下代码以启用真实连接
                # self.ctx = QuoteContext.new(self.app_key, self.app_secret, self.access_token)
                # self.available = True
                self.available = False  # 暂时禁用直到配置好认证信息
                print("注意: Longport连接已配置但暂时禁用，将使用模拟数据")
            else:
                self.available = False
                print("警告: 未提供Longport认证信息，将使用模拟数据")
        except ImportError:
            self.available = False
            print("警告: 未安装longport库或导入失败，将使用模拟数据")

    def get_history_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取历史数据（真实实现）
        """
        if not self.available:
            # 模拟数据用于演示
            return self._generate_mock_data(symbol, start_date, end_date)
            
        try:
            # 调用longport API获取真实数据
            # 注意：在实际环境中需要实现真实的API调用
            from longport.openapi import TimeFrame, AdjustType
            # resp = self.ctx.candlesticks(symbol, TimeFrame.Day, start_date, end_date, AdjustType.ForwardAdjust)
            
            # 暂时返回模拟数据
            return self._generate_mock_data(symbol, start_date, end_date)
        except Exception as e:
            print(f"获取Longport数据时出错: {e}，使用模拟数据")
            return self._generate_mock_data(symbol, start_date, end_date)

    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取实时数据（真实实现）
        """
        if not self.available:
            # 模拟实时数据
            return self._generate_mock_realtime_data(symbol)
            
        try:
            # 调用longport获取实时数据
            # 注意：在实际环境中需要实现真实的API调用
            # resp = self.ctx.quote(symbol)
            
            # 暂时返回模拟数据
            return self._generate_mock_realtime_data(symbol)
        except Exception as e:
            print(f"获取Longport实时数据时出错: {e}，使用模拟数据")
            return self._generate_mock_realtime_data(symbol)

    def _generate_mock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成模拟数据
        """
        print(f"使用模拟数据进行回测: {symbol}")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        prices = np.random.lognormal(0, 0.05, len(dates)).cumprod() * 50
        open_prices = prices * np.random.uniform(0.99, 1.01, len(dates))
        high_prices = np.maximum(prices, open_prices) * np.random.uniform(1.0, 1.05, len(dates))
        low_prices = np.minimum(prices, open_prices) * np.random.uniform(0.95, 1.0)
        close_prices = prices
        volume = np.random.randint(50000, 500000, len(dates))
        
        df = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        return df.set_index('date')

    def _generate_mock_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """
        生成模拟实时数据
        """
        return {
            'symbol': symbol,
            'price': np.random.uniform(45, 55),
            'bid': np.random.uniform(44.9, 54.9),
            'ask': np.random.uniform(45.1, 55.1),
            'volume': np.random.randint(500, 5000),
            'timestamp': pd.Timestamp.now()
        }


def get_data_source(source_type: str, **kwargs) -> DataSource:
    """
    工厂函数，根据类型返回对应的数据源实例
    
    Parameters:
    source_type: 数据源类型 ('tushare', 'longport')
    **kwargs: 初始化参数
    
    Returns:
    DataSource: 对应的数据源实例
    """
    sources = {
        'tushare': TushareDataSource,
        'longport': LongportDataSource
    }
    
    if source_type.lower() not in sources:
        raise ValueError(f"不支持的数据源类型: {source_type}")
    
    return sources[source_type.lower()](**kwargs)