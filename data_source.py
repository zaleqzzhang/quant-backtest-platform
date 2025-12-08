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
        self.pro = None
        
        # 尝试初始化tushare
        try:
            import tushare as ts
            if self.token:
                ts.set_token(self.token)
                self.pro = ts.pro_api()
                self.available = True
                print("Tushare数据源初始化成功")
            else:
                print("警告: 未提供Tushare Token，将使用模拟数据")
        except ImportError:
            print("警告: 未安装tushare库或导入失败，将使用模拟数据")

    def get_history_data(self, symbol: str, start_date: str, end_date: str, index_symbol: str = None, adj_mode: str = 'none') -> pd.DataFrame:
        """
        获取历史数据（真实实现）
        
        Parameters:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        index_symbol: 指数代码，如果提供则同时获取指数数据
        adj_mode: 复权模式 ('none', 'qfq'-前复权)
        
        Returns:
        DataFrame: 包含个股和指数数据的DataFrame
        """
        # 首先检查数据源是否可用
        if not self.available:
            # 模拟数据用于演示
            return self._generate_mock_data(symbol, start_date, end_date, index_symbol)
            
        try:
            # 处理复权参数
            adj_param = None
            if adj_mode == 'qfq':
                adj_param = 'qfq'  # 前复权
            # 其他情况不复权
                
            # 调用tushare API获取真实数据
            import tushare as ts
            if adj_param:
                # 如果需要复权数据
                df = ts.pro_bar(ts_code=symbol, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''), adj=adj_param)
            else:
                # 不复权数据
                df = self.pro.daily(ts_code=symbol, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
            
            # 检查返回数据
            if df is None or df.empty:
                print(f"警告: Tushare未返回{symbol}的数据，使用模拟数据")
                return self._generate_mock_data(symbol, start_date, end_date, index_symbol)
            
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
            
            # 如果提供了指数代码，则同时获取指数数据
            if index_symbol:
                try:
                    index_df = self.pro.index_daily(ts_code=index_symbol, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
                    if index_df is not None and not index_df.empty:
                        # 处理指数数据
                        index_df = index_df.rename(columns={
                            'trade_date': 'date',
                            'vol': 'index_volume',
                            'open': 'open_index',
                            'high': 'high_index',
                            'low': 'low_index',
                            'close': 'close_index'
                        })
                        
                        # 转换日期格式
                        index_df['date'] = pd.to_datetime(index_df['date'])
                        index_df = index_df.sort_values('date')
                        
                        # 设置日期为索引
                        index_df = index_df.set_index('date')
                        
                        # 合并个股和指数数据
                        df = pd.merge(df, index_df[['open_index', 'high_index', 'low_index', 'close_index', 'index_volume']], 
                                     left_index=True, right_index=True, how='left')
                        
                        print(f"已合并{symbol}个股数据和{index_symbol}指数数据")
                    else:
                        print(f"警告: 未能获取{index_symbol}指数数据")
                except Exception as e:
                    print(f"获取指数数据时出错: {e}")
            
            # 重新排序列
            column_order = ['open', 'high', 'low', 'close', 'volume']
            # 如果有指数数据，也加入列顺序
            if 'open_index' in df.columns:
                column_order.extend(['open_index', 'high_index', 'low_index', 'close_index', 'index_volume'])
            
            # 确保所有需要的列都存在
            for col in column_order:
                if col not in df.columns and col in ['open', 'high', 'low', 'close', 'volume']:
                    print(f"警告: 数据缺少必要列 {col}，使用模拟数据")
                    return self._generate_mock_data(symbol, start_date, end_date, index_symbol)
                    
            df = df[column_order]
            
            return df
            
        except Exception as e:
            # 捕获所有异常，包括网络错误、认证失败等
            print(f"获取Tushare数据时出错: {e}，使用模拟数据")
            return self._generate_mock_data(symbol, start_date, end_date, index_symbol)

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

    def _generate_mock_data(self, symbol: str, start_date: str, end_date: str, index_symbol: str = None) -> pd.DataFrame:
        """
        生成模拟数据
        
        Parameters:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        index_symbol: 指数代码，如果提供则同时生成指数数据
        """
        print(f"使用模拟数据进行回测: {symbol}" + (f" 和指数 {index_symbol}" if index_symbol else ""))
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # 生成个股数据
        stock_prices = np.random.lognormal(0, 0.05, len(dates)).cumprod() * 100
        stock_open_prices = stock_prices * np.random.uniform(0.99, 1.01, len(dates))
        stock_high_prices = np.maximum(stock_prices, stock_open_prices) * np.random.uniform(1.0, 1.05, len(dates))
        stock_low_prices = np.minimum(stock_prices, stock_open_prices) * np.random.uniform(0.95, 1.0)
        stock_close_prices = stock_prices
        stock_volume = np.random.randint(100000, 1000000, len(dates))
        
        data_dict = {
            'date': dates,
            'open': stock_open_prices,
            'high': stock_high_prices,
            'low': stock_low_prices,
            'close': stock_close_prices,
            'volume': stock_volume
        }
        
        # 如果提供了指数代码，则生成指数数据
        if index_symbol:
            # 生成指数数据（与个股数据有一定相关性，但不完全相同）
            index_prices = stock_prices * np.random.uniform(0.8, 1.2)  # 与个股价格相关但有差异
            index_open_prices = index_prices * np.random.uniform(0.99, 1.01, len(dates))
            index_high_prices = np.maximum(index_prices, index_open_prices) * np.random.uniform(1.0, 1.05, len(dates))
            index_low_prices = np.minimum(index_prices, index_open_prices) * np.random.uniform(0.95, 1.0)
            index_close_prices = index_prices
            index_volume = np.random.randint(1000000, 10000000, len(dates))  # 指数通常有更高的成交量
            
            # 添加指数数据到字典
            data_dict.update({
                'open_index': index_open_prices,
                'high_index': index_high_prices,
                'low_index': index_low_prices,
                'close_index': index_close_prices,
                'index_volume': index_volume
            })
        
        df = pd.DataFrame(data_dict)
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