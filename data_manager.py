"""
数据管理模块
负责数据的获取、清洗、存储和加载
"""

import pandas as pd
import os
import pickle
from datetime import datetime
from typing import Optional
from data_source import get_data_source


class DataManager:
    """
    数据管理器
    负责统一管理市场数据的获取、存储和访问
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据管理器
        
        Parameters:
        data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        # 确保数据目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _get_data_filename(self, symbol: str, start_date: str, end_date: str, data_source: str) -> str:
        """
        生成数据文件名
        
        Parameters:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        data_source: 数据源类型
        
        Returns:
        str: 数据文件路径
        """
        filename = f"{symbol}_{start_date}_{end_date}_{data_source}.pkl"
        return os.path.join(self.data_dir, filename)
    
    def fetch_and_save_data(self, symbol: str, start_date: str, end_date: str, 
                           data_source_type: str = 'tushare', **kwargs) -> pd.DataFrame:
        """
        获取并保存数据
        
        Parameters:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        data_source_type: 数据源类型
        **kwargs: 数据源配置参数
        
        Returns:
        DataFrame: 获取到的数据
        """
        # 初始化数据源
        data_source_config = {}
        if data_source_type == 'tushare':
            data_source_config['token'] = kwargs.get('token', os.getenv('TUSHARE_TOKEN'))
        elif data_source_type == 'longport':
            data_source_config['app_key'] = kwargs.get('app_key', os.getenv('LONGPORT_APP_KEY'))
            data_source_config['app_secret'] = kwargs.get('app_secret', os.getenv('LONGPORT_APP_SECRET'))
            data_source_config['access_token'] = kwargs.get('access_token', os.getenv('LONGPORT_ACCESS_TOKEN'))
        
        data_source = get_data_source(data_source_type, **data_source_config)
        
        # 获取数据
        data = data_source.get_history_data(symbol, start_date, end_date)
        data['symbol'] = symbol
        
        # 数据清洗
        cleaned_data = self._clean_data(data)
        
        # 保存数据
        data_file = self._get_data_filename(symbol, start_date, end_date, data_source_type)
        self._save_data(cleaned_data, data_file)
        
        return cleaned_data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Parameters:
        data: 原始数据
        
        Returns:
        DataFrame: 清洗后的数据
        """
        # 删除空值
        data = data.dropna()
        
        # 确保列的顺序一致
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        available_columns = [col for col in expected_columns if col in data.columns]
        data = data[available_columns]
        
        # 按日期排序
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        return data
    
    def _save_data(self, data: pd.DataFrame, filepath: str):
        """
        保存数据到文件
        
        Parameters:
        data: 要保存的数据
        filepath: 文件路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_data(self, symbol: str, start_date: str, end_date: str, 
                 data_source_type: str = 'tushare') -> Optional[pd.DataFrame]:
        """
        从文件加载数据
        
        Parameters:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        data_source_type: 数据源类型
        
        Returns:
        DataFrame or None: 加载的数据，如果文件不存在则返回None
        """
        data_file = self._get_data_filename(symbol, start_date, end_date, data_source_type)
        
        if os.path.exists(data_file):
            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                return data
            except Exception as e:
                print(f"加载数据文件时出错: {e}")
                return None
        else:
            return None
    
    def data_exists(self, symbol: str, start_date: str, end_date: str, 
                   data_source_type: str = 'tushare') -> bool:
        """
        检查数据文件是否存在
        
        Parameters:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        data_source_type: 数据源类型
        
        Returns:
        bool: 数据文件是否存在
        """
        data_file = self._get_data_filename(symbol, start_date, end_date, data_source_type)
        return os.path.exists(data_file)
    
    def get_data(self, symbol: str, start_date: str, end_date: str, 
                data_source_type: str = 'tushare', force_update: bool = False, **kwargs) -> pd.DataFrame:
        """
        获取数据（优先从本地加载，如果不存在或强制更新则重新获取）
        
        Parameters:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        data_source_type: 数据源类型
        force_update: 是否强制更新数据
        **kwargs: 数据源配置参数
        
        Returns:
        DataFrame: 数据
        """
        # 如果不强制更新且数据文件存在，则从文件加载
        if not force_update and self.data_exists(symbol, start_date, end_date, data_source_type):
            data = self.load_data(symbol, start_date, end_date, data_source_type)
            if data is not None:
                print(f"从本地文件加载数据: {symbol} ({start_date} 至 {end_date})")
                return data
        
        # 否则重新获取并保存数据
        print(f"获取新数据: {symbol} ({start_date} 至 {end_date})")
        return self.fetch_and_save_data(symbol, start_date, end_date, data_source_type, **kwargs)