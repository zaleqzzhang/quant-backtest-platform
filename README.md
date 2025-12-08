# 量化策略回测平台

一个功能完整的量化策略回测平台，支持多种数据源和策略，提供详细的回测结果分析。

## 功能特点

1. **多数据源支持**：
   - Tushare
   - Longport
   - 易于扩展其他数据源

2. **多种策略实现**：
   - 均线交叉策略
   - RSI相对强弱指数策略
   - 布林带策略
   - 易于添加自定义策略

3. **全面的回测分析**：
   - 收益曲线
   - 回撤分析
   - 买卖点标记
   - 关键绩效指标（KPI）

4. **可视化展示**：
   - 策略表现对比
   - 交易信号标注
   - 各项指标图表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 目录结构

```
backtest/
├── data_source.py      # 数据源模块
├── strategy.py         # 策略模块
├── backtest_engine.py  # 回测引擎
├── visualization.py    # 可视化模块
├── main.py             # 主程序入口
├── requirements.txt    # 依赖列表
└── README.md           # 说明文档
```

## 快速开始

1. 修改 [main.py](file:///Users/zhangqingzheng/code/backtest/main.py) 中的数据源配置（如Tushare token）
2. 运行主程序：

```bash
python main.py
```

## 核心模块说明

### 数据源模块 (data_source.py)

提供了统一的数据接口，目前支持Tushare和Longport两种数据源。通过工厂函数 `get_data_source()` 可以方便地获取所需数据源实例。

### 策略模块 (strategy.py)

实现了几种常见的量化策略：
- 均线交叉策略：当短期均线上穿长期均线时买入，下穿时卖出
- RSI策略：当RSI低于超卖区时买入，高于超买区时卖出
- 布林带策略：当价格突破布林带上轨时卖出，突破下轨时买入

所有策略都继承自抽象基类 `Strategy`，易于扩展新的策略。

### 回测引擎 (backtest_engine.py)

核心模块，负责执行策略回测，计算各项绩效指标：
- 总收益率
- 年化收益率
- 最大回撤
- 夏普比率
- 胜率
- 盈利因子

### 可视化模块 (visualization.py)

提供丰富的可视化功能：
- 权益曲线图
- 回撤曲线图
- 买卖点标记图
- 策略性能对比图
- 回测结果摘要报表

## 自定义策略

要添加新的策略，只需继承 `Strategy` 基类并实现 `generate_signal()` 方法即可。

## 扩展数据源

要添加新的数据源，只需继承 `DataSource` 基类并实现 `get_history_data()` 和 `get_realtime_data()` 方法。

## 注意事项

1. 本平台目前为单股票回测，可进一步扩展为多股票组合策略
2. 交易成本已考虑手续费，可根据实际需求调整
3. 回测采用理想化撮合，未考虑滑点等因素