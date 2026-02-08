# -*- coding: utf-8 -*-
"""测试策略发现引擎"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.strategy.strategy_discovery import train_model, get_discovery_summary

if __name__ == '__main__':
    print("=" * 60)
    print("开始从历史数据中训练策略...")
    print("=" * 60)

    result = train_model(max_stocks=100, force=True)

    if 'error' in result:
        print(f"\n训练失败: {result['error']}")
    else:
        print(f"\n{'=' * 60}")
        print(f"训练完成！")
        print(f"采样股票: {result.get('sample_stocks', 0)}")
        print(f"训练样本: {result.get('total_samples', 0)}")
        print(f"模型准确率: {result.get('accuracy', 0):.1%}")
        print(f"发现策略: {len(result.get('learned_rules', []))} 条")

        print(f"\n--- 特征重要度 Top5 ---")
        for f in result.get('top_features', []):
            print(f"  {f['name']}: {f['importance']:.4f}")

        print(f"\n--- 学到的策略规则 ---")
        for rule in result.get('learned_rules', []):
            bt = rule.get('backtest', {})
            print(f"\n  [{rule.get('grade', 'C')}级] {rule['rule_id']}")
            print(f"  条件: {rule['description']}")
            print(f"  胜率: {bt.get('win_rate', 0):.1f}%  盈亏比: {bt.get('profit_loss_ratio', 0):.2f}")
            print(f"  夏普: {bt.get('sharpe', 0):.2f}  最大回撤: {bt.get('max_drawdown', 0):.1f}%")
            print(f"  平均收益: {bt.get('avg_return', 0):+.2f}%  交易次数: {bt.get('total_trades', 0)}")
