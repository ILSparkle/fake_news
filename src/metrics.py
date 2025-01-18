from typing import Dict, List
import logging
from pathlib import Path
import json

class MetricsCalculator:
    def __init__(self):
        """初始化指标计算器"""
        self.total_metrics = {
            'true_positive': 0,  # 正确预测为真
            'true_negative': 0,  # 正确预测为假
            'false_positive': 0,  # 错误预测为真
            'false_negative': 0,  # 错误预测为假
            'total_processed': 0,
            'total_correct': 0
        }
        self.error_predictions: List[Dict] = []

    def update_metrics(self, result: Dict) -> Dict:
        """更新单个结果的指标统计
        
        Args:
            result: 预测结果字典
            
        Returns:
            当前批次的指标统计
        """
        batch_metrics = {
            'true_positive': 0,
            'true_negative': 0,
            'false_positive': 0,
            'false_negative': 0,
            'total': 1,
            'correct': 0
        }
        
        actual_label = result['actual_label']
        prediction = result['prediction']
        
        # 记录错误预测
        if not result['is_correct']:
            error_info = {
                'news_id': result['news_id'],
                'actual_label': '真实' if actual_label == 0 else '虚假',
                'predicted_label': '真实' if prediction == 0 else '虚假',
                'confidence': result['confidence'],
                'is_retrieved': result.get('is_retrieved', False)
            }
            self.error_predictions.append(error_info)
        
        # 更新指标统计
        self.total_metrics['total_processed'] += 1
        if actual_label == 0 and prediction == 0:
            batch_metrics['true_positive'] = 1
            self.total_metrics['true_positive'] += 1
            batch_metrics['correct'] = 1
            self.total_metrics['total_correct'] += 1
        elif actual_label == 1 and prediction == 1:
            batch_metrics['true_negative'] = 1
            self.total_metrics['true_negative'] += 1
            batch_metrics['correct'] = 1
            self.total_metrics['total_correct'] += 1
        elif actual_label == 1 and prediction == 0:
            batch_metrics['false_positive'] = 1
            self.total_metrics['false_positive'] += 1
        elif actual_label == 0 and prediction == 1:
            batch_metrics['false_negative'] = 1
            self.total_metrics['false_negative'] += 1
            
        return batch_metrics

    def calculate_metrics(self, metrics: Dict) -> Dict:
        """计算评估指标
        
        Args:
            metrics: 包含TP、TN、FP、FN的指标字典
            
        Returns:
            包含准确率、精确率、召回率、F1分数的字典
        """
        precision = (metrics['true_positive'] / 
                    (metrics['true_positive'] + metrics['false_positive'])
                    if (metrics['true_positive'] + metrics['false_positive']) > 0 
                    else 0)
        
        recall = (metrics['true_positive'] / 
                 (metrics['true_positive'] + metrics['false_negative'])
                 if (metrics['true_positive'] + metrics['false_negative']) > 0 
                 else 0)
        
        f1 = (2 * precision * recall / (precision + recall) 
              if (precision + recall) > 0 
              else 0)
        
        accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save_results(self, results_dir: Path) -> None:
        """保存评估结果
        
        Args:
            results_dir: 结果保存目录
        """
        # 计算总体指标
        total_stats = self.calculate_metrics({
            'true_positive': self.total_metrics['true_positive'],
            'true_negative': self.total_metrics['true_negative'],
            'false_positive': self.total_metrics['false_positive'],
            'false_negative': self.total_metrics['false_negative'],
            'total': self.total_metrics['total_processed'],
            'correct': self.total_metrics['total_correct']
        })
        
        # 保存错误预测结果
        if self.error_predictions:
            error_file = results_dir / "error_predictions.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_errors': len(self.error_predictions),
                    'error_cases': self.error_predictions
                }, f, ensure_ascii=False, indent=2)
            logging.info(f"错误预测结果已保存到: {error_file}")
            logging.info(f"总计 {len(self.error_predictions)} 条错误预测")
        
        # 保存总体结果
        with open(results_dir / "final_results.json", 'w', encoding='utf-8') as f:
            json.dump({
                'total_metrics': self.total_metrics,
                'final_statistics': {
                    **total_stats,
                    'total_errors': len(self.error_predictions)
                },
                'error_distribution': {
                    'false_positives': self.total_metrics['false_positive'],
                    'false_negatives': self.total_metrics['false_negative']
                }
            }, f, ensure_ascii=False, indent=2)
        
        # 输出总体评估结果
        logging.info(
            f"\n总体评估结果:\n"
            f"准确率: {total_stats['accuracy']:.2%} "
            f"({self.total_metrics['total_correct']}/{self.total_metrics['total_processed']})\n"
            f"精确率: {total_stats['precision']:.2%}\n"
            f"召回率: {total_stats['recall']:.2%}\n"
            f"F1分数: {total_stats['f1']:.2%}"
        ) 