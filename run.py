import re
from dotenv import load_dotenv
import asyncio
import json
from pathlib import Path
import logging
import argparse
from datetime import datetime
import os
from src.metrics import MetricsCalculator
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from model import DualBertForSequenceClassification
import numpy as np
from tqdm import tqdm

load_dotenv()
from src.dataset import FakeNewsDataset
from src.search import SearchAPI
from src.chat import ChatAPI
from src.crawler import WebCrawler
from src.prompt import keyword_prompt, verify_prompt, initial_score_prompt

def setup_logging(level: str, log_dir: str = 'logs') -> None:
    """配置日志级别和输出

    Args:
        level: 日志级别，可选值：DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # 生成日志文件名（包含时间戳）(
    log_file = log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # 配置根日志记录器
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # 输出到控制台
            logging.StreamHandler(),
            # 输出到文件
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    # 记录启动信息
    logging.info(f"日志文件已创建: {log_file}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='新闻真实性验证系统')
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='日志级别'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='日志文件保存目录'
    )
    return parser.parse_args()

class NewsVerificationSystem:
    def __init__(self, model_name: str = "bert-base-uncased", confidence_threshold: float = 0.8):
        """初始化新闻验证系统
        
        Args:
            model_name: BERT模型名称
            confidence_threshold: 置信度阈值，低于此值将触发检索增强
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = DualBertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        
        # 初始化API客户端
        self.search_api = SearchAPI()
        self.chat_api = ChatAPI()
        self.crawler = WebCrawler()
        
    def prepare_input(self, news_content: str, retrieved_content: str = "") -> Dict[str, torch.Tensor]:
        """准备模型输入
        
        Args:
            news_content: 新闻内容
            retrieved_content: 检索到的内容（可选）
        """
        # 编码新闻文本
        news_encoding = self.tokenizer(
            news_content,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 如果没有检索内容，使用空字符串
        comments_encoding = self.tokenizer(
            retrieved_content if retrieved_content else "",
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'news_input_ids': news_encoding['input_ids'].to(self.device),
            'news_attention_mask': news_encoding['attention_mask'].to(self.device),
            'comments_input_ids': comments_encoding['input_ids'].to(self.device),
            'comments_attention_mask': comments_encoding['attention_mask'].to(self.device)
        }
        
    async def verify_news(self, news_id: str, content: str, label: int = None) -> Dict:
        """验证新闻真实性
        
        Args:
            news_id: 新闻ID
            content: 新闻内容
            label: 实际标签（可选，用于评估）
        """
        # 首先进行基础预测
        inputs = self.prepare_input(content)
        with torch.no_grad():
            _, logits = self.model(**inputs)
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            confidence = confidence.item()
            prediction = prediction.item()
        
        # 如果置信度低于阈值，进行检索增强
        if confidence < self.confidence_threshold:
            logging.info(f"新闻 {news_id} 置信度较低 ({confidence:.2f})，启动检索增强")
            
            # 获取关键词
            keywords = await self.chat_api.chat(content, keyword_prompt)
            
            # 搜索相关新闻
            search_results = await self.search_api.search(keywords)
            processed_results = await self.crawler.process_search_results(search_results)
            
            # 构建检索内容
            retrieved_content = "\n".join([
                f"{result['title']}\n{result['content']}"
                for result in processed_results
            ])
            
            # 重新预测
            inputs = self.prepare_input(content, retrieved_content)
            with torch.no_grad():
                _, logits = self.model(**inputs)
                probs = torch.softmax(logits, dim=1)
                confidence, prediction = torch.max(probs, dim=1)
                confidence = confidence.item()
                prediction = prediction.item()
        
        result = {
            'news_id': str(news_id),
            'prediction': prediction,
            'confidence': confidence,
            'is_retrieved': confidence < self.confidence_threshold
        }
        
        if label is not None:
            result['actual_label'] = label
            result['is_correct'] = prediction == label
            
        return result

async def train_model(model_path: str, train_loader: DataLoader, 
                     val_loader: DataLoader, num_epochs: int = 5):
    """训练模型"""
    system = NewsVerificationSystem(model_path)
    optimizer = torch.optim.AdamW(system.model.parameters(), lr=2e-5)
    
    best_val_acc = 0
    for epoch in range(num_epochs):
        # 训练阶段
        system.model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            loss, logits = system.model(
                news_input_ids=batch['news_input_ids'].to(system.device),
                news_attention_mask=batch['news_attention_mask'].to(system.device),
                comments_input_ids=batch['comments_input_ids'].to(system.device),
                comments_attention_mask=batch['comments_attention_mask'].to(system.device),
                labels=batch['labels'].to(system.device)
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += batch['labels'].size(0)
            train_correct += (predicted == batch['labels'].to(system.device)).sum().item()
        
        # 验证阶段
        system.model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                _, logits = system.model(
                    news_input_ids=batch['news_input_ids'].to(system.device),
                    news_attention_mask=batch['news_attention_mask'].to(system.device),
                    comments_input_ids=batch['comments_input_ids'].to(system.device),
                    comments_attention_mask=batch['comments_attention_mask'].to(system.device)
                )
                
                _, predicted = torch.max(logits, 1)
                val_total += batch['labels'].size(0)
                val_correct += (predicted == batch['labels'].to(system.device)).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        logging.info(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}, "
                    f"Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            system.model.save_pretrained("best_model")
            logging.info(f"保存最佳模型，验证准确率: {val_acc:.2%}")

async def main(dataset_name: str):
    args = parse_args()
    setup_logging(args.log_level)
    
    # 加载数据集，从os.environ中获取数据集路径
    dataset = FakeNewsDataset(
        news_path=os.environ.get(f"NEWS_PATH", f"dataset/{dataset_name}_news.csv"),
        comment_path=os.environ.get(f"COMMENT_PATH", f"dataset/{dataset_name}_socialcontext.csv")
    )
    
    # 划分训练集和测试集
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 训练模型
    await train_model("bert-base-uncased", train_loader, test_loader)
    
    # 加载最佳模型进行测试
    system = NewsVerificationSystem("best_model")
    metrics_calculator = MetricsCalculator()
    
    # 测试阶段
    system.model.eval()
    for batch in tqdm(test_loader, desc="Testing"):
        for i in range(len(batch['news_id'])):
            result = await system.verify_news(
                batch['news_id'][i],
                batch['news_content'][i],
                batch['labels'][i].item()
            )
            metrics_calculator.update_metrics(result)
    
    # 保存结果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    metrics_calculator.save_results(results_dir)

if __name__ == "__main__":
    dataset = "politifact"
    asyncio.run(main(dataset))
