import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

@dataclass
class NewsItem:
    news_id: str
    label: int
    title: str
    content: str
    comments: List[str]  # 添加评论列表

class FakeNewsDataset(Dataset):
    def __init__(self, news_path: str | Path, comment_path: str | Path, 
                 tokenizer_path: str = "bert-base-uncased", max_comments: int = 10):
        """初始化假新闻数据集

        Args:
            news_path: 新闻数据CSV文件路径
            comment_path: 评论数据CSV文件路径
            tokenizer_path: 分词器路径或名称
            max_comments: 每条新闻最多保留的评论数量
        """
        self.news_data: Dict[str, NewsItem] = {}
        self.comment_data: Dict[str, CommentItem] = {}
        self.news_to_comments: Dict[str, list[str]] = {}
        self.news_ids: List[str] = []
        self.comment_sep = "[END_OF_COMMENT]\n"
        self.max_comments = max_comments
        
        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        self._load_data(news_path, comment_path)
        
    def _load_data(self, news_path: str | Path, comment_path: str | Path) -> None:
        """加载新闻和评论数据"""
        # 读取新闻数据
        news_df = pd.read_csv(news_path)
        
        # 读取评论数据并按新闻ID分组
        comment_df = pd.read_csv(comment_path)
        comment_groups = comment_df.groupby('sid')['text'].apply(list).to_dict()
        
        for _, row in news_df.iterrows():
            news_id = str(row['news_id'])
            comments = comment_groups.get(news_id, [])
            if len(comments) > self.max_comments:
                comments = comments[:self.max_comments]
                
            news_item = NewsItem(
                news_id=news_id,
                label=int(row['label']),
                title=str(row['title']),
                content=str(row['content']),
                comments=comments
            )
            self.news_data[news_item.news_id] = news_item
            self.news_ids.append(news_item.news_id)

    def __len__(self) -> int:
        return len(self.news_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取指定索引的新闻及其评论

        Args:
            idx: 数据索引

        Returns:
            包含新闻和评论信息的字典
        """
        news_id = self.news_ids[idx]
        news_item = self.news_data[news_id]
        
        # 处理新闻文本（标题 + 内容）
        news_text = f"{news_item.title} {news_item.content}"
        news_encoding = self.tokenizer(
            news_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 初始化空的评论输入
        comments_encoding = self.tokenizer(
            "",  # 空字符串作为初始评论输入
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'news_id': news_item.news_id,
            'news_input_ids': news_encoding['input_ids'].squeeze(0),
            'news_attention_mask': news_encoding['attention_mask'].squeeze(0),
            'comments_input_ids': comments_encoding['input_ids'].squeeze(0),
            'comments_attention_mask': comments_encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(news_item.label),
            'news_content': news_text,  # 保留原始文本用于检索增强
        }

    def get_news(self, news_id: str) -> Optional[NewsItem]:
        """获取指定ID的新闻"""
        return self.news_data.get(news_id)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        news_labels = [item.label for item in self.news_data.values()]
        
        return {
            "total_news": len(self.news_data),
            "news_label_distribution": {
                "fake": sum(1 for l in news_labels if l == 0),
                "real": sum(1 for l in news_labels if l == 1)
            }
        } 