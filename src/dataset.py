import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset
import torch

@dataclass
class NewsItem:
    news_id: str
    label: int
    title: str
    content: str

@dataclass
class CommentItem:
    sid: str
    label: int
    tid: str
    uid: str
    text: str
    user_desc: str

class FakeNewsDataset(Dataset):
    def __init__(self, news_path: str | Path, comment_path: str | Path):
        """初始化假新闻数据集

        Args:
            news_path: 新闻数据CSV文件路径
            comment_path: 评论数据CSV文件路径
        """
        self.news_data: Dict[str, NewsItem] = {}
        self.comment_data: Dict[str, CommentItem] = {}
        self.news_to_comments: Dict[str, list[str]] = {}
        self.news_ids: List[str] = []
        self.comment_sep = " [SEP] "  # 评论之间的分隔符
        
        self._load_data(news_path, comment_path)
        
    def _load_data(self, news_path: str | Path, comment_path: str | Path) -> None:
        """加载新闻和评论数据"""
        # 读取新闻数据
        news_df = pd.read_csv(news_path)
        for _, row in news_df.iterrows():
            news_item = NewsItem(
                news_id=str(row['news_id']),
                label=int(row['label']),
                title=str(row['title']),
                content=str(row['content'])
            )
            self.news_data[news_item.news_id] = news_item
            self.news_to_comments[news_item.news_id] = []
            self.news_ids.append(news_item.news_id)

        # 读取评论数据
        comment_df = pd.read_csv(comment_path)
        for _, row in comment_df.iterrows():
            comment_item = CommentItem(
                sid=str(row['sid']),
                label=int(row['label']),
                tid=str(row['tid']),
                uid=str(row['uid']),
                text=str(row['text']),
                user_desc=str(row['user_desc'])
            )
            self.comment_data[comment_item.uid] = comment_item
            
            # 建立新闻和评论的关联
            if comment_item.sid in self.news_to_comments:
                self.news_to_comments[comment_item.sid].append(comment_item.uid)

    def __len__(self) -> int:
        """返回数据集中的新闻数量"""
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
        comments = self.get_news_comments(news_id)
        
        comment_texts = []
        
        for comment in comments:
            comment_texts.append(comment.text)
        
        # 合并评论文本
        comments_text = self.comment_sep.join(comment_texts) if comment_texts else ""
        
        return {
            "news_id": news_item.news_id,
            "news_label": news_item.label,
            "news_title": news_item.title,
            "news_content": news_item.content,
            "comments_text": comments_text
        }
        

    def get_news(self, news_id: str) -> Optional[NewsItem]:
        """获取指定ID的新闻"""
        return self.news_data.get(news_id)

    def get_news_comments(self, news_id: str) -> List[CommentItem]:
        """获取指定新闻的所有评论"""
        comment_uids = self.news_to_comments.get(news_id, [])
        return [self.comment_data[uid] for uid in comment_uids if uid in self.comment_data]

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        news_labels = [item.label for item in self.news_data.values()]
        comment_labels = [item.label for item in self.comment_data.values()]
        
        return {
            "total_news": len(self.news_data),
            "total_comments": len(self.comment_data),
            "news_label_distribution": {
                "fake": sum(1 for l in news_labels if l == 0),
                "real": sum(1 for l in news_labels if l == 1)
            },
            "comment_label_distribution": {
                "fake": sum(1 for l in comment_labels if l == 0),
                "real": sum(1 for l in comment_labels if l == 1)
            },
            "avg_comments_per_news": len(self.comment_data) / len(self.news_data) if self.news_data else 0
        } 