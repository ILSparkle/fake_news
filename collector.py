from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

from src.dataset import FakeNewsDataset
from src.chat import ChatAPI
from src.search import SearchAPI 
from src.crawler import WebCrawler
from src.prompt import keyword_prompt

class NewsCollector:
    def __init__(self, 
                 news_path: str | Path,
                 comment_path: str | Path,
                 output_path: str | Path):
        """初始化新闻收集器

        Args:
            news_path: 新闻数据CSV文件路径
            comment_path: 评论数据CSV文件路径
            output_path: 搜索结果输出JSON文件路径
        """
        self.dataset = FakeNewsDataset(news_path, comment_path)
        self.output_path = Path(output_path)
        self.chat_api = ChatAPI()
        self.search_api = SearchAPI()
        self.crawler = WebCrawler()
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    async def extract_keywords(self, news_content: str) -> List[str]:
        """使用LLM从新闻内容中提取关键词
        
        Args:
            news_content: 新闻内容
            
        Returns:
            关键词列表
        """
        try:
            response = await self.chat_api.chat(
                question=news_content,
                system_prompt=keyword_prompt
            )
            return response.strip().split()
        except Exception as e:
            logging.error(f"提取关键词失败: {str(e)}")
            return []
            
    async def collect_single_news(self, 
                                news_id: str,
                                existing_data: Optional[Dict] = None) -> Dict:
        """收集单条新闻的相关搜索结果
        
        Args:
            news_id: 新闻ID
            existing_data: 已存在的搜索结果数据
            
        Returns:
            该新闻的搜索结果数据
        """
        # 如果已有数据则跳过
        if existing_data and news_id in existing_data:
            return existing_data[news_id]
            
        news_item = self.dataset.get_news(news_id)
        if not news_item:
            return {}
            
        try:
            # 提取关键词
            keywords = await self.extract_keywords(news_item.content)
            if not keywords:
                return {}
                
            # 构建搜索查询
            query = " ".join(keywords)
            
            # 执行搜索
            search_results = await self.search_api.search(query)
            
            # 获取网页内容
            processed_results = await self.crawler.process_search_results(search_results, max_length=100000)
            
            return {
                "news_id": news_id,
                "keywords": keywords,
                "search_results": processed_results
            }
            
        except Exception as e:
            logging.error(f"处理新闻 {news_id} 时出错: {str(e)}")
            return {}
            
    async def collect_all_news(self):
        """收集所有新闻的相关搜索结果"""
        # 加载已存在的数据
        existing_data = {}
        if self.output_path.exists():
            with open(self.output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                
        results = {}
        news_ids = self.dataset.news_ids
        
        # 使用tqdm显示进度
        for news_id in tqdm(news_ids, desc="收集新闻相关数据"):
            result = await self.collect_single_news(news_id, existing_data)
            if result:
                results[news_id] = result
                
            # 定期保存结果
            if len(results) % 10 == 0:
                self._save_results(results)
                
        # 最终保存
        self._save_results(results)
        
    def _save_results(self, results: Dict):
        """保存搜索结果到文件
        
        Args:
            results: 搜索结果数据
        """
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存结果时出错: {str(e)}")

async def main():
    """主函数"""
    collector = NewsCollector(
        news_path="dataset/politifact_news.csv",
        comment_path="dataset/politifact_socialcontext.csv", 
        output_path="dataset/politifact_search_results.json"
    )
    await collector.collect_all_news()

if __name__ == "__main__":
    asyncio.run(main()) 