from dotenv import load_dotenv
import asyncio
import json
from pathlib import Path
import logging
import argparse
from datetime import datetime

load_dotenv()
from torch.utils.data import DataLoader
from src.dataset import FakeNewsDataset
from src.search import SearchAPI
from src.chat import ChatAPI
from src.crawler import WebCrawler
from src.prompt import keyword_prompt, verify_prompt

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
    
    # 生成日志文件名（包含时间戳）
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


async def process_content(news_id: str, content: str, comments: str, label: int, search_api: SearchAPI, chat_api: ChatAPI, crawler: WebCrawler):
    # 获取关键词
    keywords = await chat_api.chat(content, keyword_prompt)
    logging.info(f"提取的关键词: {keywords}")
    
    # 搜索相关新闻
    search_results = await search_api.search(keywords)
    logging.info(f"找到 {len(search_results)} 条相关新闻")
    
    # 获取网页内容
    processed_results = await crawler.process_search_results(search_results)
    
    # 构建相关新闻文本
    related_news_text = "\n\n".join([
        f"来源: {result['url']}\n标题: {result['title']}\n内容: {result['content']}"
        for result in processed_results
    ])
    
    # 验证新闻真实性
    verification_result = await chat_api.chat(
        "",  # 实际内容通过format传入
        verify_prompt.format(
            target_news=content,
            related_news=related_news_text,
            user_comments=comments  # 添加用户评论
        ),
        stream=False
    )
    logging.info(f"真实性判断结果: {verification_result}")
    
    return {
        'news_id': str(news_id),
        'actual_label': int(label),
        'keywords': keywords,
        'search_results': processed_results,
        'comments': comments,
        'verification': verification_result
    }

async def main():
    args = parse_args()
    setup_logging(args.log_level)
    
    logging.info("开始加载数据集")
    # 创建数据集实例
    dataset = FakeNewsDataset(
        news_path="dataset/gossipcop_news.csv",
        comment_path="dataset/gossipcop_socialcontext.csv"
    )
    logging.info(f"数据集大小: {len(dataset)}")
    sample = dataset[0]
    logging.info(f"新闻标题: {sample['news_title']}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    search_api = SearchAPI()
    chat_api = ChatAPI()
    crawler = WebCrawler()
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    for batch_idx, batch in enumerate(dataloader):
        batch_results = []
        for news_id, content, comments, label in zip(batch['news_id'], 
                                                   batch['news_content'],
                                                   batch['comments_text'],
                                                   batch['news_label']):
            try:
                news_id = news_id
                content = content
                comments = comments
                label = label.item()
                
                result = await process_content(news_id, content, comments, label, 
                                            search_api, chat_api, crawler)
                batch_results.append(result)
                logging.info(f"处理完成新闻 {news_id}")
            except Exception as e:
                logging.error(f"处理新闻 {news_id} 时出错: {str(e)}")

        # 保存批次结果
        with open(results_dir / f"batch_{batch_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        break

if __name__ == "__main__":
    asyncio.run(main())
