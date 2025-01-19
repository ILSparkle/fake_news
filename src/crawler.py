import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import asyncio
from urllib.parse import urlparse
import logging
import os
from pathlib import Path

class WebCrawler:
    def __init__(self):
        """初始化网页爬虫"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
        }
        self.timeout = 10  # 请求超时时间（秒）
        
        # 从环境变量读取代理配置
        proxy = os.getenv("HTTP_PROXY")
        self.proxies = {
            "http": proxy,
            "https": proxy
        } if proxy else None
        
        if self.proxies:
            logging.info(f"已加载代理: {proxy}")
            
        # 添加不支持的文件类型
        self.unsupported_extensions = {
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', 
            '.xls', '.xlsx', '.zip', '.rar', '.mp4', 
            '.avi', '.mov', '.wmv'
        }
        
    def _fetch_sync(self, url: str) -> Optional[str]:
        """同步获取网页内容"""
        try:
            response = requests.get(
                url,
                headers=self.headers,
                proxies=self.proxies,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.text
            else:
                logging.warning(f"获取页面失败: {url}, 状态码: {response.status_code}")
                return None
                
        except Exception as e:
            logging.error(f"抓取页面出错: {url}, 错误: {str(e)}")
            return None
        
    async def fetch_content(self, url: str) -> Optional[str]:
        """异步获取网页内容

        Args:
            url: 网页URL

        Returns:
            网页文本内容，如果获取失败则返回None
        """
        return await asyncio.to_thread(self._fetch_sync, url)

    def extract_text(self, html: str) -> str:
        """从HTML中提取主要文本内容

        Args:
            html: 网页HTML内容

        Returns:
            提取的文本内容
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # 移除script和style元素
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 获取文本
            text = soup.get_text()
            
            # 处理文本
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logging.error(f"解析HTML出错: {str(e)}")
            return ""

    def _is_supported_url(self, url: str) -> bool:
        """检查URL是否支持爬取
        
        Args:
            url: 要检查的URL
            
        Returns:
            是否支持爬取
        """
        try:
            # 检查文件扩展名
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            extension = Path(path).suffix
            
            if extension in self.unsupported_extensions:
                logging.info(f"跳过不支持的文件类型: {url}")
                return False
                
            return True
            
        except Exception:
            return False
            
    def _is_valid_content(self, content: str) -> bool:
        """检查内容是否有效
        
        Args:
            content: 网页内容
            
        Returns:
            内容是否有效
        """
        if not content:
            return False
            
        # 检查是否包含过多无法解码的字符（可能是PDF等二进制文件）
        invalid_char_ratio = sum(1 for c in content if not c.isprintable()) / len(content)
        if invalid_char_ratio > 0.1:  # 如果超过10%的字符无法打印
            return False
            
        return True
        
    async def process_search_results(self, 
                                   search_results: List[Dict[str, str]], 
                                   max_length: int = 1000) -> List[Dict[str, str]]:
        """处理搜索结果，获取每个网页的内容"""
        processed_results = []
        
        for result in search_results:
            url = result['link']
            
            # 检查URL是否支持爬取
            if not self._is_supported_url(url):
                continue
                
            content = await self.fetch_content(url)
            
            # 检查内容是否有效
            if content and self._is_valid_content(content):
                text = self.extract_text(content)
                # 截取指定长度
                text = text[:max_length] + ('...' if len(text) > max_length else '')
                
                processed_results.append({
                    'title': result['title'],
                    'url': url,
                    'snippet': result['snippet'],
                    'content': text
                })
            
        return processed_results
    
    def _is_valid_url(self, url: str) -> bool:
        """检查URL是否合法

        Args:
            url: 要检查的URL

        Returns:
            URL是否合法
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False 