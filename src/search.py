import http.client
import json
from typing import List, Dict, Optional
import urllib.parse
import os

class SearchAPI:
    def __init__(self):
        """初始化搜索API，自动从环境变量读取配置"""
        self.api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        self.base_url: str = "google.serper.dev"
        self.top_k: int = int(os.getenv("GOOGLE_TOP_K", "10"))
        
        # 添加过滤规则
        self.excluded_sites = [
            'youtube.com',
            'vimeo.com',
            'bilibili.com',
            'dailymotion.com',
            'youku.com',
            'academia.edu',  # 学术网站，通常有PDF
            'researchgate.net',
            'scribd.com',
        ]
        
        if not self.api_key:
            print("警告: 未找到SERPER_API_KEY环境变量")
        
    def configure(self,
                 api_key: Optional[str] = None,
                 top_k: Optional[int] = None) -> None:
        """手动配置 Google Search API 参数

        Args:
            api_key: Google Search API 密钥，如不提供则使用环境变量
            top_k: 返回结果数量
        """
        if api_key:
            self.api_key = api_key
        if top_k:
            self.top_k = top_k
            
    async def search(self, query: str) -> List[Dict[str, str]]:
        """执行 Google 搜索

        Args:
            query: 搜索查询

        Returns:
            搜索结果列表，每个结果包含标题、链接和摘要
        """
        if not self.api_key:
            raise ValueError("请先调用 configure() 配置 API 密钥")

        try:
            # 添加过滤条件到查询
            excluded_sites = ' '.join([f'-site:{site}' for site in self.excluded_sites])
            filtered_query = f"{query} {excluded_sites} -filetype:pdf"
            
            # 创建HTTPS连接
            conn = http.client.HTTPSConnection(self.base_url)
            
            # 准备请求数据
            payload = json.dumps({
                "q": filtered_query,
                "num": self.top_k
            })
            
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            # 发送请求
            conn.request("POST", "/search", payload, headers)
            
            # 获取响应
            response = conn.getresponse()
            data = json.loads(response.read().decode("utf-8"))
            
            # 处理搜索结果
            search_results = []
            if "organic" in data:
                for item in data["organic"][:self.top_k]:
                    search_results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
            
            return search_results
            
        except Exception as e:
            raise Exception(f"调用 Serper Google Search API 时出错: {str(e)}")
        
        finally:
            # 关闭连接
            if 'conn' in locals():
                conn.close() 