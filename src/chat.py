from openai import AsyncOpenAI
from typing import Optional, Dict, Any
import os
import asyncio
import logging

class ChatAPI:
    def __init__(self):
        """初始化聊天API，自动从环境变量读取配置"""
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.api_base: Optional[str] = os.getenv("OPENAI_API_BASE")
        self.model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        self.max_retries: int = 3
        self.retry_delay: float = 2.0  # 重试间隔（秒）
        
        # 创建异步客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        ) if self.api_key else None
        
        if not self.api_key:
            print("警告: 未找到OPENAI_API_KEY环境变量")
    
    async def _retry_chat(self, messages: list, **kwargs) -> str:
        """带重试机制的聊天请求"""
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt == self.max_retries - 1:  # 最后一次尝试
                    raise Exception(f"调用 OpenAI API 失败，已重试{self.max_retries}次: {str(e)}")
                    
                logging.warning(f"调用 OpenAI API 出错 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                await asyncio.sleep(self.retry_delay * (attempt + 1))  # 指数退避
    
    async def chat(self, 
                  question: str,
                  system_prompt: Optional[str] = None,
                  **kwargs: Dict[str, Any]) -> str:
        """发送聊天请求到 OpenAI API

        Args:
            question: 用户问题
            system_prompt: 系统提示词
            **kwargs: 其他传递给 API 的参数

        Returns:
            模型的回答文本
        """
        if not self.client:
            raise ValueError("请先调用 configure() 配置 API 密钥")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        return await self._retry_chat(messages, **kwargs) 