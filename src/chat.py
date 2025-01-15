from openai import AsyncOpenAI
from typing import Optional, Dict, Any
import os

class ChatAPI:
    def __init__(self):
        """初始化聊天API，自动从环境变量读取配置"""
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.api_base: Optional[str] = os.getenv("OPENAI_API_BASE")
        self.model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        
        # 创建异步客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        ) if self.api_key else None
        
        if not self.api_key:
            print("警告: 未找到OPENAI_API_KEY环境变量")
        
    def configure(self, 
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 model_name: Optional[str] = None) -> None:
        """手动配置 OpenAI API 参数

        Args:
            api_key: OpenAI API 密钥，如不提供则使用环境变量
            api_base: API 基础 URL，用于自定义端点
            model_name: 模型名称
        """
        if api_key:
            self.api_key = api_key
            self.client = AsyncOpenAI(api_key=api_key, base_url=self.api_base)
        
        if api_base:
            self.api_base = api_base
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=api_base)
            
        if model_name:
            self.model_name = model_name
    
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

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"调用 OpenAI API 时出错: {str(e)}") 