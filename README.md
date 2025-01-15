# 虚假新闻检测

## 环境配置
```bash
conda create -n fakenews python=3.10
conda activate fakenews
pip install -r requirements.txt
```
需要配置.env文件，包含openai和google的api key，参考.env.example

## 使用方法
```bash
python run.py
```

## 数据集
gossipcop  
politifact

## 模型
目前均使用gpt-4o

## 流程
1. 加载数据集新闻和评论
2. gpt-4o提取关键词
3. google search搜索相关新闻（网站截取前1000token，后续可考虑进行分块+向量相似度优化）
4. gpt-4o推理真实性
5. 保存结果

## 结果
测试可见results/batch_i.json  
具体日志见logs目录
