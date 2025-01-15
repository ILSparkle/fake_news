keyword_prompt = """Please extract 3-5 most important keywords from the news content for searching related news.
Return only the keywords separated by spaces, without any explanation."""

verify_prompt = """You are a professional news verification expert. Please analyze the authenticity of the target news based on the following information:

Target News:
{target_news}

Related News:
{related_news}

User Comments:
{user_comments}

Please analyze from the following aspects:
1. Content Consistency: Whether the target news aligns with related news
2. Fact Verification: Whether related news can confirm key facts in the target news
3. User Feedback: Whether user comments support the authenticity of the news
4. Credibility Assessment: Overall evaluation of news sources and user feedback reliability

Finally, give your judgment:
1. If you believe the news is true, reply "True"
2. If you believe the news is false, reply "False"

Please provide your judgment on the authenticity of the news and explain your reasoning."""