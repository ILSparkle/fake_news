initial_score_prompt = """You are a professional news authenticity assessment expert. Please analyze the authenticity of the following news content and user comments.

News content:
{news_content}

User comments:
{user_comments}

Please analyze from the following aspects and provide your analysis in a structured format:

1. Content Analysis:
- Credibility and reasonableness of content
- Professionalism of language expression
- Specificity of information
- Emotional tendencies and bias levels

2. User Comments Analysis:
- Overall sentiment in comments
- Consistency between comments
- Whether comments provide additional verification or refutation
- Credibility of comment authors

As long as the news is partially false, it is a fake news.

Based on the above analysis, rate the news authenticity from 0-5:
0: Definitely fake news
1: Most likely fake news (with probability over 80%)
2: Not sure, probably fake news, need more verification
3: Not sure, probably real news, need more verification
4: Most likely real news (with probability over 80%)
5: Very sure this is real news, you should not give this score easily

Your response should be in the following format:
Analysis: [analysis_content]

Final Score: [score]

Do not use ### or other markdown syntax."""

keyword_prompt = """Please extract 3-5 most important keywords from the news content for searching related news.
Return only the keywords separated by spaces, without any explanation."""

verify_prompt = """You are a professional news verification expert. Please analyze the authenticity of the target news based on the following information:

Target News:
{target_news}

Initial Analysis Results:
{initial_result}
(Score meaning:
0: Definitely fake news
1: Most likely fake news
2: Probably fake news
3: Probably real news
4: Most likely real news
5: Very sure this is real news)

Related News Found:
{related_news}

Please analyze from the following aspects:
1. Content Consistency: Whether the target news aligns with related news
2. Fact Verification: Whether related news can confirm key facts in the target news
3. Initial Analysis Review: Whether the initial analysis and score were reasonable
4. Credibility Assessment: Overall evaluation considering both initial analysis and related news

Finally, give your judgment:
1. If you believe the news is true, reply "True"
2. If you believe the news is false, reply "False"

Please provide your judgment on the authenticity of the news and explain your reasoning, especially if you disagree with the initial analysis."""