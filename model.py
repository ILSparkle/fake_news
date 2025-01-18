import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class DualBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.bert_comments = BertModel(config)
        
        # 双线性融合层
        self.bilinear = nn.Bilinear(config.hidden_size, config.hidden_size, config.hidden_size)
        
        # 分类头
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, news_input_ids, news_attention_mask, 
                comments_input_ids, comments_attention_mask, labels=None):
        
        # 处理新闻文本
        news_outputs = self.bert(
            input_ids=news_input_ids,
            attention_mask=news_attention_mask
        )
        news_pooled = news_outputs.pooler_output
        
        # 处理评论文本
        comments_outputs = self.bert_comments(
            input_ids=comments_input_ids,
            attention_mask=comments_attention_mask
        )
        comments_pooled = comments_outputs.pooler_output
        
        # 双线性融合
        fused = self.bilinear(news_pooled, comments_pooled)
        fused = torch.tanh(fused)
        
        # 分类
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return loss, logits 