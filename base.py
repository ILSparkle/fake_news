from dataset import FakeNewsDataset
from model import DualBertForSequenceClassification
from transformers import TrainingArguments, Trainer, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
from tqdm import tqdm
import pandas as pd

def compute_metrics(pred):
    """计算评估指标"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
# 从os.environ中获取数据集路径
import os
politifact_news_path = os.environ.get("POLITICFACT_NEWS_PATH", "dataset/gossipcop_news.csv")
politifact_comment_path = os.environ.get("POLITICFACT_COMMENT_PATH", "dataset/gossipcop_socialcontext.csv")

# 初始化数据集时传入评论文件路径
dataset = FakeNewsDataset(politifact_news_path, politifact_comment_path)

# 划分数据集
train_dataset, test_dataset = train_test_split(dataset, test_size=0.25, random_state=42)

# 设置学习率等参数
learning_rate = 2e-5
num_epochs = 20
batch_size = 32

# 加载配置和初始化双输入模型
config = BertConfig.from_pretrained("bert-base-uncased")
model = DualBertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="output1",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    evaluation_strategy="epoch",  # 每个epoch结束后进行评估
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,  # 加载验证集上表现最好的模型
    metric_for_best_model="f1",   # 使用f1作为最佳模型的选择标准
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,  # 添加评估指标计算函数
)

# 在保存模型之前添加详细的测试代码
def test_model_with_details(model, test_dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_texts = []
    all_probs = []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="测试中"):
            item = test_dataset[i]
            news_input_ids = item['news_input_ids'].unsqueeze(0).to(device)
            news_attention_mask = item['news_attention_mask'].unsqueeze(0).to(device)
            comments_input_ids = item['comments_input_ids'].unsqueeze(0).to(device)
            comments_attention_mask = item['comments_attention_mask'].unsqueeze(0).to(device)
            label = item['labels']
            
            # 获取原始文本
            decoded_news = test_dataset.tokenizer.decode(
                item['news_input_ids'],
                skip_special_tokens=True
            )
            decoded_comments = test_dataset.tokenizer.decode(
                item['comments_input_ids'],
                skip_special_tokens=True
            )
            
            combined_text = f"新闻: {decoded_news[:100]}... || 评论: {decoded_comments[:100]}..."
            all_texts.append(combined_text)
            
            _, logits = model(
                news_input_ids=news_input_ids,
                news_attention_mask=news_attention_mask,
                comments_input_ids=comments_input_ids,
                comments_attention_mask=comments_attention_mask
            )
            
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).item()
            
            all_predictions.append(pred)
            all_labels.append(label)
            all_probs.append(probs[0].cpu().numpy())
    
    # 创建详细结果DataFrame
    results_df = pd.DataFrame({
        '文本': all_texts,
        '预测标签': all_predictions,
        '真实标签': all_labels,
        '预测为假新闻概率': [prob[0] for prob in all_probs],
        '预测为真新闻概率': [prob[1] for prob in all_probs]
    })
    
    # 计算并打印详细指标
    print("\n=== 详细测试结果 ===")
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions, target_names=['假新闻', '真新闻']))
    
    # 打印错误预测的样本
    print("\n=== 预测错误的样本 ===")
    errors = results_df[results_df['预测标签'] != results_df['真实标签']]
    print(f"错误预测数量: {len(errors)} / {len(results_df)} ({len(errors)/len(results_df)*100:.2f}%)")
    print("\n部分错误预测示例:")
    print(errors.head().to_string())
    
    # 保存详细结果到CSV
    results_df.to_csv('test_results_details1.csv', index=False)
    print("\n完整测试结果已保存到 'test_results_details1.csv'")
    
    return results_df

# 训练模型
trainer.train()

# 在测试集上进行常规评估
test_results = trainer.evaluate()

# 打印评估结果
print("\n测试集评估结果:")
print(f"准确率 (Accuracy): {test_results['eval_accuracy']:.4f}")
print(f"F1分数 (F1-Score): {test_results['eval_f1']:.4f}")
print(f"精确率 (Precision): {test_results['eval_precision']:.4f}")
print(f"召回率 (Recall): {test_results['eval_recall']:.4f}")

# 进行详细测试
detailed_results = test_model_with_details(model, test_dataset)

# 保存模型
model.save_pretrained("model/bert-base-uncased1")
