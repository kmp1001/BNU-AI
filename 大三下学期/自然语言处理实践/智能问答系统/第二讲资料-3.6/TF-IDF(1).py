import math
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 中文停用词列表（过滤无意义词）
STOP_WORDS = {"的", "和", "是", "在", "了", "都", "有"}

def preprocess(text):
    """中文预处理：去停用词、分词（示例已用空格分隔）"""
    words = text.split()
    return [word for word in words if word not in STOP_WORDS]


def calculate_tf(text):
    """计算词频（TF）"""
    words = preprocess(text)
    word_count = Counter(words)
    total_words = len(words)
    return {word: count / total_words for word, count in word_count.items()}

def calculate_idf(documents):
    """计算逆文档频率（IDF）"""
    N = len(documents)
    word_doc_count = {}
    for doc in documents:
        words = set(preprocess(doc))
        for word in words:
            word_doc_count[word] = word_doc_count.get(word, 0) + 1
    return {word: math.log(N / (count + 1)) for word, count in word_doc_count.items()}

def calculate_tfidf(documents):
    """计算 TF-IDF"""
    tf_scores = [calculate_tf(doc) for doc in documents]
    idf_scores = calculate_idf(documents)
    tfidf_scores = []
    for tf_doc in tf_scores:
        doc_tfidf = {word: tf * idf_scores[word] for word, tf in tf_doc.items()}
        tfidf_scores.append(doc_tfidf)
    return tfidf_scores

def visualize_tfidf(documents, tfidf_scores):
    """创建TF-IDF热力图可视化"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
    
    # 获取所有唯一词语
    all_words = sorted(list(set(word for doc_scores in tfidf_scores for word in doc_scores.keys())))
    
    # 创建矩阵数据
    matrix_data = []
    for doc_scores in tfidf_scores:
        row = [doc_scores.get(word, 0) for word in all_words]
        matrix_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(matrix_data, 
                     columns=all_words,
                     index=[f"文档 {i+1}" for i in range(len(documents))])
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建热力图
    sns.heatmap(df, 
                annot=True,  # 显示数值
                fmt='.2f',   # 数值格式化为2位小数
                cmap='YlOrRd',  # 使用黄-橙-红色映射
                cbar_kws={'label': 'TF-IDF 分数'})
    
    plt.title('TF-IDF 分数热力图')
    plt.xlabel('词语')
    plt.ylabel('文档')
    
    # 调整布局
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    plt.show()

# 在原有代码最后添加可视化调用
if __name__ == "__main__":
    documents = [
        "小狗 小狗 爱 啃 骨头 骨头",  # 文档1
        "小猫 爱 玩 毛线球",          # 文档2
        "兔子 爱 吃 胡萝卜 和 青菜"      # 文档3
    ]
    
    # 计算TF-IDF
    tfidf_scores = calculate_tfidf(documents)
    
    # 打印结果
    for i, doc_scores in enumerate(tfidf_scores):
        print(f"\n文档 {i+1} 的TF-IDF分数：")
        sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_scores:
            print(f"{word}: {score:.4f}")
    
    # 添加可视化展示
    visualize_tfidf(documents, tfidf_scores)
    