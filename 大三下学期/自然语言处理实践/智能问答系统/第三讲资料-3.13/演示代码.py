import math
from collections import defaultdict
import jieba

class Preprocessor:
    """文档预处理类，用于分词"""
    def preprocess(self, text):
        """对文本进行预处理，返回处理后的单词列表"""
        # 使用结巴分词进行分词
        words = jieba.lcut(text, cut_all=True)
        return [word.strip() for word in words if word.strip()]  # 过滤空格和空字符


class InvertedIndex:
    """倒排索引类，用于构建和查询倒排索引"""
    def __init__(self):
        self.index = defaultdict(list)  # 倒排索引
        self.doc_id_counter = 1         # 文档ID计数器，从1开始
        self.doc_lengths = defaultdict(int)  # 文档长度，用于TF-IDF计算
        self.doc_count = 0              # 文档总数

    def add_document(self, content):
        """添加文档到倒排索引"""
        doc_id = self.doc_id_counter
        self.doc_id_counter += 1
        terms = Preprocessor().preprocess(content)  # 预处理文档内容
        term_positions = defaultdict(list)         # 记录单词在文档中的位置

        # 遍历单词，记录其在文档中的位置
        for position, term in enumerate(terms):
            term_positions[term].append(position)

        # 更新倒排索引
        for term, positions in term_positions.items():
            self.index[term].append({"doc_id": doc_id, "positions": positions})

        # 计算文档长度
        self.doc_lengths[doc_id] = len(terms)
        self.doc_count += 1

    def build_index(self, documents):
        """构建倒排索引"""
        for doc in documents:
            self.add_document(doc)

    def query(self, query_terms):
        """查询倒排索引，返回包含所有查询词的文档ID列表"""
        results = None
        for term in query_terms:
            if term in self.index:
                current_docs = {entry["doc_id"] for entry in self.index[term]}
                if results is None:
                    results = current_docs
                else:
                    results &= current_docs  # 取交集
            else:
                return []  # 如果有查询词不在索引中，直接返回空列表
        return sorted(results) if results else []

    def rank(self, query):
        """对查询结果进行TF-IDF排序"""
        # 清理查询词，去掉多余的空格
        query = query.strip()
        query_terms = Preprocessor().preprocess(query)
        relevant_docs = self.query(query_terms)
        scores = {}

        # 调试信息
        print(f"查询词 '{query}' 的分词结果: {query_terms}")
        print(f"相关文档: {relevant_docs}")

        # 计算每个相关文档的TF-IDF得分
        for doc_id in relevant_docs:
            score = 0
            doc_length = self.doc_lengths.get(doc_id, 1)
            for term in query_terms:
                if term in self.index:
                    # 查找该词在文档中的出现信息
                    postings = [entry for entry in self.index[term] if entry["doc_id"] == doc_id]
                    if postings:
                        tf = len(postings[0]["positions"])  # 词频
                        idf = math.log(self.doc_count / (1 + len(self.index[term])))  # 逆文档频率
                        score += tf * idf
            scores[doc_id] = score / doc_length  # 归一化得分

        # 按得分从高到低排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores


if __name__ == "__main__":
    # 示例文档
    documents = [
        "乔布斯去了中国",
        "苹果今年仍能占据大多数触摸屏产能",
        "苹果公司首席执行官史蒂夫·乔布斯宣布，iPad2将于3月11日在美国上市",
        "乔布斯推动了世界，iPhone、iPad、iPad2，一款一款接连不断",
        "乔布斯吃了一个苹果"
    ]

    # 构建倒排索引
    index = InvertedIndex()
    index.build_index(documents)

    # 查询示例
    query = "乔布斯 苹果"
    results = index.rank(query)
    print("查询结果（按相关性排序）：")
    for doc_id, score in results:
        print(f"文档ID: {doc_id}, 相关性得分: {score:.4f}")
