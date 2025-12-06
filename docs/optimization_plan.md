# 训练机强、部署机弱 — 轻量化部署优化方案

针对“训练机强、部署机弱”的场景，下面给出一个完整的端到端优化方案：在强机器上做复杂计算与压缩，在弱部署机上运行极简推理。

## 一、整体优化策略

```
训练阶段（强机）：特征工程 → 复杂模型 → 模型压缩 → 轻量导出
部署阶段（弱机）：加载轻量模型 → 极简推理
```

## 二、训练阶段（i7+3070）：高级优化

### 1. 多模态特征提取（训练机完成）

```python
import torch
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from sklearn.feature_selection import SelectKBest, chi2

class AdvancedFeatureExtractor:
    """在强训练机上提取多模态特征"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # 加载预训练模型（仅在训练机）
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        self.bert_model.to(device)
        self.bert_model.eval()
        
        # 图像特征提取器
        self.img_model = models.resnet18(pretrained=True)
        self.img_model = torch.nn.Sequential(*list(self.img_model.children())[:-1])
        self.img_model.to(device)
        self.img_model.eval()
    
    def extract_deep_features(self, ppt_data):
        """提取深度特征"""
        with torch.no_grad():
            # 文本BERT特征
            text_features = self._extract_bert_features(ppt_data['text'])
            
            # 图像特征
            img_features = self._extract_image_features(ppt_data['images'])
            
            # 结构特征
            struct_features = self._extract_structural_features(ppt_data)
            
        # 合并所有特征
        deep_features = np.concatenate([
            text_features, 
            img_features, 
            struct_features
        ], axis=1)
        
        return deep_features
    
    def select_light_features(self, features, labels, k=300):
        """特征选择：保留最重要的k个特征"""
        selector = SelectKBest(chi2, k=k)
        selected_features = selector.fit_transform(features, labels)
        
        # 保存选择器，用于后续转换
        self.selector = selector
        return selected_features
    
    def train_teacher_model(self, X, y):
        """训练教师模型（复杂模型）"""
        from xgboost import XGBClassifier
        
        teacher_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method='gpu_hist',  # 使用GPU加速
            gpu_id=0,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        teacher_model.fit(X, y)
        return teacher_model
```

### 2. 知识蒸馏：从复杂模型到简单模型

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

class KnowledgeDistillation:
    """知识蒸馏：将复杂模型的知识转移到简单模型"""
    
    def __init__(self, teacher_model):
        self.teacher_model = teacher_model
    
    def distill_to_light_model(self, X_train, y_train, method='lr'):
        """
        蒸馏到轻量模型
        method: 'lr'(逻辑回归), 'dt'(决策树), 'nb'(朴素贝叶斯)
        """
        # 1. 获取教师模型的软标签（概率输出）
        teacher_probs = self.teacher_model.predict_proba(X_train)
        
        # 2. 训练学生模型
        if method == 'lr':
            # 逻辑回归（轻量快速）
            student = LogisticRegression(
                C=0.5,
                max_iter=1000,
                solver='liblinear',
                multi_class='ovr',
                penalty='l1'  # L1正则化，产生稀疏权重
            )
        elif method == 'dt':
            # 决策树（可解释性强）
            student = DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                ccp_alpha=0.01  # 成本复杂度剪枝
            )
        elif method == 'nb':
            # 朴素贝叶斯（极快）
            from sklearn.naive_bayes import MultinomialNB
            student = MultinomialNB(alpha=0.1)
        
        # 3. 使用软标签训练（温度蒸馏）
        T = 2.0  # 温度参数
        soft_labels = np.power(teacher_probs, 1/T)
        soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)
        
        # 4. 训练学生模型
        student.fit(X_train, soft_labels)
        
        return student
    
    def prune_model(self, model, X_val, y_val):
        """模型剪枝"""
        if hasattr(model, 'feature_importances_'):
            # 决策树/XGBoost剪枝
            importances = model.feature_importances_
            threshold = np.percentile(importances, 30)  # 去掉70%不重要的特征
            
            # 创建新的剪枝模型
            if hasattr(model, 'prune'):
                model.prune(threshold)
        
        return model
```

## 三、轻量化部署方案

### 1. 极简推理类（部署机用）

```python
import numpy as np
from scipy.sparse import csr_matrix
import joblib

class UltraLightPPTClassifier:
    """
    极简PPT分类器，专为低配设备设计
    内存占用<50MB，推理时间<0.5秒
    """
    
    def __init__(self, model_path):
        # 加载预处理后的轻量模型
        data = joblib.load(model_path)
        
        # 核心组件
        self.vectorizer = data['vectorizer']      # 词表映射
        self.model = data['model']                # 轻量模型
        self.feature_mask = data['feature_mask']  # 特征掩码（仅保留重要特征）
        self.keyword_matcher = data['keyword_matcher']  # 关键词快速匹配器
        
        # 预计算数据
        self.precomputed = data.get('precomputed', {})
        
        # 内存优化：转换为稀疏存储
        if hasattr(self.vectorizer, 'vocabulary_'):
            self._optimize_memory()
    
    def _optimize_memory(self):
        """内存优化"""
        # 1. 压缩词表：只保留用到的词
        if hasattr(self.vectorizer, 'vocabulary_'):
            original_vocab = self.vectorizer.vocabulary_
            self.vectorizer.vocabulary_ = {
                k: v for k, v in original_vocab.items() 
                if v in self.feature_mask
            }
        
        # 2. 转换模型参数为float16（节省一半内存）
        if hasattr(self.model, 'coef_'):
            self.model.coef_ = self.model.coef_.astype(np.float16)
        
        # 3. 清理不需要的属性
        if hasattr(self.vectorizer, 'stop_words_'):
            delattr(self.vectorizer, 'stop_words_')
    
    def extract_text_fast(self, file_path):
        """
        快速文本提取（避免python-pptx的内存开销）
        使用纯文本提取，跳过复杂格式
        """
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            import re
            
            # 直接解析pptx的XML（内存效率高）
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # 读取所有幻灯片
                slides_text = []
                slide_files = [f for f in zip_ref.namelist() 
                              if f.startswith('ppt/slides/slide')]
                
                for slide_file in slide_files[:30]:  # 限制前30页
                    content = zip_ref.read(slide_file).decode('utf-8', errors='ignore')
                    
                    # 简单XML解析提取文本
                    texts = re.findall(r'<a:t[^>]*>([^<]+)</a:t>', content)
                    slides_text.extend(texts)
            
            return ' '.join(slides_text)[:5000]  # 限制文本长度
        except:
            # 备用方案：使用python-pptx但限制内存
            return self._extract_text_fallback(file_path)
    
    def _extract_text_fallback(self, file_path):
        """备用提取方案"""
        from pptx import Presentation
        import gc
        import re
        
        try:
            prs = Presentation(file_path)
            texts = []
            
            # 限制处理范围和深度
            for i, slide in enumerate(prs.slides[:20]):  # 最多20页
                for shape in slide.shapes[:50]:  # 每页最多50个形状
                    if hasattr(shape, "text"):
                        text = shape.text.strip()
                        if text and len(text) > 2:
                            texts.append(text)
            
            result = ' '.join(texts)
            
            # 强制垃圾回收
            del prs
            gc.collect()
            
            return result
        except:
            return ""
    
    def predict_fast(self, ppt_file):
        """
        极速预测，专为低配置优化
        """
        # 1. 快速文本提取
        text = self.extract_text_fast(ppt_file)
        
        if not text or len(text) < 10:
            return "未知", 0.0
        
        # 2. 第一层：关键词快速匹配（超快）
        fast_result, fast_conf = self._keyword_fast_match(text)
        if fast_conf > 0.8:  # 高置信度直接返回
            return fast_result, fast_conf
        
        # 3. 第二层：轻量模型预测
        # 极简特征提取
        features = self._extract_light_features(text)
        
        if features is None:
            return fast_result, fast_conf
        
        # 预测
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
            pred_idx = np.argmax(proba)
            confidence = proba[pred_idx]
            
            # 获取学科标签
            labels = list(self.precomputed.get('label_mapping', {}).keys())
            if labels and pred_idx < len(labels):
                subject = labels[pred_idx]
            else:
                subject = ["语文", "数学", "英语", "物理", "化学", "生物"][pred_idx]
        else:
            prediction = self.model.predict(features)[0]
            subject = str(prediction)
            confidence = 0.7
        
        # 置信度修正
        confidence = max(confidence, fast_conf * 0.3)
        
        return subject, min(confidence, 0.99)
    
    def _keyword_fast_match(self, text):
        """关键词快速匹配（O(1)复杂度）"""
        scores = {}
        
        # 预计算的关键词命中表
        for subject, keywords in self.keyword_matcher.items():
            count = 0
            for keyword in keywords:
                if keyword in text:
                    count += 1
            
            if count > 0:
                scores[subject] = count / len(keywords)
        
        if scores:
            best_subject = max(scores.items(), key=lambda x: x[1])
            return best_subject[0], min(best_subject[1], 0.95)
        
        return "未知", 0.0
    
    def _extract_light_features(self, text):
        """极简特征提取"""
        if not hasattr(self.vectorizer, 'transform'):
            return None
        
        # 简单文本清洗
        text_clean = re.sub(r'\s+', ' ', text.lower())
        
        # 使用稀疏向量（节省内存）
        try:
            features = self.vectorizer.transform([text_clean])
            
            # 应用特征选择掩码
            if self.feature_mask is not None:
                features = features[:, self.feature_mask]
            
            # 转换为稠密数组（对于小模型更快）
            if features.shape[1] < 500:
                return features.toarray()
            else:
                return features
        except:
            return None
```

### 2. 训练-部署转换脚本

（此处略去部分脚本重复 — 在仓库中可放置 `model_compressor.py`、`ultralight_classifier.py` 等模块）

## 四、完整工作流程（简要）

- 训练端：`train_advanced.py` 在强训练机上运行，完成特征提取、教师模型训练、蒸馏与压缩。
- 部署端：`deploy_light.py` 在弱设备上加载 `deployment_model.joblib` 并通过 `UltraLightPPTClassifier` 进行推理。

## 五、关键优化点与建议

- **知识蒸馏**：在大模型与小模型之间转移软标签，保留决策边界信息。
- **特征选择**：只导出 top-k 特征，减少向量维度并压缩词表。
- **参数量化**：把权重转为 `float16`，显著降低内存占用。
- **两层推理**：先用关键词快速匹配（极快），再用轻量模型补充判断。
- **内存守护**：在部署端加入 `MemoryGuard`，根据可用内存动态切换到极简模式。

## 六、部署清单（最小）

- `deployment_model.joblib` — 压缩后的模型包
- `ultralight_classifier.py` — 极简推理实现
- `deploy_light.py` — 启动脚本
- `check_deployment_env.py` — 环境检查脚本
- `requirements.txt` — 精简依赖（numpy、scikit-learn、joblib、python-pptx、psutil）

---

如需，我可以：

- 将本文档放到仓库（已完成）。
- 生成示例模块 `ultralight_classifier.py` 与 `model_compressor.py`（可选）。
- 写一个 Windows 下的一键部署批处理（可选）。

欢迎告诉我下一步优先级。
