import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from train.feature_extractor import AdvancedFeatureExtractor
from config import RAW_DATA_DIR

testpath = RAW_DATA_DIR / "语文"

extractor = AdvancedFeatureExtractor()
features = extractor.extract_features([str(testpath / i) for i in os.listdir(testpath)])
print(f"提取的特征维度: {features.shape}")
print(f"前5个特征样本:\n{features[:5]}")
