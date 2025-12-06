```
ppt_sorter/
├── README.md                           # 项目说明文档
├── requirements_train.txt              # 训练环境依赖
├── requirements_deploy.txt             # 部署环境依赖
├── config.py                           # 配置文件
├── data/                               # 数据目录
│   ├── raw/                           # 原始PPT文件
│   │   ├── 语文/
│   │   ├── 数学/
│   │   ├── 英语/
│   │   ├── 物理/
│   │   ├── 化学/
│   │   └── 生物/
│   └── processed/                     # 处理后的数据
│       └── subject_keywords.json      # 关键词词典
├── train/                              # 训练相关代码
│   ├── train_main.py                   # 主训练脚本
│   ├── feature_extractor.py           # 特征提取器
│   ├── model_builder.py               # 模型构建器
│   ├── knowledge_distiller.py         # 知识蒸馏
│   └── model_compressor.py            # 模型压缩
├── deploy/                             # 部署相关代码
│   ├── ultra_light_classifier.py      # 极简分类器
│   ├── deploy_main.py                 # 部署主脚本
│   ├── memory_guard.py                # 内存守护
│   └── check_environment.py           # 环境检查
├── models/                             # 模型文件
│   ├── teacher_model.joblib           # 教师模型
│   ├── student_model.joblib           # 学生模型
│   └── deployment_model.joblib        # 部署模型
├── utils/                              # 工具函数
│   ├── __init__.py
│   ├── ppt_parser.py                  # PPT解析器
│   ├── text_processor.py              # 文本处理器
│   ├── keyword_manager.py             # 关键词管理器
│   └── evaluation.py                  # 评估工具
├── scripts/                            # 脚本文件
│   ├── build_deployment_package.sh    # 构建部署包脚本
│   ├── batch_classify.py              # 批量分类脚本
│   └── benchmark.py                   # 性能测试脚本
└── tests/                              # 测试文件
    ├── test_classifier.py
    └── test_performance.py

```