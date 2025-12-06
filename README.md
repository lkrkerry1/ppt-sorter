

# PPT Sorter

根据ppt内容进行分类

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GNU License][license-shield]][license-url]

 
## 目录

- [PPT Sorter](#ppt-sorter)
  - [目录](#目录)
  - [特点](#特点)
  - [快速开始](#快速开始)
    - [训练阶段（需要GPU/高性能CPU）](#训练阶段需要gpu高性能cpu)
      - [准备数据：将PPT按学科放入data/raw/对应目录](#准备数据将ppt按学科放入dataraw对应目录)
      - [训练模型](#训练模型)
      - [部署阶段](#部署阶段)
  - [文件目录说明](#文件目录说明)
  - [贡献者](#贡献者)
      - [如何参与开源项目](#如何参与开源项目)
  - [版本控制](#版本控制)
  - [鸣谢](#鸣谢)


## 特点
- 🚀 **双阶段优化**：强训练机训练，弱部署机运行
- 📦 **模型极小**：部署模型<10MB，内存占用<50MB
- ⚡ **推理快速**：单个PPT分类<1秒
- 🎯 **准确率高**：充足样本下>85%准确率
- 🔧 **易部署**：无需GPU，Python基础环境即可运行


## 快速开始

### 训练阶段（需要GPU/高性能CPU）
```bash
# 安装依赖
pip install -r requirements_train.txt
```

#### 准备数据：将PPT按学科放入data/raw/对应目录

#### 训练模型
```bash
python train/train_main.py
```

#### 部署阶段
```bash
# 安装轻量依赖
pip install -r requirements_deploy.txt

# 检查环境
python deploy/check_environment.py

# 运行分类器
python deploy/deploy_main.py path/to/your.pptx

# 批量处理
python scripts/batch_classify.py --input folder/with/ppts --output results.csv
```

## 文件目录说明
见 `architecture.md`

## 贡献者

#### 如何参与开源项目

贡献使开源社区成为一个学习、激励和创造的绝佳场所。你所作的任何贡献都是**非常感谢**的。


1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。


## 鸣谢

- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)

<!-- links -->
[your-project-path]:lkrkerry1/ppt-sorter#
[contributors-shield]: https://img.shields.io/github/contributors/lkrkerry1/ppt-sorter.svg?style=flat-square
[contributors-url]: https://github.com/lkrkerry1/ppt-sorter/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/lkrkerry1/ppt-sorter.svg?style=flat-square
[forks-url]: https://github.com//lkrkerry1/ppt-sorter/network/members
[stars-shield]: https://img.shields.io/github/stars/lkrkerry1/ppt-sorter.svg?style=flat-square
[stars-url]: https://github.com/lkrkerry1/ppt-sorter/stargazers
[issues-shield]: https://img.shields.io/github/issues/lkrkerry1/ppt-sorter.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/lkrkerry1/ppt-sorter.svg
[license-shield]: https://img.shields.io/github/license/lkrkerry1/ppt-sorter.svg?style=flat-square
[license-url]: https://github.com/lkrkerry1/ppt-sorter/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555



