# SadTalker - 数字人语音驱动视频生成系统

## 📖 项目简介

SadTalker是一个先进的数字人语音驱动视频生成系统，能够从单张图片和音频文件生成逼真的说话视频。该系统基于3DMM（3D Morphable Model）技术，通过音频信号驱动面部表情和头部动作，创造出自然流畅的数字人说话效果。

## ✨ 主要特性

- 🎯 **单图生成**: 仅需一张人物照片即可生成说话视频
- 🎵 **音频驱动**: 支持多种音频格式，自动同步口型和表情
- 🎨 **高质量输出**: 支持多种图像增强技术，提升视频清晰度
- 🔧 **灵活配置**: 丰富的参数选项，满足不同场景需求
- 🚀 **易于使用**: 简单的命令行接口，快速上手

## 🛠️ Windows环境部署方法

### 1. 安装Miniconda环境
下载并安装Miniconda：
https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

### 2. 配置环境变量
将以下路径添加到系统Path环境变量中：
```bash
# 添加Path环境变量
C:\ProgramData\miniconda3
C:\ProgramData\miniconda3\Library\bin
C:\ProgramData\miniconda3\Scripts
```

### 3. 安装依赖包
```bash
# 创建Python环境
conda create -n sadtalker python=3.8
conda activate sadtalker

# 安装PyTorch（GPU版本）
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# 安装FFmpeg
conda install ffmpeg

# 安装项目依赖（可能需要开启VPN）
pip install -r requirements.txt

# 安装图像增强组件（可选）
pip install gfpgan
pip install realesrgan
```

### 4. 下载模型文件
```bash
# 自动下载所需的预训练模型
bash scripts/download_models.sh
```

## 🚀 快速开始

### 基础使用示例
```bash
# 基础生成命令
python inference.py \
    --driven_audio examples/driven_audio/chinese_news.wav \
    --source_image examples/source_image/art_6.png \
    --result_dir ./results
```

### 高质量生成示例
```bash
# 带图像增强的高质量生成
python inference.py \
    --driven_audio examples/driven_audio/chinese_news.wav \
    --source_image examples/source_image/full_body_1.png \
    --result_dir ./results \
    --still \
    --preprocess full \
    --enhancer RestoreFormer \
    --background_enhancer realesrgan \
    --expression_scale 1.3 \
    --size 512
```

## 📋 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--driven_audio` | str | - | 驱动音频文件路径 |
| `--source_image` | str | - | 源图像文件路径 |
| `--result_dir` | str | `./results` | 结果输出目录 |
| `--enhancer` | str | None | 面部增强器 [gfpgan, RestoreFormer] |
| `--background_enhancer` | str | None | 背景增强器 [realesrgan] |
| `--expression_scale` | float | 1.0 | 表情强度缩放因子 |
| `--size` | int | 256 | 输出图像尺寸 |
| `--still` | bool | False | 静止模式，减少头部运动 |
| `--preprocess` | str | crop | 预处理方式 [crop, resize, full] |

## 📂 项目结构

```
SadTalker/
├── src/                          # 源代码目录
│   ├── audio2exp_models/         # 音频到表情模型
│   ├── audio2pose_models/        # 音频到姿态模型
│   ├── facerender/               # 面部渲染模块
│   ├── face3d/                   # 3D面部重建模块
│   ├── utils/                    # 工具函数
│   └── config/                   # 配置文件
├── examples/                     # 示例文件
│   ├── driven_audio/             # 示例音频
│   └── source_image/             # 示例图像
├── checkpoints/                  # 模型检查点
├── docs/                         # 文档目录
├── scripts/                      # 脚本文件
├── inference.py                  # 主推理脚本
└── README.md                     # 项目说明
```

## 🎨 高级用法

### 1. 自定义头部姿态
```bash
python inference.py \
    --driven_audio audio.wav \
    --source_image image.png \
    --input_yaw -20 30 10 \
    --input_pitch -10 10 -5 \
    --input_roll -5 5 0
```

### 2. 使用参考视频
```bash
python inference.py \
    --driven_audio audio.wav \
    --source_image image.png \
    --ref_eyeblink reference_video.mp4 \
    --ref_pose reference_pose.mp4
```

### 3. 3D可视化模式
```bash
python inference.py \
    --driven_audio audio.wav \
    --source_image image.png \
    --face3dvis
```

## 🔍 故障排除

### 常见问题
1. **CUDA相关错误**: 请确保安装了正确版本的PyTorch和CUDA
2. **内存不足**: 尝试减小batch_size或使用CPU模式
3. **模型下载失败**: 检查网络连接或手动下载模型文件
4. **图像质量不佳**: 尝试使用不同的增强器组合

### 性能优化建议
- 使用GPU加速（推荐GTX 1080或更高配置）
- 合理设置batch_size以平衡速度和内存使用
- 选择合适的图像尺寸（256/512）

## 📄 许可证

本项目遵循相应的开源许可证。请查看LICENSE文件了解详情。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📧 联系方式

如有问题或建议，请通过GitHub Issues联系我们。