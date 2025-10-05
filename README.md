# Windows部署方法
## 1. 安装minicoda环境
https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

## 2. 配置环境变量
```base
# 添加Path环境变量
C:\ProgramData\miniconda3
C:\ProgramData\miniconda3\Library\bin
C:\ProgramData\miniconda3\Scripts
```

## 3. 安装依赖包
```base
# 创建环境
conda create -n sadtalker python=3.8
conda activate sadtalker

# 安装依赖
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

conda install ffmpeg
# 运行下面命令需要开VPN
pip install -r requirements.txt
```

## 4. 运行
```base
python inference.py --driven_audio ./10s.mp3 --source_image ./1.png --result_dir ./results --still --preprocess full --enhancer gfpgan
```
