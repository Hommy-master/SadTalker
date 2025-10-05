import os
import torch 

from gfpgan import GFPGANer

from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2

import cv2


class GeneratorWithLen(object):
    """ 带有长度信息的生成器包装类
    来源: https://stackoverflow.com/a/7460929 
    用于为生成器提供__len__方法，使其可以被传递给需要调用len()的函数
    """

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

def enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan'):
    """
    增强图像列表并返回列表结果
    
    Args:
        images: 图像列表或视频文件路径
        method: 面部增强方法 ('gfpgan', 'RestoreFormer', 'codeformer')
        bg_upsampler: 背景上采样方法 ('realesrgan')
        
    Returns:
        增强后的图像列表
    """
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    return list(gen)

def enhancer_generator_with_len(images, method='gfpgan', bg_upsampler='realesrgan'):
    """ 
    提供一个带有__len__方法的生成器，使其可以被传递给调用len()的函数
    
    Args:
        images: 图像列表或视频文件路径
        method: 面部增强方法
        bg_upsampler: 背景上采样方法
        
    Returns:
        带有长度信息的增强器生成器
    """

    if os.path.isfile(images): # 处理视频文件转图像序列
        # TODO: 为load_video_to_cv2创建一个生成器版本
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    gen_with_len = GeneratorWithLen(gen, len(images))
    return gen_with_len

def enhancer_generator_no_len(images, method='gfpgan', bg_upsampler='realesrgan'):
    """ 
    提供一个生成器函数，使得所有增强后的图像不需要同时存储在内存中。
    这可以相比于enhancer函数节省大量的RAM内存。
    
    Args:
        images: 图像列表或视频文件路径
        method: 面部增强方法 ('gfpgan', 'RestoreFormer', 'codeformer')
        bg_upsampler: 背景上采样方法 ('realesrgan')
        
    Yields:
        增强后的图像
    """

    print('face enhancer....')
    if not isinstance(images, list) and os.path.isfile(images): # 处理视频文件转图像序列
        images = load_video_to_cv2(images)

    # ------------------------ 设置 GFPGAN 恢复器 ------------------------
    if  method == 'gfpgan':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif method == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    elif method == 'codeformer': # TODO: 待实现
        arch = 'CodeFormer'
        channel_multiplier = 2
        model_name = 'CodeFormer'
        url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    else:
        raise ValueError(f'Wrong model version {method}.')


    # ------------------------ 设置背景上采样器 ------------------------
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU环境
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            # 创建RRDBNet网络模型
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            # 创建RealESRGAN上采样器
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,        # 分块处理尺寸
                tile_pad=10,     # 分块边界填充
                pre_pad=0,       # 预填充
                half=True)       # 在CPU模式下需要设置为False
    else:
        bg_upsampler = None

    # 确定模型路径
    model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    
    if not os.path.isfile(model_path):
        model_path = os.path.join('checkpoints', model_name + '.pth')
    
    if not os.path.isfile(model_path):
        # 从网络下载预训练模型
        model_path = url

    # 创建GFPGAN恢复器
    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,                    # 上采样倍数
        arch=arch,                    # 网络架构
        channel_multiplier=channel_multiplier,  # 通道倍数
        bg_upsampler=bg_upsampler)    # 背景上采样器

    # ------------------------ 执行面部恢复 ------------------------
    for idx in tqdm(range(len(images)), 'Face Enhancer:'):
        
        # 将RGB格式转换为BGR格式（OpenCV使用BGR格式）
        img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
        
        # 恢复面部和背景（如果需要）
        cropped_faces, restored_faces, r_img = restorer.enhance(
            img,
            has_aligned=False,      # 输入图像是否已经对齐
            only_center_face=False, # 是否只处理中心面部
            paste_back=True)        # 是否将恢复后的面部粘贴回原图
        
        # 将BGR格式转换回RGB格式
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        yield r_img
