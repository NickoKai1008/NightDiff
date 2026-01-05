import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap

def natural_sort_key(s):
    """
    自然排序键函数，用于正确处理文件名中的数字
    例如: ['1.png', '2.png', '10.png'] 而不是 ['1.png', '10.png', '2.png']
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def calculate_image_metrics(image_path, brightness_threshold=200, output_dir=None, target_size=(512, 512)):
    """
    计算图像的三个指标：
    1. 像素级亮度（平均灰度值）
    2. 亮斑总面积（像素数）
    3. 亮斑平均重心距
    
    参数:
    image_path: 输入图像路径
    brightness_threshold: 亮度阈值（0-255）
    output_dir: 输出目录，用于保存灰度图和亮斑图
    target_size: 目标图像尺寸，默认为(512, 512)
    
    返回:
    metrics: 包含三个指标的字典
    """
    # 读取图像 - 使用numpy方法支持中文路径
    try:
        # 使用numpy从文件读取字节，然后用OpenCV解码
        img_array = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法解码图像")
    except Exception as e:
        raise ValueError(f"无法读取图像: {image_path}, 错误: {str(e)}")
    
    # 调整图像大小为512x512
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 生成基于照明工程风格的伪彩色热图（深蓝→蓝绿→绿→黄绿→黄→橙→亮红）
    heatmap_colors = [
        '#001a6e',  # deep blue (low)
        '#0aa5a7',  # blue-green
        '#1e9d4b',  # green
        '#b7e000',  # yellow-green
        '#ffd500',  # yellow
        '#ff8c1a',  # orange
        '#ff2a2a',  # bright red (high)
    ]
    cmap_dialux = LinearSegmentedColormap.from_list('dialux_like', heatmap_colors)
    gray_norm = (gray.astype(np.float32) / 255.0)
    rgb_float = cmap_dialux(gray_norm)[..., :3]  # (H,W,3), in RGB, 0-1
    rgb_uint8 = (rgb_float * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    
    # 1. 计算平均像素亮度
    mean_brightness = np.mean(gray)
    
    # 创建亮斑掩码（二值化）
    _, binary = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    # 2. 计算亮斑总面积
    lighted_area = np.sum(binary == 255)
    
    # 3. 计算亮斑平均重心距
    # 找到所有亮斑的轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 计算图像中心点
    center_x, center_y = target_size[0] // 2, target_size[1] // 2
    
    # 计算每个亮斑的质心到中心的距离
    distances = []
    for contour in contours:
        # 计算轮廓的矩
        M = cv2.moments(contour)
        if M["m00"] != 0:
            # 计算质心
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # 计算到中心的距离
            distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            distances.append(distance)
    
    # 计算平均距离
    mean_distance = np.mean(distances) if distances else 0
    
    # 保存输出图像
    if output_dir:
        output_dir = Path(output_dir)
        
        # 获取原始文件名（不含扩展名）
        base_name = Path(image_path).stem
        
        # 创建三个子目录
        resize_dir = output_dir / "resize"
        spots_dir = output_dir / "spots"
        gray_dir = output_dir / "gray"
        heatmap_dir = output_dir / "heatmap"
        
        resize_dir.mkdir(parents=True, exist_ok=True)
        spots_dir.mkdir(parents=True, exist_ok=True)
        gray_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存调整大小后的原始图像到resize目录 - 支持中文路径
        resized_path = resize_dir / f"{base_name}_resized.png"
        is_success, im_buf_arr = cv2.imencode(".png", img)
        if is_success:
            im_buf_arr.tofile(str(resized_path))
        
        # 保存灰度图到gray目录 - 支持中文路径
        gray_path = gray_dir / f"{base_name}_gray.png"
        is_success, im_buf_arr = cv2.imencode(".png", gray)
        if is_success:
            im_buf_arr.tofile(str(gray_path))
        
        # 保存亮斑图到spots目录 - 支持中文路径
        spots_path = spots_dir / f"{base_name}_spots.png"
        is_success, im_buf_arr = cv2.imencode(".png", binary)
        if is_success:
            im_buf_arr.tofile(str(spots_path))

        # 保存热力图到heatmap目录 - 支持中文路径
        heatmap_path = heatmap_dir / f"{base_name}_heatmap.png"
        is_success, im_buf_arr = cv2.imencode(".png", heatmap_bgr)
        if is_success:
            im_buf_arr.tofile(str(heatmap_path))
    
    return {
        "mean_brightness": mean_brightness,
        "lighted_area": lighted_area,
        "mean_distance": mean_distance
    }

def process_directory(input_dir, output_dir=None, brightness_threshold=200, target_size=(512, 512)):
    """
    处理目录中的所有图像
    
    参数:
    input_dir: 输入图像目录
    output_dir: 输出目录
    brightness_threshold: 亮度阈值
    target_size: 目标图像尺寸
    """
    input_dir = Path(input_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件（包括jpg和png）并按自然顺序排序
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(image_files, key=natural_sort_key)
    
    results = []
    for img_path in image_files:
        try:
            metrics = calculate_image_metrics(
                str(img_path),
                brightness_threshold=brightness_threshold,
                output_dir=output_dir,  # 直接传递主输出目录
                target_size=target_size
            )
            results.append({
                "image": img_path.name,
                **metrics
            })
            print(f"处理完成: {img_path.name}")
        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {str(e)}")
    
    return results

def create_colorbar(output_dir):
    """
    创建并保存色彩渐变条，显示亮度值与颜色的对应关系（照明工程风格）
    
    参数:
    output_dir: 输出目录
    """
    # 创建渐变条图像 (高x宽)
    colorbar_height = 50
    colorbar_width = 512
    margin_left = 30  # 左边距
    margin_right = 30  # 右边距
    margin_top = 30
    margin_bottom = 50
    total_height = colorbar_height + margin_top + margin_bottom
    total_width = colorbar_width + margin_left + margin_right
    
    # 创建白色背景
    colorbar_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # 创建照明工程风格的颜色映射
    heatmap_colors = [
        '#001a6e',  # deep blue (low)
        '#0aa5a7',  # blue-green
        '#1e9d4b',  # green
        '#b7e000',  # yellow-green
        '#ffd500',  # yellow
        '#ff8c1a',  # orange
        '#ff2a2a',  # bright red (high)
    ]
    cmap_dialux = LinearSegmentedColormap.from_list('dialux_like', heatmap_colors)
    
    # 创建颜色查找表
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        normalized_value = i / 255.0
        rgb_color = cmap_dialux(normalized_value)[:3]  # 获取RGB值（0-1范围）
        # 转换为BGR格式（OpenCV使用BGR）
        lut[i, 0] = int(rgb_color[2] * 255)  # B
        lut[i, 1] = int(rgb_color[1] * 255)  # G
        lut[i, 2] = int(rgb_color[0] * 255)  # R
    
    # 绘制渐变条
    for x in range(colorbar_width):
        # 将x坐标映射到0-255的亮度值
        brightness_value = int(x * 255 / (colorbar_width - 1))
        color = lut[brightness_value]
        colorbar_img[margin_top:margin_top+colorbar_height, margin_left + x] = color
    
    # 添加文字标注
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (0, 0, 0)  # 黑色
    
    # 标注标题
    title = "Brightness Value"
    title_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
    title_x = (total_width - title_size[0]) // 2
    cv2.putText(colorbar_img, title, (title_x, 20), font, font_scale, text_color, font_thickness)
    
    # 标注数值刻度（0, 64, 128, 192, 255）
    tick_values = [0, 64, 128, 192, 255]
    for val in tick_values:
        x_pos = margin_left + int(val * (colorbar_width - 1) / 255)
        
        # 绘制刻度线
        cv2.line(colorbar_img, (x_pos, margin_top + colorbar_height), 
                 (x_pos, margin_top + colorbar_height + 10), text_color, 1)
        
        # 添加数值标签
        text = str(val)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x_pos - text_size[0] // 2
        text_y = margin_top + colorbar_height + 30
        cv2.putText(colorbar_img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    
    # 保存渐变条
    colorbar_path = Path(output_dir) / 'colorbar_legend_heatmap.png'
    cv2.imwrite(str(colorbar_path), colorbar_img)
    print(f"照明风格色彩渐变条已保存到: {colorbar_path}")

def save_results_to_csv(results, output_dir):
    """
    将结果保存到CSV文件
    
    参数:
    results: 结果列表
    output_dir: 输出目录
    """
    # 检查结果是否为空
    if not results:
        print("\n警告: 没有成功处理任何图像，无法生成CSV文件")
        print("请检查:")
        print("1. 输入目录中是否有有效的图像文件")
        print("2. 图像文件路径中是否包含特殊字符导致无法读取")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 重命名列
    df.columns = ['图像名称', '平均亮度', '亮斑面积(像素)', '平均重心距(像素)']
    
    # 保存到CSV - 使用正确的相对路径
    csv_path = Path(output_dir) / 'image_metrics_results.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {csv_path}")

def main():
    # 基础路径：代码所在目录
    base_dir = Path(__file__).parent
    
    # 定义要处理的文件夹配置
    # 格式: (输入文件夹路径, 输出文件夹名称)
    folder_configs = [
        (base_dir / "baseline" / "CycleGAN", "CycleGAN"),
        (base_dir / "baseline" / "GPT4o", "GPT4o"),
        (base_dir / "baseline" / "Pix2Pix", "Pix2Pix"),
        (base_dir / "baseline" / "SD_lora", "SD_lora"),
        (base_dir / "baseline" / "VAE", "VAE"),
        (base_dir / "ground_truth", "ground_truth"),
        (base_dir / "Nightdiff", "Nightdiff"),
    ]
    
    # 输出基础目录
    output_base_dir = base_dir / "illumination" / "output_images" / "2d"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置处理参数
    target_size = (512, 512)
    brightness_threshold = 200
    
    # 依次处理每个文件夹
    for input_dir, folder_name in folder_configs:
        print(f"\n{'='*60}")
        print(f"正在处理: {folder_name}")
        print(f"输入目录: {input_dir}")
        print(f"{'='*60}")
        
        # 检查输入目录是否存在
        if not input_dir.exists():
            print(f"⚠ 警告: 输入目录不存在，跳过: {input_dir}")
            continue
        
        # 检查输入目录是否有图像文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(input_dir.glob(ext))
        
        if len(image_files) == 0:
            print(f"⚠ 警告: 输入目录中没有找到图像文件，跳过: {input_dir}")
            continue
        
        # 设置输出目录
        output_dir = output_base_dir / folder_name
        
        # 处理所有图像
        results = process_directory(
            input_dir, 
            output_dir, 
            brightness_threshold=brightness_threshold,
            target_size=target_size
        )
        
        # 保存结果到CSV
        save_results_to_csv(results, output_dir)
        
        # 生成照明风格色彩渐变条
        create_colorbar(output_dir)
        
        # 打印该文件夹的处理结果摘要
        print(f"\n{folder_name} 处理完成:")
        print(f"  成功处理: {len(results)} 张图像")
        if results:
            avg_brightness = np.mean([r['mean_brightness'] for r in results])
            total_area = sum([r['lighted_area'] for r in results])
            avg_distance = np.mean([r['mean_distance'] for r in results])
            print(f"  平均亮度: {avg_brightness:.2f}")
            print(f"  总亮斑面积: {total_area} 像素")
            print(f"  平均重心距: {avg_distance:.2f} 像素")
    
    print(f"\n{'='*60}")
    print("所有文件夹处理完成！")
    print(f"输出目录: {output_base_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 