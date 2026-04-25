import cv2
import os

def convert_to_u8_512(input_path, output_path):
    # 1. 读取图像
    # cv2.IMREAD_GRAYSCALE 确保以灰度模式读取，即使原图是彩色
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"错误: 无法加载图像 {input_path}")
        return

    # 2. 缩放到 512x512
    # cv2.INTER_AREA 适合缩小图像，cv2.INTER_CUBIC 适合放大
    resized_img = cv2.resize(img, (513, 513), interpolation=cv2.INTER_LANCZOS4)

    # 3. 确保数据类型是 uint8 (u8)
    # 如果原图是 16-bit 或 浮点数，此操作会强制截断/转换
    u8_img = resized_img.astype('uint8')

    # 4. 保存结果
    cv2.imwrite(output_path, u8_img)
    print(f"转换成功！已保存至: {output_path}")

# 使用示例
input_file = "T96_1.png"  # 你的原始高度图
output_file = "T96.png"
convert_to_u8_512(input_file, output_file)
