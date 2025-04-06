import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract  # 纯 CPU 的 OCR 方案

def preprocess_image(image_path):
    """图像加载与预处理（仅使用 CPU）"""
    # 1. 读取图像（在 CPU 上读取）
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图像 {image_path} 未找到或无法读取")

    # 2. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 4. 固定阈值二值化
    ret, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    # 5. 使用连通域分析去除小噪点
    min_area = 50  # 根据实际情况调整
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for i in range(1, num_labels):  # 忽略背景标签 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == i] = 255

    # 6. 形态学闭操作填充数字内部的间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

    # 7. 反转图像
    processed = cv2.bitwise_not(closed)

    return image, gray, processed

def ocr_digits(processed_image):
    """执行 OCR 识别（使用 pytesseract，仅使用 CPU）"""
    # 使用 pytesseract 执行 OCR 识别，--psm 6 适用于单一文本块
    result = pytesseract.image_to_string(processed_image, config='--psm 6')
    # 合并所有识别结果，并过滤出数字
    digits = ''.join(filter(str.isdigit, result))
    return digits.strip()

def visualize_results(image, gray, processed, digits):
    """可视化处理流程"""
    plt.figure(figsize=(15, 5))

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # 灰度图像
    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Gray Image")
    plt.axis('off')

    # 二值化图像
    plt.subplot(1, 3, 3)
    plt.imshow(processed, cmap='gray')
    plt.title("Processed Image")
    plt.axis('off')

    plt.suptitle(f"OCR results: {digits}", y=0.85)
    plt.show()

# ======================================
# 主程序
# ======================================
if __name__ == "__main__":
    # 图像路径（请根据实际情况修改）
    image_path = "/home/du/ME5413_Final_Project/src/3.png"

    # 预处理图像（仅使用 CPU）
    image, gray, processed = preprocess_image(image_path)

    # OCR 识别（使用 pytesseract，仅使用 CPU）
    digits = ocr_digits(processed)

    # 结果输出
    if not digits:
        print("⚠️ 未检测到数字！请检查：")
        print("  1. 图像是否包含清晰数字")
        print("  2. 阈值是否合适（尝试调整 preprocess_image 中的 thresh 值）")
    else:
        print("✅ 最终识别结果:", digits)

    # 可视化结果
    visualize_results(image, gray, processed, digits)
