import cv2
import pytesseract
import numpy as np

def deskew(image):
    """
    对图像进行倾斜校正，确保数字区域水平对齐
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 获取非零像素坐标
    coords = np.column_stack(np.where(gray > 0))
    # 计算最小外接矩形的旋转角度
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # 根据计算的角度旋转图像
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# 读取待识别的图像
img = cv2.imread('/home/du/ME5413_Final_Project/src/W.png')

# 图像预处理：校正倾斜
rotated = deskew(img)

# 转换为灰度图像并进行二值化处理
gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 配置Tesseract参数：
# --psm 7：将图像作为单行文本处理
# tessedit_char_whitelist：限制只识别数字
custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789'

# 进行OCR识别，并获取每个识别到字符的位置信息
data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)

# 提取识别到的数字及其在图像中的左侧坐标
num_list = []
for i, text in enumerate(data['text']):
    if text.strip() != "" and text.strip().isdigit():
        x = data['left'][i]
        num_list.append((x, text.strip()))

# 按照左坐标排序，确保顺序为从左到右
num_list.sort(key=lambda x: x[0])

# 将识别的数字依次排列
result = [num for (x, num) in num_list]
print("识别到的数字顺序:", result)

