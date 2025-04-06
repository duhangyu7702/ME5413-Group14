import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图片
image = cv2.imread('/home/du/ME5413_Final_Project/src/p.jpg')

# 转换颜色通道（OpenCV是BGR格式，matplotlib需要RGB格式）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图片尺寸
height, width = image.shape[:2]

# 创建一个与图像大小相同的掩码，初始为全白（255）
mask = np.ones_like(image_rgb, dtype=np.uint8) * 255

# 定义三角形的三个顶点
triangle = np.array([
    [0, 0],
    [0, int(0.3 * height)],
    [int(0.85 * width), 0]
])

# 在掩码上绘制三角形并填充为黑色（0, 0, 0）
cv2.fillPoly(mask, [triangle], (0, 0, 0))

# 将掩码应用到图像上
image_rgb = cv2.bitwise_and(image_rgb, mask)

# 显示结果
plt.imshow(image_rgb)
plt.axis('off')
plt.show()