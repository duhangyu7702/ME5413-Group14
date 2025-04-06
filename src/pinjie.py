import cv2
import sys
# 读取三张图片（请确保图片有足够的重叠区域）
img1 = cv2.imread("/home/du/ME5413_Final_Project/src/1.png")
img2 = cv2.imread("/home/du/ME5413_Final_Project/src/2.png")
img3 = cv2.imread("/home/du/ME5413_Final_Project/src/3.png")
img4 = cv2.imread("/home/du/ME5413_Final_Project/src/4.png")
img5 = cv2.imread("/home/du/ME5413_Final_Project/src/5.png")
img6 = cv2.imread("/home/du/ME5413_Final_Project/src/6.png")
img7 = cv2.imread("/home/du/ME5413_Final_Project/src/7.png")
img8 = cv2.imread("/home/du/ME5413_Final_Project/src/8.png")
# 将图片放入列表中
images = [img1, img2, img3, img4, img5, img6, img7, img8]

# 创建拼接器
# 注意：对于 OpenCV 3.4 及以上版本，可以使用 cv2.Stitcher_create()
# 如果使用旧版本的 OpenCV，可能需要使用 cv2.createStitcher()
stitcher = cv2.Stitcher_create()

# 执行拼接操作
(status, stitched) = stitcher.stitch(images)

# 根据状态码判断是否拼接成功
if status == cv2.Stitcher_OK:
    cv2.imwrite('result.jpg', stitched)
    print("拼接成功，结果保存在 result.jpg")
else:
    print("拼接失败，错误码为:", status)