from PIL import Image
import numpy as np

# 假设你有一组图像文件，存储在一个列表中
image_files = ['cat1.jpg', 'cat2.jpg', 'dog1.jpg', 'dog2.jpg']

# 创建一个空列表，用于存储图像数据
images = []

# 遍历每个图像文件
for image_file in image_files:
    # 使用 PIL 库打开图像文件
    image = Image.open(image_file)

    # 将图像转换为灰度图像
    image = image.convert('L')

    # 将图像调整为指定大小（例如 128x128 像素）
    image = image.resize((128, 128))

    # 将图像转换为 numpy 数组
    image_array = np.array(image)

    # 将图像数据添加到列表中
    images.append(image_array)

# 将图像数据堆叠在一起，形成一个大型数据数组
data = np.stack(images)

# 将数据数组保存到文件中
np.save('data.npy', data)