# from PIL import Image
# import numpy as np
#
# # 读取CAM生成的热力图
# cam_heatmap_path = "/disk/sdb/zhangqian0620/fenlei/1_16_re.png"
# cam_heatmap = Image.open(cam_heatmap_path).convert('L')  # 转为灰度图
#
# # 将CAM热力图转换为二值图像
# threshold = 127  # 设定阈值
# cam_binary = np.array(cam_heatmap) > threshold  # 使用阈值进行二值化处理
#
# # 读取分割网络的预测结果和真实标签
# segmentation_prediction_path = "/disk/sdb/zhangqian0620/fenlei/097.png"
# true_label_path = "/disk/sdb/zhangqian0620/fenlei/098.png"
#
# segmentation_prediction = Image.open(segmentation_prediction_path).convert('L')
# true_label = Image.open(true_label_path).convert('L')
#
# # 将分割预测结果和真实标签转换为二值图像
# segmentation_prediction = np.array(segmentation_prediction) > 127  # 以127为阈值二值化
# true_label = np.array(true_label) > 127  # 以127为阈值二值化
#
# # 结合CAM输出、分割结果和真实标签生成伪标签
# pseudo_label = np.logical_and(np.logical_and(cam_binary, segmentation_prediction), true_label).astype(np.uint8) * 255
#
# # 保存伪标签
# pseudo_label_path = "he.png"
# Image.fromarray(pseudo_label).save(pseudo_label_path)  # 保存伪标签图像
#
# # 现在伪标签图像已经生成


import os
from PIL import Image
import numpy as np

import os
from PIL import Image
import numpy as np


import os
from PIL import Image
import numpy as np

def generate_pseudo_label(cam_heatmap_path, segmentation_prediction_path, true_label_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取CAM生成的热力图并转为灰度图
    cam_heatmap = Image.open(cam_heatmap_path).convert('L')

    # 将CAM热力图转换为二值图像
    threshold = 127
    cam_binary = np.array(cam_heatmap) > threshold

    # 读取分割网络的预测结果和真实标签，并转为灰度图
    segmentation_prediction = Image.open(segmentation_prediction_path).convert('L')
    true_label = Image.open(true_label_path).convert('L')

    # 将分割预测结果和真实标签转换为二值图像
    segmentation_prediction_binary = np.array(segmentation_prediction) > threshold
    true_label_binary = np.array(true_label) > threshold


    # 结合CAM输出、分割结果和真实标签生成伪标签
    pseudo_label = np.logical_and(np.logical_and(cam_binary, segmentation_prediction_binary), true_label_binary).astype(
        np.uint8) * 255

    # 计算白色区域的占比
    white_area_ratio = np.sum(pseudo_label == 255) / (pseudo_label.shape[0] * pseudo_label.shape[1])

    white_area_threshold = 0.1
    # 如果白色区域占比小于阈值，则使用分割预测和真实标签的结合结果
    if white_area_ratio < white_area_threshold:
        pseudo_label = np.logical_and(segmentation_prediction_binary, true_label_binary).astype(np.uint8) * 255

    # 获取热力图文件名（不包含路径）
    cam_filename = os.path.basename(cam_heatmap_path)

    # 保存伪标签到输出文件夹，文件名与热力图相同
    pseudo_label_path = os.path.join(output_folder, cam_filename)
    Image.fromarray(pseudo_label).save(pseudo_label_path)

# 定义热力图、预测标签、真实标签的文件夹路径
cam_heatmap_folder = "/disk/sdb/zhangqian0620/fenlei/out_mamba"
segmentation_prediction_folder = "/disk/sdb/zhangqian0620/VM-UNet-main/output_olp_300_train"
true_label_folder = "/disk/sdb/zhangqian0620/VM-UNet-main/data/olp_box/train/masks"


# 定义输出文件夹路径
output_folder = "/disk/sdb/zhangqian0620/fenlei/pseudo_labels"

# 获取所有文件名（这里假设所有文件都是.png格式）
file_list = [f for f in os.listdir(cam_heatmap_folder) if f.endswith('.png')]

# 迭代文件列表，为每个文件生成伪标签
for filename in file_list:
    # 构造完整的文件路径
    cam_heatmap_path = os.path.join(cam_heatmap_folder, filename)
    segmentation_prediction_path = os.path.join(segmentation_prediction_folder, filename)
    true_label_path = os.path.join(true_label_folder, filename)

    # 调用函数生成伪标签
    generate_pseudo_label(cam_heatmap_path, segmentation_prediction_path, true_label_path, output_folder)



# import cv2
#
# # 读取图像
# image_path = "/disk/sdb/zhangqian0620/fenlei/098.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像
#
# # 计算像素值的最大值
# max_pixel_value = image.max()
#
# print("图像中像素值的最大值是：", max_pixel_value)