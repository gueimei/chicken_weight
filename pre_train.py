import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np
from skimage import draw,color,transform,feature
import matplotlib.pyplot as plt
import math

def crop(img , x, y, x_max, y_max): #剪裁影像
    print(x, y ,x_max, y_max)
    return img[y:y_max, x:x_max]

def set_xy():
    temp = open(xml_source)
    tree=ET.parse(temp)
    root = tree.getroot()
    times = 0
    for obj in root.iter('object'):
        times += 1
        if times % 1 == 0:
            xmlbox = obj.find('bndbox')
            ##box -> {[0]==x,[1]==y,[2]==x_max,[3]==y_max}
            box = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            return box[0], box[1], box[2], box[3]

def grab_cut(source):
	# 读取图片
    img = source
	# 圖片寬、高
    img_x = img.shape[1]
    img_y = img.shape[0]
    # 準備分割的矩形範圍
    rect = (1, 1, img_x, img_y)
	# 背景模式,必须为1行,13x5列
    bgModel = np.zeros((1, 65), np.float64)
	# 前景模式,必须为1行,13x5列
    fgModel = np.zeros((1, 65), np.float64)
	# 图像掩模,取值有0,1,2,3
    mask = np.zeros(img.shape[:2], np.uint8)
	# grabCut处理,GC_INIT_WITH_RECT模式
    cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
	# 将背景0,2设成0,其余设成1
    mask2 = np.where((mask==2) | (mask==0) , 0, 1).astype('uint8')
	# 重新计算图像着色,对应元素相乘
    #img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.float64)
    #img = img + mask2[:, :, np.newaxis]
    #img = (img*255).astype(np.uint8)
    img = img*mask2[:, :, np.newaxis]
    #cv2.imshow("Result", img)
    #cv2.waitKey(0)
    return img

def RGB_equalization(img):
    #change RGB picture to ycrcb picture
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    color_channels = cv2.split(ycrcb)
    #均衡Y
    cv2.equalizeHist(color_channels[0], color_channels[0])
    cv2.merge(color_channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img
    
def detection_Edge(img):
    #邊緣偵測
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    #blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    #canny = feature.canny(blurred, sigma=2)
    canny = cv2.Canny(gray, 290, 310)
    y,x=np.nonzero(canny) #  注：矩陣的座標系表示和繪圖用的座標系不同，故需要做座標轉換
    canny_list = np.array([[_x,_y] for _x,_y in zip(x,y)]) # 邊界點座標
    return canny_list

def trans_To_Oval(edge):
    #执行椭圆变换
    #result = transform.hough_ellipse(edge, threshold=50, accuracy=5, min_size=1)
    result =transform.hough_ellipse(edge, accuracy=20, threshold=250,min_size=100, max_size=120)
    #result =transform.hough_ellipse(edge)
    result.sort(order='accumulator') #根据累加器排序
    print(result)
    
    #估计椭圆参数
    best = list(result[-1])  #排完序后取最后一个
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    #在原图上画出椭圆
    print("short:", str(a), "  long:", str(b))
    cy, cx =draw.ellipse_perimeter(yc, xc, a, b, orientation)
    crop_img[cy, cx] = (0, 0, 255) #在原图中用紅色表示检测出的椭圆
    cv2.imshow("final",crop_img)
    cv2.waitKey(0)
    
def fit_ellipse(edge_list, ori):
    _ellipse = cv2.fitEllipse(edge_list) #calculate ellipse
    print(_ellipse)
    edge_clone=ori.copy()
    cv2.ellipse(edge_clone, _ellipse, (255,0,0),2) #paint ellipse
    plt.imshow(edge_clone)
    return _ellipse[1][0]/2, _ellipse[1][1]/2
    
def contrast_Img(img, contrast, bright):
    h, w, channel = img.shape
    black_img = np.zeros([h, w, channel], img.dtype)
    const_img = cv2.addWeighted(img, contrast, black_img, 1-contrast, bright)
    return const_img

def calcu_area(img):
    height, width = img.shape
    area = 0
    print(height, width)
    for i in range(height):
        for j in range(width):
            if img[i, j] == 0:
                area += 1
    return area

def calcu_ellipse_area(short, long):
    ellipse_area = math.pi * short * long
    return ellipse_area

if __name__ == "__main__":
    '''
    sets = [('train_name')]
    classes = ['chicken']
    for images in sets:
        if not os.path.exists('D:\\lab\\chicken_video\\dataset\\test'):
            os.makedirs('D:\\lab\\chicken_video\\dataset\\test')
    image_ids = open('D:\\lab\\chicken_video\\dataset\\%s.txt'%(images)).read().strip().split()
    
    for image_id in image_ids:
        img_source = "D:\\lab\\chicken_video\\dataset\\images\\%s.jpg"%(image_id)
        xml_source = "D:\\lab\\chicken_video\\dataset\\xml\\%s.xml"%(image_id)
        img = cv2.imread(img_source)
        xml = open(xml_source)
        xml_out = xml.read()
'''
    x = 793
    y = 436
    x_max = 901
    y_max = 655
    
    x = 502
    y = 868
    x_max = 726
    y_max = 987
        
    img = cv2.imread("D:\\lab\\chicken_video\\2019.6.27\\20190627_062900_ch3.jpg")

    img = RGB_equalization(img)
    
    #x, y ,x_max, y_max = set_xy()
    crop_img = crop(img, x, y, x_max, y_max)
    #cv2.imshow("cropped", crop_img)
    
    contr_img = contrast_Img(crop_img, 1.5, 3)
    grab = grab_cut(contr_img)
    gray = cv2.cvtColor(grab, cv2.COLOR_BGR2GRAY)
    
    area = calcu_area(gray)
    print("area:", str(area))
    
    edge = detection_Edge(grab) ##return edge point list
    #trans_To_Oval(edge)
    longAxis, shortAxis = fit_ellipse(edge, crop_img)
    ellipseArea = calcu_ellipse_area(shortAxis, longAxis)
    print("ellipse area:", str(ellipseArea))
    
    cv2.waitKey(0)