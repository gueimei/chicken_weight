import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np
from skimage import draw,color,transform,feature
import matplotlib.pyplot as plt
import math
import time

def crop(img , x, y, x_max, y_max): #剪裁影像
    return img[y:y_max, x:x_max]


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


def fit_ellipse(edge_list, ori, x):
    _ellipse = cv2.fitEllipse(edge_list) #calculate ellipse
    edge_clone=ori.copy()
    #cv2.ellipse(edge_clone, _ellipse, (0, 0, 255),2) #paint ellipse
    #cv2.imwrite("D:\\lab\\chicken_project\\dataset\\weight_test\\%d.jpg"%(x), edge_clone)
    cv2.imwrite("/home/pytorch-yolo-v3/imgs/%d.jpg"%(x), edge_clone)
    #plt.imshow(edge_clone)
    return _ellipse[1][0]/2, _ellipse[1][1]/2
    
def contrast_Img(img, contrast, bright):
    h, w, channel = img.shape
    black_img = np.zeros([h, w, channel], img.dtype)
    const_img = cv2.addWeighted(img, contrast, black_img, 1-contrast, bright)
    return const_img

def calcu_area(img):
    height, width = img.shape
    area = 0
    for i in range(height):
        for j in range(width):
            if img[i, j] != 0:
                area += 1
    return area

def calcu_ellipse_area(short, long):
    ellipse_area = math.pi * short * long
    return ellipse_area

def calcu_ellipse_eccentricity(short, long):
    focal = math.sqrt(long * long - short *short)
    e = focal / long
    return e

def calcu_hull(edge, ori):
    hull = cv2.convexHull(edge)
    #cv2.polylines(ori, [hull], True, ( 0, 255, 0), 2)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    
    M = cv2.moments(hull)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    #cv2.circle(ori, (cX, cY), 5, ( 0, 0, 227), -1)
    return hull_area, hull_perimeter, cX, cY


## data loader and set
def load_xml(): #load every xml file
    sets = [('weight_train_name')]
    ac_data = []
    train_data = []
    #classes = ['chicken']
    for files in sets:
        if not os.path.exists('D:\lab\chicken_project\dataset\weight_test'):
            os.makedirs('D:\lab\chicken_project\dataset\weight_test')
    ids = open('D:\\lab\\chicken_project\\dataset\\%s.txt'%(files)).read().strip().split()
    
    for ID in ids:
        img_source = "D:\\lab\\chicken_project\\dataset\\weight_images\\%s.jpg"%(ID)
        xml_source = "D:\\lab\\chicken_project\\dataset\\weight_xml\\%s.xml"%(ID)
        img = cv2.imread(img_source)
        #print (ID)
        load_Data(xml_source, ac_data, train_data, img)
    
    return ac_data, train_data

        
def load_Data(source, ac_data, train_data, img): ##load necessary data from a xml file(object coordinate)
    xml = open(source)
    tree=ET.parse(xml)
    root = tree.getroot()
    times = 0
    for obj in root.iter('object'):
        ac_data.append(float(obj[0].text)) ##load ac_data
        times += 1
        xmlbox = obj.find('bndbox')
        ##box -> {[0]==x,[1]==y,[2]==x_max,[3]==y_max}
        box = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        #start = time.time()
        input_data = collect_data(box[0], box[1], box[2], box[3], img)
        if input_data[6] == 0:
            ac_data.pop()
            print("pop")
        else:
            input_data = list(map(lambda x: math.log(x), input_data))
            train_data.append(input_data) ##load train_data
        #cost = time.time()-start
            
            
def collect_data(x, y, x_max, y_max, img):
    #print(x, y ,x_max, y_max)
    ori = crop(img, x, y, x_max, y_max)
    img = RGB_equalization(img)
    
    crop_img = crop(img, x, y, x_max, y_max)
    #cv2.imshow("cropped", crop_img)
    
    contr_img = contrast_Img(crop_img, 1.5, 3)
    blurred = cv2.GaussianBlur(contr_img, (9, 9), 0)
    grab = grab_cut(blurred)
    gray = cv2.cvtColor(grab, cv2.COLOR_BGR2GRAY)
    
    area = calcu_area(gray)
    if area < 30:
        print("small", str(x))
        return [0, 0, 0, 0, 0, 0, 0, 0]
    
    edge = detection_Edge(grab) ##return edge point list
    
    hullArea, perimeter, gravity_cx, gravity_cy = calcu_hull(edge, ori)
    gravity_cx += x #real location
    gravity_cy += y
    
    shortAxis, longAxis = fit_ellipse(edge, ori, x)
    ellipseArea = calcu_ellipse_area(shortAxis, longAxis)
    eccentricity = calcu_ellipse_eccentricity(shortAxis, longAxis)
        
    data = [float(gravity_cx), float(gravity_cy), hullArea, perimeter,
            shortAxis, longAxis, ellipseArea, eccentricity]
    
    return data
    
if __name__ == "__main__":
    
    data = load_xml()
    print(data)
    
    cv2.waitKey(0)
