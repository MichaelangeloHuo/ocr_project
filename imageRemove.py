#coding:utf-8
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import itertools
from PIL import ImageEnhance
from glob import glob
import os.path

def get_image(path):
    #获取图片
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def Gaussian_Blur(gray):
    # 高斯去噪
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    return blurred

def Sobel_gradient(blurred):
    # 索比尔算子来计算x、y方向梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient

def Thresh_and_blur(gradient):

    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    (_, thresh) = cv2.threshold(blurred, 130, 255, cv2.THRESH_OTSU)

    return thresh

def image_morphology(thresh):
    # 建立一个椭圆核函数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    # 执行图像形态学, 细节直接查文档，很简单
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    return closed

def findcnts_and_box_point(original_img,closed):
    # 这里opencv3返回的是三个参数
    (fc_img, cnts, hierarchy) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    # print(c)
    box = np.int0(cv2.boxPoints(rect))

    dds_img = cv2.drawContours(original_img.copy(),cnts,-1,(0,255,0),3)
    return box

def drawcnts_and_cut(original_img, box):
    # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
    # draw a bounding box arounded the detected barcode and display the image
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

    originalImgShape = original_img.shape
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = max(min(Xs), 0)
    x2 = min(max(Xs), originalImgShape[1])
    y1 = max(min(Ys), 0)
    y2 = min(max(Ys), originalImgShape[0])
    hight = y2 - y1
    width = x2 - x1
    #print(y1, hight, x1, width)
    crop_img = original_img[y1:y1+hight, x1:x1+width]

    shape = crop_img.shape
    '''if shape[0] and shape[1]:#长宽大于0
        cv2.namedWindow('crop_img',cv2.WINDOW_NORMAL)
        cv2.imshow('crop_img',crop_img)'''
    return draw_img, crop_img

def remove(original_img,box):
    originalImgShape = original_img.shape
    removedImg = original_img
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = max(min(Xs), 0)
    x2 = min(max(Xs), originalImgShape[1])
    y1 = max(min(Ys), 0)
    y2 = min(max(Ys), originalImgShape[0])
    hight = y2 - y1
    width = x2 - x1
    white = np.zeros((hight, width, 3)) + 255
    removedImg[y1:y1+hight, x1:x1+width] = white
    return removedImg

def getImageHorizontalAndVerticalSum(ImageThre):
    rows, cols = ImageThre.shape
    horsum = []
    versum = []
    for i in range(cols):
        val = np.array(ImageThre[:, i]).sum()
        horsum.append(val)
        # print(val)
    for i in range(rows):
        val = np.array(ImageThre[i, :]).sum()
        versum.append(val)
        # print(val)
    return horsum, versum

def imageCrop(imagePath):

    img_path = imagePath
    original_img, gray = get_image(img_path)
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)
    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    box = findcnts_and_box_point(original_img, closed)
    draw_img, crop_img = drawcnts_and_cut(original_img, box)
    removedImg = remove(original_img.copy(), box)

    return removedImg, crop_img

def ImageEnhanceSelf(image):
    #file = 'E:\ocr\\testFigures\\test8.jpg'
    #image = Image.open(file)
    #image.show()
    #print(image)
    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 2.5
    image_brightened = enh_bri.enhance(brightness)
    #image_brightened.show()
    # 色度增强
    enh_col = ImageEnhance.Color(image_brightened)
    color = 1.5
    image_colored = enh_col.enhance(color)
    #image_colored.show()
    # 对比度增强
    enh_con = ImageEnhance.Contrast(image_colored)
    contrast = 3
    image_contrasted = enh_con.enhance(contrast)
    #image_contrasted.show()
    # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    sharpness = 2.0
    image_sharped = enh_sha.enhance(sharpness)
    #image_sharped.show()
    return image_sharped

def custom_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
    h, w = gray.shape[:2]
    #print(h, w)
    m = np.reshape(gray, [1, w*h])
    mean = m.sum()/(w*h)
    #print("mean:", mean)
    ret, binary = cv2.threshold(gray, mean, 1, cv2.THRESH_BINARY)
    #img = Image.fromarray(binary, 'L')
    #img.show()
    return binary

def projectionY(thresh1):
    # cv2.imshow(thresh1)
    (h, w) = thresh1.shape  # 返回高和宽
    sumX = w - thresh1.sum(axis=1)  # 对行求和
    sumXBinary = sumX
    #print(sumXBinary)
    for i in range(0, h):
        # print(sumXBinary[i])
        if sumXBinary[i] > 2:#每行像素超过2个，则认为有值
            sumXBinary[i] = 1
        else:
            sumXBinary[i] = 0
    #print(sumXBinary)
    num_times = [(k, len(list(v))) for k, v in itertools.groupby(sumXBinary)]
    #print(num_times)
    #print(num_times)
    #print(num_times)
    sum = 0
    flag_start = []
    flag_end = []
    for i in range(0, len(num_times)):
        if num_times[i][1] <= 250:
            sum = sum + num_times[i][1]
        elif num_times[i][1] >= 250 and num_times[i][0] == 0:
            sum = sum + num_times[i][1]
        elif num_times[i][1] >= 250 and num_times[i][0] == 1:
            #print(num_times[i][1])
            flag_start.append(sum)
            flag_end.append(num_times[i][1] + sum)
            sum = num_times[i][1] + sum
    return flag_start, flag_end

def singleImageRemove(image_file):
    imgOrigin = Image.open(image_file)
    imageEnhance = ImageEnhanceSelf(imgOrigin)
    img2 = cv2.cvtColor(np.asarray(imageEnhance), cv2.COLOR_RGB2BGR)
    gray = custom_threshold(img2)
    #print(gray)
    flagY_start, flagY_end = projectionY(gray)
    #print(flagY_start, flagY_end)
    if len(flagY_start) == 0 and len(flagY_end) == 0:
        #print("no images")
        pass
    else:
        horsum, versum = getImageHorizontalAndVerticalSum(gray)
        #print('=====\n', imgName, '\n', len(horsum), '\n', len(versum))
        p, imageName = os.path.split(image_file)
        print("the image is :" + imageName)
        removedImg, crop_img = imageCrop(image_file)
        cv2.namedWindow('rm', cv2.WINDOW_NORMAL)
        cv2.imshow('rm', crop_img)
        cv2.namedWindow('or', cv2.WINDOW_NORMAL)
        cv2.imshow('or', removedImg)
        cv2.waitKey(1000)
        print(type(removedImg))
    return removedImg

if __name__ == '__main__':
    imgsName = glob('E:\ocr\TestInput\\*.jpg')
    #filePath = './TestInput/'
    #filePath = 'E:\ocr\TestInput\\'
    #imgsName = os.listdir(filePath)
    for imagePath in sorted(imgsName):
        #imagePath = filePath + imgName
        singleImageRemove(imagePath)
        '''
        # 投影
        img,gray = get_image(imagePath)
        blurred = Gaussian_Blur(gray)
        horsum,versum = getImageHorizontalAndVerticalSum(blurred)
        print('=====\n',imgName,'\n',len(horsum),'\n',len(versum))
        plt.subplot(121)
        plt.plot(horsum)
        plt.subplot(122)
        plt.plot(versum)
        plt.show()
        '''
        # 形态学
        # removeImg = imageCrop(imagePath)
        # cv2.namedWindow('rm',cv2.WINDOW_NORMAL)
        # cv2.imshow('rm',removeImg)
        # cv2.waitKey(1000)
    cv2.destroyAllWindows()
