import os
import cv2 as cv
import numpy as np
import random
from math import *
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


chars = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
     '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
     '新',
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
     'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
     'V', 'W', 'X', 'Y', 'Z', 'I'
     ]

index = {j:i for i, j in enumerate(chars)}

dicts = {'A01':'京','A02':'津','A03':'沪','B02':'蒙',
         'S01':'皖','S02':'闽','S03':'粤','S04':'甘',
        'S05': '贵', 'S06': '鄂', 'S07': '冀', 'S08': '黑', 'S09': '湘',
        'S10': '豫', 'S12': '吉', 'S13': '苏', 'S14': '赣', 'S15': '辽',
        'S17': '川', 'S18': '鲁', 'S22': '浙',
        'S30':'渝', 'S31':'晋', 'S32':'桂', 'S33':'琼', 'S34':'云', 'S35':'藏',
        'S36':'陕','S37':'青', 'S38':'宁', 'S39':'新'}

decode_dicts = {j:i for i, j in dicts.items()}


def AddSmudginess(img, Smu):
    """
    模糊处理
    :param img: 输入图像
    :param Smu: 模糊图像
    :return: 添加模糊后的图像
    """
    rows = r(Smu.shape[0] - 50)
    cols = r(Smu.shape[1] - 50)
    adder = Smu[rows:rows + 50, cols:cols + 50]
    adder = cv.resize(adder, (50, 50))
    img = cv.resize(img,(50,50))
    img = cv.bitwise_not(img)
    img = cv.bitwise_and(adder, img)
    img = cv.bitwise_not(img)
    return img


def rot(img, angel, shape, max_angel):
    """
    添加透视畸变
    """
    size_o = [shape[1], shape[0]]
    size = (shape[1]+ int(shape[0] * cos((float(max_angel ) / 180) * 3.14)), shape[0])
    interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]))
    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])
    if angel > 0:
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, size)
    return dst


def rotRandrom(img, factor, size):
    """
    添加放射畸变
    :param img: 输入图像
    :param factor: 畸变的参数
    :param size: 图片目标尺寸
    :return: 放射畸变后的图像
    """
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [r(factor), shape[0] - r(factor)], [shape[1] - r(factor), r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, size)
    return dst


def tfactor(img):
    """
    添加饱和度光照的噪声
    """
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + np.random.random() * 0.7)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.2 + np.random.random() * 0.8)
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return img


def random_envirment(img, noplate_bg):
    """
    添加自然环境的噪声, noplate_bg为不含车牌的背景图
    """
    bg_index = r(len(noplate_bg))
    env = cv.imread(noplate_bg[bg_index])
    env = cv.resize(env, (img.shape[1], img.shape[0]))
    bak = (img == 0)
    bak = bak.astype(np.uint8) * 255
    inv = cv.bitwise_and(bak, env)
    img = cv.bitwise_or(inv, img)
    return img


def GenCh(f, val):
    """
    生成中文字符
    """
    img = Image.new("RGB", (45, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3), val, (0, 0, 0), font=f)
    img =  img.resize((23, 70))
    A = np.array(img)
    return A


def GenCh1(f, val):
    """
    生成英文字符
    """
    img =Image.new("RGB", (23, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2), val, (0, 0, 0), font=f)    # val.decode('utf-8')
    A = np.array(img)
    return A


def AddGauss(img, level):
    """
    添加高斯模糊
    """ 
    return cv.blur(img, (level * 2 + 1, level * 2 + 1))


def r(val):
    return int(np.random.random() * val)


def AddNoiseSingleChannel(single):
    """
    添加高斯噪声
    """
    diff = 255 - single.max()
    noise = np.random.normal(0, 1 + r(6), single.shape)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise *= diff
    # noise= noise.astype(np.uint8)
    dst = single + noise
    return dst


def addNoise(img):    # sdev = 0.5,avg=10
    img[:, :, 0] = AddNoiseSingleChannel(img[:, :, 0])
    img[:, :, 1] = AddNoiseSingleChannel(img[:, :, 1])
    img[:, :, 2] = AddNoiseSingleChannel(img[:, :, 2])
    return img

class GenPlate:
    def __init__(self, fontCh, fontEng, NoPlates):
        self.fontC = ImageFont.truetype(fontCh, 43, 0)
        self.fontE = ImageFont.truetype(fontEng, 60, 0)
        #self.img = np.array(Image.new("RGB", (226, 70),(255, 255, 255)))
        self.img = np.array(Image.new("RGB", (226, 70),(255, 255, 255)))
        self.bg  = cv.resize(cv.imread("/home/zj/ocr/car_data/images/template.bmp"), (226, 70))    # template.bmp:车牌背景图
        #self.bg  = cv.resize(cv.imread("/home/zj/ocr/car_data/images/yellow.jpg"), (226, 70))
        self.smu = cv.imread("/home/zj/ocr/car_data/images/smu2.jpg")    # smu2.jpg:模糊图像
        self.noplates_path = []
        for parent, parent_folder, filenames in os.walk(NoPlates):
            for filename in filenames:
                path = parent + "/" + filename
                self.noplates_path.append(path)

    def draw(self, val):
        offset = 2
        self.img[0:70, offset+8:offset+8+23] = GenCh(self.fontC, val[0])
        self.img[0:70, offset+8+23+6:offset+8+23+6+23] = GenCh1(self.fontE, val[1])
        for i in range(5):
            base = offset + 8 + 23 + 6 + 23 + 17 + i * 23 + i * 6
            self.img[0:70, base:base+23] = GenCh1(self.fontE, val[i+2])
        return self.img

    def generate(self, text):
        if len(text) == 7:
            fg = self.draw(text)    # decode(encoding="utf-8")
            #com = cv.bitwise_and(fg, self.bg) #黄色底黑色字
            fg = cv.bitwise_not(fg)
            com = cv.bitwise_or(fg, self.bg)
            #com = rot(com, r(60)-30, com.shape,30)
            #com = rotRandrom(com, 10, (com.shape[1], com.shape[0]))
            #com = tfactor(com)
            # com = random_envirment(com, self.noplates_path)
            com = AddGauss(com, 1+r(4))
            com = addNoise(com)
            return com

    @staticmethod
    def genPlateString(pos, val):
        """
        生成车牌string，存为图片
        生成车牌list，存为label
        """
        plateStr = ""
        plateList=[]
        box = [0, 0, 0, 0, 0, 0, 0]
        if pos != -1:
            box[pos] = 1
        for unit, cpos in zip(box, range(len(box))):
            if unit == 1:
                plateStr += val
                plateList.append(val)
            else:
                if cpos == 0:
                    plateStr += chars[r(31)]
                    plateList.append(plateStr)
                elif cpos == 1:
                    plateStr += chars[41 + r(26)]
                    plateList.append(plateStr)
                else:
                    random_num = random.random()
                    if random_num < 0.33:
                        plateStr += chars[41 + r(26)]
                        plateList.append(plateStr)
                    else:
                        plateStr += chars[31 + r(10)]
                        plateList.append(plateStr)
        plate = [plateList[0]]
        b = [plateList[i][-1] for i in range(len(plateList))]
        plate.extend(b[1:7])
        return plateStr, plate

    @staticmethod
    def genBatch(batchsize, outputPath, size):
        """
        将生成的车牌图片写入文件夹，对应的label写入label.txt
        :param batchsize:  批次大小
        :param outputPath: 输出图像的保存路径
        :param size: 输出图像的尺寸
        :return: None
        """
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        outfile = open('/home/zj/ocr/car_data/label.txt', 'w', encoding='utf-8')
        for i in range(batchsize):
            plateStr, plate = G.genPlateString(-1, -1)
            print(plateStr, plate)
            name = str(decode_dicts[plateStr[0]]) + '_' + str(plateStr[1:]) + '_0'
            img = G.generate(plateStr)
            img = cv.resize(img, size)
            cv.imwrite(outputPath + "/" + name + ".jpg", img)
            outfile.write(str(plate) + "\n")


if __name__ == '__main__':
    G = GenPlate("/home/zj/ocr/car_data/font/platech.ttf", '/home/zj/ocr/car_data/font/platechar.ttf', "/home/zj/ocr/car_data/NoPlates")
    G.genBatch(2000, '/home/zj/ocr/car_data/data/plate', (136, 36))