#-*- coding:utf-8 -*-
import os
import ocr
import time
import shutil
import sys
import numpy as np
from PIL import Image
from glob import glob
from gensim import corpora, models, similarities
import pandas as pd
from match import match
import re

#编码问题
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

image_files = glob('./TestInput/image_37.jpg')

def replaceTextOut(textOut):
    #if u'×' in textOut:
    textOut = re.sub(u'×', r'\\times', textOut)
    #if u'÷' in textOut:
    textOut = re.sub(u'÷', r'\div', textOut)
    textOut = re.sub(u'\[', r'{\\rm{[}}', textOut)
    textOut = re.sub(u']', r'{\\rm{]}}', textOut)
    textOut = re.sub(u'\xb0', r'^{\\rm{^\circ }}', textOut)#修改度
    return textOut

if __name__ == '__main__':
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        image = np.array(Image.open(image_file).convert('RGB'))
        t = time.time()
        result, image_framed = ocr.model(image)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)

        #print("result:", result)
        textResult = []
        for key in result:
            #print(result[key][0])
            textResult.append(result[key][1])
        textResultOut = ''.join(textResult)
        print(textResultOut)
        textResultOut = replaceTextOut(textResultOut)
        print(textResultOut)
        #print("\nRecognition Result:\n")
        #print(textResultOut)
        '''with open('./test_result.txt', 'a+') as f:
            f.write(textResultOut)
            f.write('\r\n')'''

    match(textResultOut, 0.1)#文本匹配，连续相似的值不超过0.1

print("Mission complete, it took {:.3f}s".format(time.time() - t))