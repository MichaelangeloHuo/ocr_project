import re

def replaceTextOut(textInput):
    regex = r'((\(*([0-9.a-zA-Z=]+)[-+\(\)0-9.a-zA-Z\\=]+)|(-[0-9.]+)|=)'
    str = re.search(regex, textInput)
    len = 0
    textOut = textInput
    while str!=None:
        textOut = textOut[:str.start()+len]+"$"+ str.group()+ "$" + textOut[str.end()+len:]
        textInput = textInput[:str.start()] + "$" + "$" + textInput[str.end():]
        len += str.end() - str.start()
        str = re.search(regex, textInput)
    return textOut

if __name__ == '__main__':
    str = "用竖sss式计算。  a\\div9.2-(6.7+9.2)=  -4.81 ="
    str = replaceTextOut(str)
    print(str)