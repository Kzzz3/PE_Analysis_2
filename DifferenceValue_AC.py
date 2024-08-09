import os

from PIL import Image
from tqdm import tqdm
import numpy as np


def CalculateDV(strImagePath):
    if not os.path.exists(strImagePath):
        return "File not found"

    img = Image.open(strImagePath)
    arrImg = np.array(img, dtype=np.int64)
    nSumDiff = 0
    for j in range(arrImg.shape[0]):
        for k in range(arrImg.shape[1] - 1):
            nSumDiff += abs(arrImg[j][k] - arrImg[j][k + 1])
    for j in range(arrImg.shape[0] - 1):
        for k in range(arrImg.shape[1]):
            nSumDiff += abs(arrImg[j][k] - arrImg[j + 1][k])
    return nSumDiff

strSrcImgPath = r"C:\Users\kzzz3\Desktop\Result\InputImage"
strDstImgPath = r"C:\Users\kzzz3\Desktop\Result\OutputImage\AcEncryption"

dictResult = {}
strRoot = strDstImgPath
arrQFs = os.listdir(strDstImgPath)
for QF in tqdm(arrQFs):
    dictResult[int(QF[QF.find("=")+1:])] = {}
    strQFRoot = os.path.join(strRoot, QF)
    arrSTs = os.listdir(strQFRoot)
    for ST in tqdm(arrSTs):
        strSTRoot = os.path.join(strQFRoot, ST)

        DVs = []
        for strImageName in os.listdir(strSrcImgPath):
            strImgPath = os.path.join(strSTRoot, strImageName[:strImageName.rfind(".")] + ".jpg")
            if not os.path.exists(strImgPath):
                continue
            DVs.append(CalculateDV(strImgPath))

        dAvgDV = sum(DVs) / len(DVs)
        dictResult[int(QF[QF.find("=") + 1:])][int(ST[ST.find("=") + 1:])] = dAvgDV

#draw the result
arrQFs = [ QF for QF in list(dictResult.keys())]
arrSTs = [ ST for ST in list(dictResult.values())[0]]
arrSTs.sort()


for QF in arrQFs:
    print("QF="+str(QF))
    for ST in arrSTs:
        print(ST, round(dictResult[QF][ST]/1e5,5), r'\\')
