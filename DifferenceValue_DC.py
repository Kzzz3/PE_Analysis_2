import os

import numpy as np
from PIL import Image
from tqdm import tqdm


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
strDstImgPath = r"C:\Users\kzzz3\Desktop\Result\OutputImage\DcEncryption"

dictResult = {}
strRoot = strDstImgPath
for RegionSize in tqdm(os.listdir(strDstImgPath)):
    dictResult[int(RegionSize[RegionSize.find("=") + 1:])] = {}
    strRegionSizeRoot = os.path.join(strRoot, RegionSize)

    arrQFs = os.listdir(strRegionSizeRoot)
    for QF in tqdm(arrQFs):
        dictResult[int(RegionSize[RegionSize.find("=") + 1:])][int(QF[QF.find("=") + 1:])] = {}
        strQFRoot = os.path.join(strRegionSizeRoot, QF)
        strSpanRoot = os.path.join(strQFRoot, os.listdir(strQFRoot)[0])

        arrSTs = os.listdir(strSpanRoot)
        for ST in tqdm(arrSTs):
            strSTRoot = os.path.join(strSpanRoot, ST)

            DVs = []
            for strImageName in os.listdir(strSrcImgPath):
                strImgPath = os.path.join(strSTRoot, strImageName[:strImageName.rfind(".")] + ".jpg")
                if not os.path.exists(strImgPath):
                    continue
                DVs.append(CalculateDV(strImgPath))
            dictResult[int(RegionSize[RegionSize.find("=") + 1:])][int(QF[QF.find("=") + 1:])][int(ST[ST.find("=") + 1:])] = sum(DVs) / len(DVs)

#draw the result
arrRegionSizes = [ RegionSize for RegionSize in list(dictResult.keys())]
arrQFs = { RegionSize:[ QF for QF in list(dictResult[RegionSize].keys())] for RegionSize in arrRegionSizes }
arrSTs = { RegionSize:{ QF:[ ST for ST in list(dictResult[RegionSize][QF].keys())] for QF in arrQFs[RegionSize]} for RegionSize in arrRegionSizes }

arrRegionSizes.sort()
arrQFs = {key: sorted(value) for key, value in arrQFs.items()}
arrSTs = {RegionSize: {key: sorted(value) for key, value in value.items()} for RegionSize, value in arrSTs.items()}

for RegionSize in arrRegionSizes:
    print("RegionSize="+str(RegionSize))
    for QF in arrQFs[RegionSize]:
        print("QF="+str(QF))
        for ST in arrSTs[RegionSize][QF]:
            print(ST, round(dictResult[RegionSize][QF][ST]/1e5,5),r"\\")




