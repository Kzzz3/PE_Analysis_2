import os

from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# AC Encryption
strImageName = "Baboon.bmp"
strSrcImgPath = r"C:\Users\kzzz3\Desktop\Result\InputImage"
strDstImgPath = r"C:\Users\kzzz3\Desktop\Result\OutputImage\DcEncryption"

dictPSNR = {}
dictSSIM = {}
strRoot = strDstImgPath

for RegionSize in os.listdir(strDstImgPath):
    dictPSNR[int(RegionSize[RegionSize.find("=") + 1:])] = {}
    dictSSIM[int(RegionSize[RegionSize.find("=") + 1:])] = {}
    strRegionSizeRoot = os.path.join(strRoot, RegionSize)

    arrQFs = os.listdir(strRegionSizeRoot)
    for QF in arrQFs:
        dictPSNR[int(RegionSize[RegionSize.find("=") + 1:])][int(QF[QF.find("=") + 1:])] = {}
        dictSSIM[int(RegionSize[RegionSize.find("=") + 1:])][int(QF[QF.find("=") + 1:])] = {}
        strQFRoot = os.path.join(strRegionSizeRoot, QF)
        strSpanRoot = os.path.join(strQFRoot, os.listdir(strQFRoot)[0])

        arrSTs = os.listdir(strSpanRoot)
        for ST in arrSTs:
            strSTRoot = os.path.join(strSpanRoot, ST)

            strPlainImagePath = os.path.join(strSrcImgPath, strImageName)
            strCipherImagePath = os.path.join(strSTRoot, strImageName[:strImageName.rfind(".")] + ".jpg")

            # 计算图像psnr 和 ssim
            imgSrc = io.imread(strPlainImagePath)
            imgDst = io.imread(strCipherImagePath)
            psnr = peak_signal_noise_ratio(imgDst, imgSrc)
            ssim = structural_similarity(imgDst, imgSrc)

            dictPSNR[int(RegionSize[RegionSize.find("=") + 1:])][int(QF[QF.find("=") + 1:])][int(ST[ST.find("=") + 1:])] = psnr
            dictSSIM[int(RegionSize[RegionSize.find("=") + 1:])][int(QF[QF.find("=") + 1:])][int(ST[ST.find("=") + 1:])] = ssim

#draw the result
arrRegionSizes = [ RegionSize for RegionSize in list(dictPSNR.keys())]
arrQFs = { RegionSize:[ QF for QF in list(dictPSNR[RegionSize].keys())] for RegionSize in arrRegionSizes }
arrSTs = { RegionSize:{ QF:[ ST for ST in list(dictPSNR[RegionSize][QF].keys())] for QF in arrQFs[RegionSize]} for RegionSize in arrRegionSizes }

arrRegionSizes.sort()
arrQFs = {key: sorted(value) for key, value in arrQFs.items()}
arrSTs = {RegionSize: {key: sorted(value) for key, value in value.items()} for RegionSize, value in arrSTs.items()}

for RegionSize in arrRegionSizes:
    print("RegionSize="+str(RegionSize))
    for QF in arrQFs[RegionSize]:
        print("QF="+str(QF))
        print("PSNR:")
        for ST in arrSTs[RegionSize][QF]:
            print(ST, round(dictPSNR[RegionSize][QF][ST],3),r"\\")
        print("SSIM:")
        for ST in arrSTs[RegionSize][QF]:
            print(ST, round(dictSSIM[RegionSize][QF][ST],3),r"\\")