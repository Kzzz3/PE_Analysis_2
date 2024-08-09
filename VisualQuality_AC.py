import os

from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# AC Encryption
strImageName = "barbara_gray.bmp"
strSrcImgPath = r"C:\Users\kzzz3\Desktop\Result\InputImage"
strDstImgPath = r"C:\Users\kzzz3\Desktop\Result\OutputImage\AcEncryption"

dictPSNR = {}
dictSSIM = {}
strRoot = strDstImgPath
arrQFs = os.listdir(strDstImgPath)
for QF in arrQFs:
    dictPSNR[int(QF[QF.find("=") + 1:])] = {}
    dictSSIM[int(QF[QF.find("=") + 1:])] = {}
    strQFRoot = os.path.join(strRoot, QF)
    arrSTs = os.listdir(strQFRoot)
    for ST in arrSTs:
        strSTRoot = os.path.join(strQFRoot, ST)

        strPlainImagePath = os.path.join(strSrcImgPath, strImageName)
        strCipherImagePath = os.path.join(strSTRoot, strImageName[:strImageName.rfind(".")] + ".jpg")

        # 计算图像psnr 和 ssim
        imgSrc = io.imread(strPlainImagePath)
        imgDst = io.imread(strCipherImagePath)
        psnr = peak_signal_noise_ratio(imgDst, imgSrc)
        ssim = structural_similarity(imgDst, imgSrc)

        dictPSNR[int(QF[QF.find("=")+1:])][int(ST[ST.find("=")+1:])] = psnr
        dictSSIM[int(QF[QF.find("=")+1:])][int(ST[ST.find("=")+1:])] = ssim

#draw the result
arrQFs = [ QF for QF in list(dictPSNR.keys())]
arrSTs = [ 64-ST for ST in list(dictPSNR.values())[0]]
arrSTs.sort()

for QF in arrQFs:
    print("QF="+str(QF))
    # print("PSNR:")
    # for ST in arrSTs:
    #     print(ST, round(dictPSNR[QF][64-ST],3),r"\\")
    print("SSIM:")
    for ST in arrSTs:
        print(ST, round(dictSSIM[QF][64-ST],3),r"\\")