import os

from PIL import Image


def CaculateBPP(strImagePath):
    if not os.path.exists(strImagePath):
        return "File not found"

    with Image.open(strImagePath) as img:
        file_size = os.path.getsize(strImagePath) * 8  # Convert to bits
        width, height = img.size
        total_pixels = width * height
        bpp = file_size / total_pixels
        return bpp


strSrcImgPath = r"C:\Users\kzzz3\Desktop\Result\InputImage"
strDstImgPath = r"C:\Users\kzzz3\Desktop\Result\OutputImage\DcEncryption"

dictResult = {}
strRoot = strDstImgPath
for RegionSize in os.listdir(strDstImgPath):
    dictResult[int(RegionSize[RegionSize.find("=") + 1:])] = {}
    strRegionSizeRoot = os.path.join(strRoot, RegionSize)

    arrQFs = os.listdir(strRegionSizeRoot)
    for QF in arrQFs:
        dictResult[int(RegionSize[RegionSize.find("=") + 1:])][int(QF[QF.find("=") + 1:])] = {}
        strQFRoot = os.path.join(strRegionSizeRoot, QF)
        strSpanRoot = os.path.join(strQFRoot, os.listdir(strQFRoot)[0])

        arrSTs = os.listdir(strSpanRoot)
        for ST in arrSTs:
            strSTRoot = os.path.join(strSpanRoot, ST)

            BPPs = []
            for strImageName in os.listdir(strSrcImgPath):
                strImgPath = os.path.join(strSTRoot, strImageName[:strImageName.rfind(".")] + ".jpg")
                if not os.path.exists(strImgPath):
                    continue
                BPPs.append(CaculateBPP(strImgPath))
            dictResult[int(RegionSize[RegionSize.find("=") + 1:])][int(QF[QF.find("=") + 1:])][int(ST[ST.find("=") + 1:])] = sum(BPPs) / len(BPPs)

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
            print(ST, round(dictResult[RegionSize][QF][ST],3),r"\\")

#2.409 1.300 0.938 0.664


# arrQFs = os.listdir(strDstImgPath)
# for QF in arrQFs:
#     dictResult[int(QF[QF.find("=")+1:])] = {}
#     strQFRoot = os.path.join(strRoot, QF)
#     strSpanRoot = os.path.join(strQFRoot, os.listdir(strQFRoot)[0])
#
#     arrSTs = os.listdir(strSpanRoot)
#     for ST in arrSTs:
#         strSTRoot = os.path.join(strSpanRoot, ST)
#         strImgPath = os.path.join(strSTRoot, strImageName[:strImageName.rfind(".")] + ".jpg")
#
#         dictResult[int(QF[QF.find("=")+1:])][int(ST[ST.find("=")+1:])] = CaculateBPP(strImgPath)

#draw the result
# arrQFs = [ QF for QF in list(dictResult.keys())]
# arrSTs = { QF:[ ST for ST in list(dictResult[QF].keys())] for QF in arrQFs }
# arrQFs.sort()
# arrSTs = {key: sorted(value) for key, value in arrSTs.items()}


# for QF in arrQFs:
#     print("QF="+str(QF))
#     for ST in arrSTs[QF]:
#         print(ST, round(dictResult[QF][ST],3), r'\\')



