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

strImageName = "Baboon.bmp"
strSrcImgPath = r"C:\Users\kzzz3\Desktop\Result\InputImage"
strDstImgPath = r"C:\Users\kzzz3\Desktop\Result\OutputImage\AcEncryption"


dictResult = {}
dictPlainResult = {}
strRoot = strDstImgPath
arrQFs = os.listdir(strDstImgPath)
for QF in arrQFs:
    dictResult[int(QF[QF.find("=")+1:])] = {}
    strQFRoot = os.path.join(strRoot, QF)
    arrSTs = os.listdir(strQFRoot)
    for ST in arrSTs:
        strSTRoot = os.path.join(strQFRoot, ST)
        strImgPath = os.path.join(strSTRoot, strImageName[:strImageName.rfind(".")] + ".jpg")

        if ST == "ST=64":
            dictPlainResult[int(QF[QF.find("=")+1:])] = CaculateBPP(strImgPath)

        dictResult[int(QF[QF.find("=")+1:])][int(ST[ST.find("=")+1:])] = CaculateBPP(strImgPath)

#draw the result
arrQFs = [ QF for QF in list(dictResult.keys())]
arrSTs = [ ST for ST in list(dictResult.values())[0]]
arrSTs.sort()


for QF in arrQFs:
    print("QF="+str(QF))
    for ST in arrSTs:
        print(ST, round(dictResult[QF][ST],3), r'\\')
    for ST in arrSTs:
        print(ST, round(dictPlainResult[QF],3),r'\\')


for QF in arrQFs:
    print("QF="+str(QF))
    for ST in arrSTs:
        print(ST, round((dictResult[QF][ST]-dictPlainResult[QF])/dictPlainResult[QF]*1000,3), r"\\")

# for QF in arrQFs:
#     plt.plot(arrSTs, [dictResult[QF][ST] for ST in arrSTs], label="QF="+str(QF))
# for QF in arrQFs:
#     plt.plot(arrSTs, [dictPlainResult[QF] for ST in range(len(arrSTs))], label="QF="+str(QF)+" Plain")
# plt.grid(True)
# plt.xlim(1, max(arrSTs))
# plt.legend()
# plt.xlabel("ST")
# plt.ylabel("Compression Rate")
# plt.show()
#
# for QF in arrQFs:
#     plt.plot(arrSTs, [(dictPlainResult[QF]-dictResult[QF][ST]) for ST in arrSTs], label="QF="+str(QF))
#
# plt.grid(True)
# plt.xlim(1, max(arrSTs))
# plt.legend()
# plt.xlabel("ST")
# plt.ylabel("Expansion Rate")
# plt.show()

