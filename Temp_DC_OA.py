import os

import cv2 as cv
import gurobipy as gp
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

arrZigZag_8 = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                        [2, 4, 7, 13, 16, 26, 29, 42],
                        [3, 8, 12, 17, 25, 30, 41, 43],
                        [9, 11, 18, 24, 31, 40, 44, 53],
                        [10, 19, 23, 32, 39, 45, 52, 54],
                        [20, 22, 33, 38, 46, 51, 55, 60],
                        [21, 34, 37, 47, 50, 56, 59, 61],
                        [35, 36, 48, 49, 57, 58, 62, 63]])

arrZigZag_16 = np.array([[0, 1, 5, 6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91, 119, 120],
                         [2, 4, 7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92, 118, 121, 150],
                         [3, 8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93, 117, 122, 149, 151],
                         [9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94, 116, 123, 148, 152, 177],
                         [10, 19, 23, 32, 40, 49, 61, 70, 86, 95, 115, 124, 147, 153, 176, 178],
                         [20, 22, 33, 39, 50, 60, 71, 85, 96, 114, 125, 146, 154, 175, 179, 200],
                         [21, 34, 38, 51, 59, 72, 84, 97, 113, 126, 145, 155, 174, 180, 199, 201],
                         [35, 37, 52, 58, 73, 83, 98, 112, 127, 144, 156, 173, 181, 198, 202, 219],
                         [36, 53, 57, 74, 82, 99, 111, 128, 143, 157, 172, 182, 197, 203, 218, 220],
                         [54, 56, 75, 81, 100, 110, 129, 142, 158, 171, 183, 196, 204, 217, 221, 234],
                         [55, 76, 80, 101, 109, 130, 141, 159, 170, 184, 195, 205, 216, 222, 233, 235],
                         [77, 79, 102, 108, 131, 140, 160, 169, 185, 194, 206, 215, 223, 232, 236, 245],
                         [78, 103, 107, 132, 139, 161, 168, 186, 193, 207, 214, 224, 231, 237, 244, 246],
                         [104, 106, 133, 138, 162, 167, 187, 192, 208, 213, 225, 230, 238, 243, 247, 252],
                         [105, 134, 137, 163, 166, 188, 191, 209, 212, 226, 229, 239, 242, 248, 251, 253],
                         [135, 136, 164, 165, 189, 190, 210, 211, 227, 228, 240, 241, 249, 250, 254, 255]])

matIct_8 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                     [5, 3, 2, 1, -1, -2, -3, -5],
                     [3, 1, -1, -3, -3, -1, 1, 3],
                     [3, -1, -5, -2, 2, 5, 1, -3],
                     [1, -1, -1, 1, 1, -1, -1, 1],
                     [2, -5, 1, 3, -3, -1, 5, -2],
                     [1, -3, 3, -1, -1, 3, -3, 1],
                     [1, -2, 3, -5, 5, -3, 2, -1]])

matIct_16 = np.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                      [4, 4, 0, 2, 2, 4, 0, 0, 0, 0, -4, -2, -2, 0, -4, -4],
                      [4, 2, 2, 0, 0, -2, -2, -4, -4, -2, -2, 0, 0, 2, 2, 4],
                      [4, 2, 0, -4, 0, -4, -2, 0, 0, 2, 4, 0, 4, 0, -2, -4],
                      [4, 1, -1, -4, -4, -1, 1, 4, 4, 1, -1, -4, -4, -1, 1, 4],
                      [0, 0, -2, -4, 0, 2, 4, 4, -4, -4, -2, 0, 4, 2, 0, 0],
                      [2, 0, -4, -2, 2, 4, 0, -2, -2, 0, 4, 2, -2, -4, 0, 2],
                      [2, -4, -4, 0, 4, 0, 0, -2, 2, 0, 0, -4, 0, 4, 4, -2],
                      [2, -2, -2, 2, 2, -2, -2, 2, 2, -2, -2, 2, 2, -2, -2, 2],
                      [2, 0, 0, 4, 0, -4, 4, 2, -2, -4, 4, 0, -4, 0, 0, -2],
                      [2, -4, 0, 2, -2, 0, 4, -2, -2, 4, 0, -2, 2, 0, -4, 2],
                      [4, -4, 2, 0, -4, 2, 0, 0, 0, 0, -2, 4, 0, -2, 4, -4],
                      [1, -4, 4, -1, -1, 4, -4, 1, 1, -4, 4, -1, -1, 4, -4, 1],
                      [0, -2, 4, 0, 4, 0, -2, 4, -4, 2, 0, -4, 0, -4, 2, 0],
                      [0, -2, 2, -4, 4, -2, 2, 0, 0, 2, -2, 4, -4, 2, -2, 0],
                      [0, 0, 4, -2, 2, 0, 4, -4, 4, -4, 0, -2, 2, -4, 0, 0]])


def ComputeContributionMatrix():
    a = np.zeros([8, 8, 8, 8], dtype=np.float64)
    c = np.zeros(8, dtype=np.float64)
    c[0], c[1:] = (1 / 8) ** 0.5, (2 / 8) ** 0.5
    c = c[:, None] * c[None, :]
    i = np.arange(8).astype(np.float64)
    cos = np.cos((i[:, None] + 0.5) * i[None, :] * np.pi / 8)
    a = c[None, None] * cos[:, None, :, None] * cos[None, :, None, :]
    return a


def PreProcess(strInputImagePath: str,
               nRegionSize: int,
               nStartIndex: int,
               nSpan: int):
    # ICT and Zigzag
    matIct = np.zeros((nRegionSize // 8, nRegionSize // 8), dtype=np.float64)
    matZigzag = np.zeros((nRegionSize // 8, nRegionSize // 8), dtype=np.float64)
    if nRegionSize == 64:
        matIct = matIct_8
        matZigzag = arrZigZag_8
    elif nRegionSize == 128:
        matIct = matIct_16
        matZigzag = arrZigZag_16

    arrImgData = cv.imread(strInputImagePath, cv.IMREAD_GRAYSCALE).astype(np.float64) - 128
    nHeight, nWidth = arrImgData.shape

    arrDCT = np.zeros_like(arrImgData, dtype=np.float64)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            arrDCT[nRow:nRow + 8, nCol:nCol + 8] = cv.dct(arrImgData[nRow:nRow + 8, nCol:nCol + 8])

    arrMask = np.zeros_like(arrDCT, dtype=bool)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrZigZag_8[i, j] >= 1 and arrZigZag_8[i, j] <= 40:
                        arrMask[nRow + i, nCol + j] = True

    arrDcDct = np.zeros((nHeight // 8, nWidth // 8), dtype=np.float64)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            arrDcDct[nRow // 8, nCol // 8] = arrDCT[nRow, nCol]

    nDcHeight, nDcWidth = arrDcDct.shape
    for nRow in range(0, nDcHeight, nRegionSize // 8):
        for nCol in range(0, nDcWidth, nRegionSize // 8):
            arrDcDct[nRow:nRow + nRegionSize // 8, nCol:nCol + nRegionSize // 8] = matIct.T @ arrDcDct[nRow:nRow + nRegionSize // 8, nCol:nCol + nRegionSize // 8] @ matIct

    arrDcMask = np.zeros_like(arrDcDct, dtype=bool)
    for nRow in range(0, nDcHeight, nRegionSize // 8):
        for nCol in range(0, nDcWidth, nRegionSize // 8):
            for i in range(nRegionSize // 8):
                for j in range(nRegionSize // 8):
                    if matZigzag[i, j] >= nStartIndex - nSpan + 1 and matZigzag[i, j] <= nStartIndex:
                        arrDcMask[nRow + i, nCol + j] = True

    return arrDCT, arrMask, arrDcDct, arrDcMask


def OptimizationBasedAttack(
        nRegionSize: int,
        arrDct: np.ndarray,
        arrAcMask: np.ndarray,
        arrDcDct: np.ndarray,
        arrDcAcMask: np.ndarray
) -> np.ndarray:
    nHeight, nWidth = arrDct.shape

    # ICT and Zigzag
    matIct = np.zeros((nRegionSize // 8, nRegionSize // 8), dtype=np.float64)
    if nRegionSize == 64:
        matIct = matIct_8
    elif nRegionSize == 128:
        matIct = matIct_16

    # 计算像素贡献矩阵并分离已知和未知部分
    arrContributionMatrix = ComputeContributionMatrix().reshape(8 * 8, 8 * 8)

    # 构造模型
    modelOptimization = gp.Model("Optimization-based Attack")
    modelOptimization.setParam('Threads', os.cpu_count())

    # 构造变量
    arrDiffValueHorizontal = modelOptimization.addMVar((nHeight, nWidth - 1), vtype=gp.GRB.CONTINUOUS)
    arrDiffValueVertical = modelOptimization.addMVar((nHeight - 1, nWidth), vtype=gp.GRB.CONTINUOUS)
    arrVarMissingAc = modelOptimization.addMVar(arrAcMask.sum(), vtype=gp.GRB.CONTINUOUS, name="MissingAc")
    arrVarMissingDcAc = modelOptimization.addMVar(arrDcAcMask.sum(), vtype=gp.GRB.CONTINUOUS, name="MissingDcAc")
    arrVarMissingAc = np.asarray(arrVarMissingAc.tolist())
    arrVarMissingDcAc = np.asarray(arrVarMissingDcAc.tolist())

    arrFullPixel = np.zeros((nHeight, nWidth), dtype=gp.LinExpr)

    # 构造AC表达式
    nIndex = 0
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrAcMask[nRow + i, nCol + j]:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                arrVarMissingAc[np.newaxis, nIndex:nIndex + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)
                        nIndex += 1
                    else:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                arrDct[nRow + i:nRow + i + 1, nCol + j:nCol + j + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)

    # 构造DCAC表达式
    nIndex = 0
    nDcHeight, nDcWidth = arrDcDct.shape
    arrFullDc = np.zeros(arrDcDct.shape, dtype=gp.LinExpr)
    arrDiag = np.linalg.inv(matIct @ matIct.T)
    for nRow in range(0, nDcHeight, nRegionSize // 8):
        for nCol in range(0, nDcWidth, nRegionSize // 8):
            arrFullDc[nRow:nRow + nRegionSize // 8, nCol:nCol + nRegionSize // 8] += arrDiag @ matIct @ arrDcDct[nRow:nRow + nRegionSize // 8,
                                                                                                        nCol:nCol + nRegionSize // 8] @ matIct.T @ arrDiag

            arrRecoveredDcAc = np.zeros((nRegionSize // 8, nRegionSize // 8), dtype=gp.LinExpr)
            for i in range(nRegionSize // 8):
                for j in range(nRegionSize // 8):
                    if arrDcAcMask[nRow + i, nCol + j]:
                        arrRecoveredDcAc[i, j] = arrVarMissingDcAc[nIndex]
                        nIndex += 1

            arrFullDc[nRow:nRow + nRegionSize // 8, nCol:nCol + nRegionSize // 8] += arrDiag @ matIct @ arrRecoveredDcAc @ matIct.T @ arrDiag

    # 将DcDct还原至Dct
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            arrFullPixel[nRow, nCol] = arrFullDc[nRow // 8, nCol // 8]

    # 构造约束
    modelOptimization.addConstr(arrDiffValueHorizontal >= arrFullPixel[:, :-1] - arrFullPixel[:, 1:])
    modelOptimization.addConstr(arrDiffValueHorizontal >= arrFullPixel[:, 1:] - arrFullPixel[:, :-1])
    modelOptimization.addConstr(arrDiffValueVertical >= arrFullPixel[:-1, :] - arrFullPixel[1:, :])
    modelOptimization.addConstr(arrDiffValueVertical >= arrFullPixel[1:, :] - arrFullPixel[:-1, :])

    # 构造目标函数
    modelOptimization.setObjective(arrDiffValueHorizontal.sum() + arrDiffValueVertical.sum(), gp.GRB.MINIMIZE)

    modelOptimization.update()
    modelOptimization.optimize()

    arrRecoveredDcDct = np.zeros_like(arrVarMissingDcAc, dtype=np.float64)
    nIndex = 0
    for i in arrVarMissingDcAc:
        arrRecoveredDcDct[nIndex] = i.X
        nIndex += 1

    nIndex = 0
    arrFullDc = np.zeros_like(arrDcDct, dtype=np.float64)
    for nRow in range(0, nDcHeight, nRegionSize // 8):
        for nCol in range(0, nDcWidth, nRegionSize // 8):
            arrFullDc[nRow:nRow + nRegionSize // 8, nCol:nCol + nRegionSize // 8] += arrDiag @ matIct @ arrDcDct[nRow:nRow + nRegionSize // 8,
                                                                                                        nCol:nCol + nRegionSize // 8] @ matIct.T @ arrDiag

            arrRecoveredDcAc = np.zeros((nRegionSize // 8, nRegionSize // 8), dtype=np.float64)
            for i in range(nRegionSize // 8):
                for j in range(nRegionSize // 8):
                    if arrDcAcMask[nRow + i, nCol + j]:
                        arrRecoveredDcAc[i, j] = arrRecoveredDcDct[nIndex]
                        nIndex += 1

            arrFullDc[nRow:nRow + nRegionSize // 8, nCol:nCol + nRegionSize // 8] += arrDiag @ matIct @ arrRecoveredDcAc @ matIct.T @ arrDiag

    # 将DcDct还原至Dct
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            arrDct[nRow, nCol] = arrFullDc[nRow // 8, nCol // 8]

    arrRecoveredDct = np.zeros_like(arrVarMissingAc, dtype=np.float64)
    nIndex = 0
    for i in arrVarMissingAc:
        arrRecoveredDct[nIndex] = i.X
        nIndex += 1

    nIndex = 0
    arrFullPixel = np.zeros((nHeight, nWidth), dtype=np.float64)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrAcMask[nRow + i, nCol + j]:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                arrRecoveredDct[np.newaxis, nIndex:nIndex + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)
                        nIndex += 1
                    else:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                arrDct[nRow + i:nRow + i + 1, nCol + j:nCol + j + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)
    return (arrFullPixel + 128).clip(0, 255)
# AC Encryption
strSrcImageName = "zelda.bmp"
strDstImageName = "zelda_ST=20.jpg"
strSrcImgPath = r"C:\Users\kzzz3\Desktop\Result\InputImage"
strDstImgPath = r"C:\Users\kzzz3\Desktop\Fig\SecurityAnalysis"

strPlainImagePath = os.path.join(strSrcImgPath, strSrcImageName)
strCipherImagePath = os.path.join(strDstImgPath, strDstImageName)
strOutputImagePath = os.path.join(strDstImgPath, strDstImageName[:strDstImageName.rfind(".")] + "_OA.bmp")

if not os.path.exists(strDstImgPath):
    # 针对CipherImage 进行优化攻击
    arrDCT, arrMask, arrDcDct, arrDcMask = PreProcess(strCipherImagePath, 128, 72, 50)
    arrRecoveredImage = OptimizationBasedAttack(128, arrDCT, arrMask, arrDcDct, arrDcMask)
    cv.imwrite(strOutputImagePath, arrRecoveredImage)

print(strSrcImageName)

# 计算图像psnr 和 ssim
imgPlain = io.imread(strPlainImagePath)
imgCipher = io.imread(strCipherImagePath)
imgOutput = io.imread(strOutputImagePath)
psnr = peak_signal_noise_ratio(imgCipher, imgPlain)
ssim = structural_similarity(imgCipher, imgPlain)

print("PSNR:",psnr)
print("SSIM:",ssim)

psnr = peak_signal_noise_ratio(imgOutput, imgPlain)
ssim = structural_similarity(imgOutput, imgPlain)

print("PSNR:",psnr)
print("SSIM:",ssim)