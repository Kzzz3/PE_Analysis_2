import os

import cv2 as cv
import gurobipy as gp
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

arrZigZag = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                      [2, 4, 7, 13, 16, 26, 29, 42],
                      [3, 8, 12, 17, 25, 30, 41, 43],
                      [9, 11, 18, 24, 31, 40, 44, 53],
                      [10, 19, 23, 32, 39, 45, 52, 54],
                      [20, 22, 33, 38, 46, 51, 55, 60],
                      [21, 34, 37, 47, 50, 56, 59, 61],
                      [35, 36, 48, 49, 57, 58, 62, 63]])

def ZigzagScan(arrMatrix: np.ndarray) -> np.ndarray:
    nHeight, nWidth = arrMatrix.shape[:2]
    arrZigzagScaned = np.zeros((nHeight, nWidth, 8 * 8), dtype=np.float64)
    for nRow in range(nHeight):
        for nCol in range(nWidth):
            for i in range(8):
                for j in range(8):
                    arrZigzagScaned[nRow, nCol, arrZigZag[i, j]] = arrMatrix[nRow, nCol, i, j]
    return arrZigzagScaned


def ComputeContributionMatrix():
    a = np.zeros([8, 8, 8, 8], dtype=np.float64)
    c = np.zeros(8, dtype=np.float64)
    c[0], c[1:] = (1 / 8) ** 0.5, (2 / 8) ** 0.5
    c = c[:, None] * c[None, :]
    i = np.arange(8).astype(np.float64)
    cos = np.cos((i[:, None] + 0.5) * i[None, :] * np.pi / 8)
    a = c[None, None] * cos[:, None, :, None] * cos[None, :, None, :]
    return a

def PreProcess(strInputImagePath,nST):
    arrImgData = cv.imread(strInputImagePath, cv.IMREAD_GRAYSCALE).astype(np.float32) - 128
    nHeight, nWidth = arrImgData.shape

    arrDCT = np.zeros_like(arrImgData, dtype=np.float32)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            arrDCT[nRow:nRow + 8, nCol:nCol + 8] = cv.dct(arrImgData[nRow:nRow + 8, nCol:nCol + 8])

    arrMask = np.zeros_like(arrDCT, dtype=bool)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrZigZag[i, j] >= nST:
                        arrMask[nRow + i, nCol + j] = True
    return arrDCT,arrMask

def OptimizationBasedAttack(
        arrDct: np.ndarray,
        arrMask: np.ndarray
) -> np.ndarray:
    nHeight, nWidth = arrDct.shape

    # 计算像素贡献矩阵并分离已知和未知部分
    arrContributionMatrix = ComputeContributionMatrix().reshape(8 * 8, 8 * 8)

    # 构造模型
    modelOptimization = gp.Model("Optimization-based Attack")

    # 构造变量
    arrDiffValueHorizontal = modelOptimization.addMVar((nHeight, nWidth - 1), vtype=gp.GRB.CONTINUOUS)
    arrDiffValueVertical = modelOptimization.addMVar((nHeight - 1, nWidth), vtype=gp.GRB.CONTINUOUS)
    arrVarMissingAc = modelOptimization.addMVar(arrMask.sum(), vtype=gp.GRB.CONTINUOUS, name="MissingAc")
    arrVarMissingAc = np.asarray(arrVarMissingAc.tolist())

    arrFullPixel = np.zeros((nHeight, nWidth), dtype=gp.LinExpr)

    nIndex = 0
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrMask[nRow + i, nCol + j]:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                    arrVarMissingAc[np.newaxis, nIndex:nIndex + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)
                        nIndex += 1
                    else:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                    arrDct[nRow + i:nRow + i + 1, nCol + j:nCol + j + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)

    # 构造约束
    modelOptimization.addConstr(arrDiffValueHorizontal >= arrFullPixel[:, :-1] - arrFullPixel[:, 1:])
    modelOptimization.addConstr(arrDiffValueHorizontal >= arrFullPixel[:, 1:] - arrFullPixel[:, :-1])
    modelOptimization.addConstr(arrDiffValueVertical >= arrFullPixel[:-1, :] - arrFullPixel[1:, :])
    modelOptimization.addConstr(arrDiffValueVertical >= arrFullPixel[1:, :] - arrFullPixel[:-1, :])

    # 构造目标函数
    modelOptimization.setObjective(arrDiffValueHorizontal.sum() + arrDiffValueVertical.sum(), gp.GRB.MINIMIZE)

    modelOptimization.update()
    modelOptimization.optimize()

    arrRecoveredDct = np.zeros_like(arrVarMissingAc, dtype=np.float32)
    nIndex = 0
    for i in arrVarMissingAc:
        arrRecoveredDct[nIndex] = i.X
        nIndex += 1

    nIndex = 0
    arrFullPixel = np.zeros((nHeight, nWidth), dtype=np.float32)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrMask[nRow + i, nCol + j]:
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

#针对CipherImage 进行优化攻击
if not os.path.exists(strOutputImagePath):
    arrDCT,arrMask = PreProcess(strCipherImagePath,20)
    arrRecoveredImage = OptimizationBasedAttack(arrDCT,arrMask)
    cv.imwrite(strOutputImagePath, arrRecoveredImage)


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