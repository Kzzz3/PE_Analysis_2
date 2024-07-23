import subprocess

# 第一个Python脚本的文件名
script1 = "OptimizationBasedAttack_AC.py"

# 第二个Python脚本的文件名
script2 = "OptimizationBasedAttack_DC.py"

# 运行第一个Python脚本
subprocess.run(["python", script1])

# 运行第二个Python脚本
subprocess.run(["python", script2])
