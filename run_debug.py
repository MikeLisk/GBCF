import subprocess
import sys

# 运行main.py并捕获所有输出
process = subprocess.Popen(
    [sys.executable, "main.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True
)

# 读取并打印输出
epoch_count = 0
for line in process.stdout:
    print(line, end='')
    # 在第一个epoch之后终止
    if "Epoch   1/50" in line:
        epoch_count += 1
        if epoch_count >= 1:
            # 等待几秒钟以捕获完整输出
            import time
            time.sleep(2)
            process.terminate()
            break

# 等待进程结束
try:
    process.wait(timeout=5)
except subprocess.TimeoutExpired:
    process.kill()
