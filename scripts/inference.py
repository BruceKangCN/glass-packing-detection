# %%
#
# 1. 导入所需的包并进行全局配置
#

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from ultralytics import YOLO

try:
    # 检测 IPython 环境以判断是否处于笔记本模式中
    get_ipython() # type: ignore
    # 若是，则使用 widget 渲染后端
    mpl.use("widget")
except NameError:
    # 若未处于笔记本环境中，则使用 QtAgg 渲染后端
    mpl.use("QtAgg")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if len(sys.argv) < 2:
    sys.exit(1)

# %%
#
# 1. 加载模型权重
#

if len(sys.argv) >= 3:
    model_path = sys.argv[2]
else:
    model_path = PROJECT_ROOT / "weights" / "best.pt"
model = YOLO(model_path)

# %%
#
# 2. 进行推理
#

results = model.predict(sys.argv[1], conf=0.6)
result = results[0] # type: ignore
boxes = result.boxes # type: ignore

if boxes is None or len(boxes) == 0:
    print("No object detected.")
else:
    print("Detections (xywh):")
    print(boxes.xywh)
    print("Detections (xywhn):")
    print(boxes.xywhn)

# %%
#
# 3. 可视化
#

annotated_img = result.plot()[:, :, ::-1] # type: ignore

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(annotated_img)

plt.show()
