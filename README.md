# 機器學習 Project 競賽二

隊伍：TEAM_8740  
隊員：葉騏緯  
Public Leaderboard：0.94526 / Rank 121  

<img src="https://meee.com.tw/5coNCNm.png" alt="project2 leaderboard">

## 資料處理與程式環境

### 資料與切分

- 原始資料：50 位病人的 2D 影像切片與 YOLO 版標註（單一類別 `aortic_valve`）。
- 目前切分：train=patient0001–0040（2168 張）、val=patient0041–0050（619 張）、test=patient0051–0100（16620 張，無標註），皆置於 `datasets/{train,val,test}/{images,labels}`。

### 目錄與環境

- 主要腳本：`train.sh`（啟動訓練）、`infer.sh`（批次推論）、`train.py`/`infer.py`（核心邏輯）。
- 依賴管理：`uv` 優先；若系統無 `uv`，腳本會改用 `python3`。Python 版本要求 `>=3.12`，套件列於 `pyproject.toml`（Ultralytics YOLO 8.3.226）。
- 權重與輸出：訓練結果存到 `runs/train/<run_name>/`，預設 `run_name=project_2_res`；推論輸出 txt 置於 `predict_txt/`。
- 硬體實測：Ubuntu 24.04.3 + NVIDIA RTX A4500 20GB，CUDA 12.2。

`pyproject.toml`:

```toml
[project]
name = "ml2025f-project-2"
version = "0.1.0"
description = "AI CUP 2025秋季賽 - 電腦斷層心臟肌肉影像分割競賽 II - 主動脈瓣物件偵測"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "huggingface-hub>=1.1.4",
    "python-dotenv>=1.2.1",
    "ultralytics>=8.3.226",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
fixable = ["ALL"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "lf"

```

## 模型設計與訓練策略

[專案程式碼](https://github.com/chiwei085/ml2025f_project_2.git)

- 模型：Ultralytics YOLOv12（預訓練權重 `yolo12m.pt`）。
- 任務：單類別小物件偵測（主動脈瓣），以 640×640 輸入、最多輸出 1 個 bbox。
- 超參數（`train_conf.yaml` 可直接調整）：

| hyper    | value           |
| :------- | :-------------- |
| weights  | yolo12m.pt      |
| epochs   | 50              |
| imgsz    | 640             |
| batch    | 16              |
| patience | 40              |
| device   | 0               |
| project  | runs/train      |
| name     | project_2_res   |
| max_det  | 1（推論端設定） |

### 訓練與推論流程

- 訓練：`train.sh` 會讀取 `training.yaml`（資料路徑與類別名）與 `train_conf.yaml`（超參數），並呼叫 Ultralytics YOLO


- 推論：`infer.sh` 預設載入 `runs/train/project_2_res2/weights/best.pt`，輸出到 `predict_txt/project_2_res.txt`。可在命令列覆寫 `--weights`、`--source`、`--name`、`--conf`、`--iou`、`--max-det`：

推論會同時在 `runs/detect/<auto_run>/` 產生 Ultralytics 預設的圖檔，txt 則以 `predict_txt/<name>.txt` 儲存，檔名預設為權重所屬 run 名稱。

`train.sh`:

```sh
#!/usr/bin/env bash

DATA_PATH=${DATA_PATH:-training.yaml}
CONFIG_PATH=${CONFIG_PATH:-train_conf.yaml}

if command -v uv >/dev/null 2>&1; then
  RUNNER="uv run"
else
  RUNNER="python3"
fi

$RUNNER train.py --data "$DATA_PATH" --config "$CONFIG_PATH"
```

`infer.sh`:

```sh
#!/usr/bin/env bash

if command -v uv >/dev/null 2>&1; then
  RUNNER="uv run"
else
  RUNNER="python3"
fi

$RUNNER infer.py \
  --weights runs/train/project_2_res2/weights/best.pt \
  --name project_2_res \
```

## 分析與結論

本次競賽聚焦於醫學 CT 影像中單類別的小尺寸器官偵測，相較一般自然影像資料，其特性包含：背景高度相似、對比不明顯、物件位置變化受患者體位限制、且標註僅有 1 個目標。由於資料量相對有限，模型選擇需在**容量、泛化能力與過擬合風險**之間平衡。實驗結果顯示：由 yolo12n → yolo12s → yolo12m 的漸進式擴模確實提升性能，但當模型容量超過資料可支撐範圍時，增加訓練 epochs 反而會導致過擬合，尤以 YOLOv12m 在長訓練下 mAP 下降最為明顯。

此外，嘗試將優化器從 AdamW 切換至 SGD 後，發現雖然長 epochs 下的 loss 曲線更穩定，但並未帶來顯著性能突破，推測原因為醫療單目標偵測任務對梯度震盪與泛化調控較不敏感，且模型錯誤大多集中於極低對比切片而非典型學習不足。因此，後續並未再提高模型規模或加長訓練回合數。如此才能有效改善召回率並提升 leaderboard 表現。

## 使用的外部資源與參考

Ultralytics. (2024). *YOLOv12 documentation*. Retrieved from https://docs.ultralytics.com/models/yolo12/
