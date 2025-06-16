# README

## 本地系統環境

| 元件     | 版本                      |
| ------ | ----------------------- |
| 作業系統   | Windows 11              |
| CUDA   | 12.1 (cuda\_12.1.r12.1) |
| Python | 3.11.5                  |

> **建議**：確保 GPU 驅動與 CUDA 版本相容，否則 LLM 可能落到 CPU 執行。

---

## 1 安裝指引

### 1.1 安裝 Anaconda／Miniconda

1. 從官方網站下載對應平台（64‑bit、Python 3.11）版本。
2. 安裝完成後在終端驗證：

   ```bash
   conda --version
   ```

### 1.2 建立虛擬環境

```bash
conda env create -f environment.yml   # 依 YAML 建立環境
conda activate <env_name>             # 啟用（名稱以 YAML 中的 name 為準）
```

---

## 2 模型說明

| 類型  | Repository                                  |
| --- | ------------------------------------------- |
| ASR | **openai/whisper-large-v3-turbo**           |
| LLM | **suayptalha/DeepSeek-R1-Distill-Llama-3B** |

---

## 3 任務流程

### Task 1 — 取得轉錄結果 (雲端執行)

1. 打開 **Google Colab**，上傳並開啟 `hf2ct-colab.ipynb`。
2. 依 Notebook 指示 **手動下載** ASR 模型權重並放置於指定路徑。
3. 全部執行完畢後，Colab 會輸出：

   ```
   task1_answer_timestamps.json   # 轉錄結果 + timestamps
   ```

### Task 2 — LLM訓練與推論 (本地執行)

| 步驟  | Notebook                                           | 輸出檔                                    | 摘要                                              |
| --- | -------------------------------------------------- | -------------------------------------- | ----------------------------------------------- |
| 1   | `時間戳轉轉錄文本.ipynb`                                   | `task1_answer`                         | 依 timestamps 切割原始 JSON                          |
| 2‑a | `標點符號處理_STEP1.ipynb`                               | `task1_answer_timestamps_cleaned.json` | 初步清洗標點                                          |
| 2‑b | **手動**：將上一步輸出檔重新命名為 `task1_answer_timestamps.json` |                                        | 覆寫舊檔以供下一步使用                                     |
| 2‑c | `標點符號後處理日期_STEP2.ipynb`                            | —                                      | 二次標點／日期修正                                       |
| 3   | `AICUP2025_Phase1_Dataset.ipynb`                   | —                                      | 從「Task2」註記開始執行；若僅推論，可直接跳至 *Load finetune model* |

> **提示**：若不需微調，可在載入必要套件後，直接跳到 Notebook 裡的 `Load finetune model` 區塊。此外，`AICUP2025_Phase1_Dataset.ipynb`不需要跑Task1部分，因為已經在`hf2ct-colab.ipynb`做過了。


---

## 4 參考資源

* [Anaconda 官方下載](https://www.anaconda.com/download)
* [CUDA Toolkit 12.1 Release Notes](https://docs.nvidia.com/cuda/)
* [Whisper (OpenAI) GitHub](https://github.com/openai/whisper)
* [DeepSeek Models](https://huggingface.co/suayptalha/DeepSeek-R1-Distill-Llama-3B)
