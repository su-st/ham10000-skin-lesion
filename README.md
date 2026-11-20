# 以遷移學習分類皮膚鏡病灶：類別不平衡與資料洩漏分析

國立屏東科技大學資訊管理系「數位醫學與影像導論」期末專題。以 HAM10000 進行七類皮膚鏡病灶分類，比較傳統機器學習與 EfficientNet-B0 遷移學習，並分析類別不平衡處理與資料切分方式（資料洩漏）對評估結果的影響，最後實作一個網頁原型。

**原型網址**：https://huggingface.co/spaces/JUNG1127/skin-lesion-demo

> 本專題為課程學術研究，模型與原型僅供課程展示，不能取代醫師診斷。

## 研究問題

- **RQ1**：遷移學習 CNN 的 macro-F1 是否優於傳統機器學習 baseline？
- **RQ2**：類別權重、過採樣與資料增強，對少數類別（特別是黑色素瘤）召回率的影響為何？
- **RQ3**：依影像隨機切分與依病灶切分，模型效能的差異為何？

## 實驗

| 實驗 | 內容 | RQ |
|---|---|---|
| E1 | SVM、Random Forest、EfficientNet-B0，依病灶切分 | RQ1 |
| E2 | EfficientNet-B0 五種不平衡處理，含黑色素瘤閾值分析與 AP | RQ2 |
| E3 | 依影像切分與依病灶切分的比較 | RQ3 |
| E3b | 同一份依影像切分的測試集：模型 A（原訓練集）、B（移除與測試影像同病灶的影像）、C（移除等量且無關的病灶），以 C − B 為主要比較，並依病灶是否出現過分群 | RQ3 |
| E4 | ISIC 2018 Task 3 驗證集與測試集（1,705 張）的有限度外部驗證，含排除方式的敏感度分析 | RQ1、RQ3 |

所有實驗以隨機種子 42 至 46 重複執行，結果以平均值 ± 標準差與各種子的配對差值呈現，僅作描述性比較。超參數、回合數與決策閾值只依驗證集決定，測試集僅用於回報最終結果。HAM10000 中經確認為同一病灶但編號不同的影像（HAM_0004919 與 HAM_0000140），於切分前合併為同一病灶。

以下為看到正式結果後追加的分析，報告中標明為事後追加，正式結果不變：

- **03 固定回合敏感度分析**：模型 B、C 都固定訓練 20 回合、不提前停止，確認 E3b 的結論不是由提前停止的回合不同造成。
- **補充分析**：病灶層級置換檢驗、各類別效能、AUC、黑色素瘤敏感度與特異度、性別與年齡子群組。設計在計算前寫定，見 `docs/補充分析_事先設計.md`。

## 執行順序

在 Google Colab 開啟，依序執行。筆記本共用雲端硬碟的 `ham10000_project/` 資料夾，可分次執行，已完成的部分會自動略過。

| 筆記本 | 硬體 | 內容 |
|---|---|---|
| `00_data_check.ipynb` | CPU | HAM10000 近似重複檢查（pHash 候選＋SSIM 確認）、ISIC 2018 下載、病灶編號與同一照片比對 |
| `01_baseline.ipynb` | CPU | Baseline 調參、E1／E3／E3b／E4 |
| `02_cnn.ipynb` | T4 GPU | 回合數試跑、EfficientNet-B0 的 E1～E4、彙總、Grad-CAM |
| `03_e3b_fixed_epochs.ipynb` | T4 GPU | E3b 固定 20 回合敏感度分析（約 2 小時） |
| `04_prototype_gradcam.ipynb` | CPU | 驗證 Gradio 版原型與 02 的預測一致，產生正確、漏判、誤報三種 Grad-CAM 案例圖 |
| `05_web_prototype_check.ipynb` | CPU | 驗證網頁版原型的 model.onnx 與 02 的預測、Grad-CAM 一致 |

補充分析：`python analysis/supplementary_analysis.py --base <雲端硬碟的 ham10000_project> --meta <HAM10000_metadata.csv>`，結果見 `results/supplementary/`。

## 系統原型

`prototype_web/` 是正式部署的網頁版，部署於 Hugging Face Spaces（Static）。

- 模型轉成 ONNX 格式（`tools/export_onnx.py`），以 onnxruntime-web 1.20.1 在使用者的瀏覽器中執行，上傳的影像不會傳送到伺服器。
- 功能：七類預測機率、Grad-CAM 熱圖、黑色素瘤就醫提示（閾值 0.16，為種子 42 驗證集上 F2 分數最大的閾值）、衛教資源連結。
- 前處理與 02 相同：轉 RGB → 雙線性縮放為 224 × 224 → ImageNet 正規化。`lesion.js` 的縮放移植自 Pillow，與 PIL 逐像素相同（`python tools/check_resize_port.py`）。
- 驗證：以種子 42 依病灶切分的測試集（1,503 張）比對，七類機率最大差異 0.0140，預測類別一致 99.73%，差異來自 02 以 GPU 半精度計算。熱圖與 PyTorch Grad-CAM 的最大差異為 0.0039。
- 本機試用：在 `prototype_web/` 執行 `python -m http.server`，開啟 http://localhost:8000。

`prototype_gradio/` 是較早的 Gradio 版本，功能相同，用於 04 的驗證。由於 Hugging Face 免費方案無法再選用 CPU 執行 Gradio Space，正式部署改用網頁版。

## 儲存庫結構

```
notebooks/          00 至 05 筆記本
analysis/           補充分析程式
prototype_web/      網頁版原型（正式部署）
prototype_gradio/   Gradio 版原型（04 驗證用）
tools/              ONNX 匯出、縮放移植檢查
docs/               補充分析事先設計、正式版修改紀錄
results/            實驗結果（baseline、cnn、data_check、e3b_fixed、supplementary）
```

影像、快取、預測檔與模型權重不放入儲存庫（網頁版的 model.onnx 除外）。

## 資料

- HAM10000（Tschandl et al., 2018），透過 Kaggle「Skin Cancer MNIST: HAM10000」下載，CC BY-NC-SA 4.0。
- ISIC 2018 Task 3 驗證集與測試集（Codella et al., 2019），由 ISIC Challenge 官方頁面下載，CC BY-NC。

影像不包含在本儲存庫中，筆記本會自動下載。

## 環境

Python 3、PyTorch、torchvision、scikit-learn、scikit-image、imagehash、onnxruntime。實際版本記錄於 `results/` 內的 environment 檔案。
