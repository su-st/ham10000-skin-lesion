---
title: 皮膚鏡影像病灶分類原型
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-4.0
---

# 皮膚鏡影像病灶分類原型

數位醫學與影像導論期末專題的系統原型，僅供課程展示，不能取代醫師診斷。

- 模型：EfficientNet-B0（類別權重＋資料增強，依病灶切分，種子 42），以 HAM10000 訓練，轉為 ONNX 格式
- 功能：上傳皮膚鏡影像，顯示七類預測機率、Grad-CAM 熱圖、黑色素瘤就醫提示（閾值 0.16）與衛教連結
- 模型以 onnxruntime-web 在使用者的瀏覽器中執行，影像不會上傳到伺服器
- 只適用皮膚鏡影像，一般手機照片的結果沒有參考價值
