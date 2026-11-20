"""皮膚鏡影像病灶分類原型（數位醫學與影像導論期末專題，課程展示用）

模型：EfficientNet-B0，類別權重＋資料增強，依病灶切分，種子 42（02_cnn 產出的 model__cw-aug__lesion__seed42.pt）
前處理與 02_cnn 相同：轉 RGB → 雙線性縮放為 224 × 224 → 除以 255 → ImageNet 平均值與標準差正規化
黑色素瘤提示閾值：0.16，為種子 42 驗證集上 F2 分數最大的閾值（E2_threshold_per_seed.csv）
"""
import json
import os
import time

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(HERE, 'model__cw-aug__lesion__seed42.pt'))
LINKS_PATH = os.path.join(HERE, 'links.json')
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MEL_THRESHOLD = 0.16
CLASS_ORDER = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_ZH = {
    'akiec': '光化性角化症與表皮內癌',
    'bcc': '基底細胞癌',
    'bkl': '良性角化病灶',
    'df': '皮膚纖維瘤',
    'mel': '黑色素瘤',
    'nv': '黑色素細胞痣',
    'vasc': '血管病灶',
}
MEL = CLASS_ORDER.index('mel')
torch.set_num_threads(max(1, os.cpu_count() or 1))

# ---------- 模型 ----------
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_ORDER))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
model.eval()
with torch.no_grad():  # 暖機，讓顯示的推論時間反映穩定狀態
    model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))

with open(LINKS_PATH, encoding='utf-8') as f:
    LINKS = json.load(f)


def preprocess(img):
    """與 02_cnn 的 load_resized 與 to_input 相同"""
    img = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    x = (arr.astype(np.float32) / 255.0 - MEAN) / STD
    return arr, torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def predict_probs(x):
    return model(x).softmax(1)[0].numpy()


def grad_cam(x, class_idx):
    """與 02_cnn 的 grad_cam 相同，取 features 最後一層"""
    acts, grads = {}, {}

    def fwd_hook(m, i, o):
        acts['a'] = o
        o.register_hook(lambda g: grads.__setitem__('g', g))

    h = model.features[-1].register_forward_hook(fwd_hook)
    try:
        model.zero_grad()
        out = model(x)
        out[0, class_idx].backward()
    finally:
        h.remove()
    w = grads['g'].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts['a']).sum(1)).squeeze().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0


def overlay(arr, cam, alpha=0.4):
    """把熱圖（jet 色階）疊在原圖上，並放大到 448 × 448 方便觀看"""
    import matplotlib
    heat = matplotlib.colormaps['jet'](cam)[..., :3]
    mix = (1 - alpha) * (arr / 255.0) + alpha * heat
    return Image.fromarray(np.uint8(np.clip(mix, 0, 1) * 255)).resize((448, 448), Image.BILINEAR)


def link_md(item):
    if not item.get('url'):
        return '連結整理中'
    src = f"（{item['source']}）" if item.get('source') else ''
    return f"[{item.get('title') or item['url']}]({item['url']}){src}"


def education_md(cls):
    lines = [f"**{CLASS_ZH[cls]}**"]
    for it in LINKS.get(cls, []):
        lines.append(f"- {link_md(it)}")
    lines.append('')
    lines.append('**皮膚自我檢查（不論預測結果都請參考）**')
    lines.append(f"- {link_md(LINKS.get('general', {}))}")
    lines.append('')
    lines.append('不論系統預測為何，若病灶出現不對稱、邊緣不規則、顏色不均、直徑變大或持續變化，請儘速就醫。')
    return '\n'.join(lines)


def analyze(img):
    if img is None:
        raise gr.Error('請先上傳一張皮膚鏡影像')
    arr, x = preprocess(img)
    t0 = time.perf_counter()
    probs = predict_probs(x)
    infer_ms = (time.perf_counter() - t0) * 1000
    top = int(probs.argmax())
    label = {f'{CLASS_ZH[c]}（{c}）': float(probs[i]) for i, c in enumerate(CLASS_ORDER)}

    t1 = time.perf_counter()
    cam_top = overlay(arr, grad_cam(x.clone().requires_grad_(False), top))
    show_mel_cam = probs[MEL] >= MEL_THRESHOLD and top != MEL
    cam_mel = overlay(arr, grad_cam(x.clone(), MEL)) if show_mel_cam else None
    cam_ms = (time.perf_counter() - t1) * 1000

    p_mel = probs[MEL]
    if p_mel >= MEL_THRESHOLD:
        alert = (f"### ⚠️ 建議就醫進一步檢查\n黑色素瘤機率為 **{p_mel:.2f}**，已達本系統的提示門檻（{MEL_THRESHOLD}）。"
                 '門檻刻意設得較低，寧可多提醒，也不要漏掉黑色素瘤，因此即使機率最高的不是黑色素瘤，仍會出現此提示。')
    else:
        alert = (f"### 未達黑色素瘤提示門檻\n黑色素瘤機率為 **{p_mel:.2f}**，未達本系統的提示門檻（{MEL_THRESHOLD}）。"
                 '這不代表病灶一定是良性，仍請依下方的自我檢查原則留意皮膚變化。')

    timing = f"推論時間：{infer_ms:.0f} 毫秒（模型預測），Grad-CAM 計算：{cam_ms:.0f} 毫秒，於 CPU 執行。"
    cam_top_label = f'Grad-CAM 熱圖：模型判斷為「{CLASS_ZH[CLASS_ORDER[top]]}」時注意的區域'
    return (label, alert,
            gr.update(value=cam_top, label=cam_top_label),
            gr.update(value=cam_mel, visible=bool(show_mel_cam)),
            education_md(CLASS_ORDER[top]), timing)


DISCLAIMER = ('**本系統僅供數位醫學與影像導論課程展示，不能取代醫師診斷。** '
              '模型以 HAM10000 皮膚鏡影像訓練，只適用皮膚鏡影像，一般手機拍攝的照片結果沒有參考價值。'
              '模型會判斷錯誤，測試資料中約三分之一的黑色素瘤未被判為黑色素瘤。')

with gr.Blocks(title='皮膚鏡影像病灶分類原型') as demo:
    gr.Markdown('# 皮膚鏡影像病灶分類原型')
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type='pil', label='上傳皮膚鏡影像（jpg 或 png）', height=360)
            btn = gr.Button('開始分析', variant='primary')
            timing = gr.Markdown()
        with gr.Column(scale=1):
            label = gr.Label(num_top_classes=7, label='七類預測機率')
            alert = gr.Markdown()
    with gr.Row():
        cam_top = gr.Image(label='Grad-CAM 熱圖', height=360)
        cam_mel = gr.Image(label='Grad-CAM 熱圖：模型判斷為「黑色素瘤」時注意的區域', height=360, visible=False)
    gr.Markdown('熱圖越紅的區域，代表模型判斷時越依賴該區域。熱圖只能呈現模型注意的位置，不代表模型理解了病灶特徵。')
    gr.Markdown('## 衛教資源')
    edu = gr.Markdown('上傳影像並分析後，會顯示對應類別的衛教連結。')
    btn.click(analyze, inputs=inp, outputs=[label, alert, cam_top, cam_mel, edu, timing])

if __name__ == '__main__':
    demo.launch(share=bool(os.environ.get('GRADIO_SHARE')))
