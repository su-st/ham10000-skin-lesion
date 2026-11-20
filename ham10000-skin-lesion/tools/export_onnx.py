"""把 02 的模型（model__cw-aug__lesion__seed42.pt）轉成網頁版用的 model.onnx

輸出兩個結果：
  probs：七類機率
  cams ：七類各自的 CAM（未經 ReLU 與正規化），形狀 1 × 7 × 7 × 7
EfficientNet-B0 的 features 之後是全域平均池化再接線性層，logit_c 對最後特徵圖的梯度在空間上是常數 W_c /（H × W），
因此 Grad-CAM 等於 ReLU(Σ_k W_ck · A_k) 乘上正的常數，正規化後與 app.py 的 grad_cam 相同（05 筆記本有驗證）。

用法：python tools/export_onnx.py model__cw-aug__lesion__seed42.pt prototype_web/model.onnx
需要 torch、torchvision、onnx
"""
import sys
import torch, torch.nn as nn
from torchvision import models

pt, out = sys.argv[1], sys.argv[2]
m = models.efficientnet_b0(weights=None)
m.classifier[1] = nn.Linear(m.classifier[1].in_features, 7)
m.load_state_dict(torch.load(pt, map_location='cpu', weights_only=True)); m.eval()


class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        a = self.m.features(x)                                   # 1 × 1280 × 7 × 7
        logits = self.m.classifier(self.m.avgpool(a).flatten(1))
        W = self.m.classifier[1].weight                          # 7 × 1280
        b, k, h, w = a.shape
        cams = torch.matmul(W, a.reshape(b, k, h * w)).reshape(b, W.shape[0], h, w)
        return logits.softmax(1), cams


torch.onnx.export(Wrap(m).eval(), (torch.zeros(1, 3, 224, 224),), out, input_names=['input'],
                  output_names=['probs', 'cams'], opset_version=17, dynamo=False)
print('已輸出', out)
