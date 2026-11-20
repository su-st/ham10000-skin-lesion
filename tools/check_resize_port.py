"""確認 prototype_web/lesion.js 的縮放與 PIL Image.resize(BILINEAR) 逐像素相同

用法：python tools/check_resize_port.py（需要 Node.js 與 Pillow）
以隨機影像測試多種尺寸（含 4032 × 3024 手機照片尺寸、放大與縮小），全部相同才會通過。
"""
import json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)
sizes = [(450, 600), (3024, 4032), (224, 224), (100, 80), (224, 600), (1000, 224), (7, 7)]
with tempfile.TemporaryDirectory() as d:
    cases = []
    for i, (h, w) in enumerate(sizes):
        a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8); a.tofile(f'{d}/in{i}.bin')
        for o in (224, 448):
            np.asarray(Image.fromarray(a).resize((o, o), Image.BILINEAR)).tofile(f'{d}/out{i}_{o}.bin')
            cases.append(dict(i=i, w=w, h=h, ow=o, oh=o, ch=3))
    g = rng.integers(0, 256, (7, 7), dtype=np.uint8); g.tofile(f'{d}/in_g.bin')
    np.asarray(Image.fromarray(g).resize((224, 224), Image.BILINEAR)).tofile(f'{d}/out_g_224.bin')
    cases.append(dict(i='_g', w=7, h=7, ow=224, oh=224, ch=1))
    json.dump(cases, open(f'{d}/cases.json', 'w'))
    r = subprocess.run(['node', os.path.join(HERE, 'check_resize_port.js'), d], capture_output=True, text=True)
    print(r.stdout, r.stderr)
    sys.exit(r.returncode)
