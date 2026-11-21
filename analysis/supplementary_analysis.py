"""期末補充分析（事先設計見 docs/補充分析_事先設計.md）

使用 01、02 已存的預測結果，不重新訓練。輸出：
  S1  置換檢驗（逐張影像打亂，保留供查核）
  S1b 置換檢驗（以病灶為單位打亂，報告採用）
  S2  EfficientNet-B0 各類別效能
  S3  EfficientNet-B0 的 AUC
  S4  三個模型的黑色素瘤敏感度與特異度
  S5  EfficientNet-B0 的性別、年齡子群組
  R1  各模型預測為各類別的張數
  R2  各年齡組的黑色素瘤被判為各類別的張數

用法（在 Colab 掛載雲端硬碟後）：
  python analysis/supplementary_analysis.py --base /content/drive/MyDrive/ham10000_project \
      --meta /path/to/HAM10000_metadata.csv --out results/supplementary
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True, help='雲端硬碟的 ham10000_project 資料夾')
ap.add_argument('--meta', required=True, help='HAM10000_metadata.csv')
ap.add_argument('--out', default='results/supplementary')
ap.add_argument('--n-perm', type=int, default=10000)
args = ap.parse_args()

CNN_DIR = os.path.join(args.base, 'results_final_cnn')
BASE_DIR = os.path.join(args.base, 'results_final_baseline')
OUT = args.out
os.makedirs(OUT, exist_ok=True)
CLS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']; L = list(range(7)); MEL = 4
SEEDS = range(42, 47)
MODELS = ['SVM', 'Random Forest', 'EfficientNet-B0']
sa = pd.read_csv(os.path.join(BASE_DIR, 'split_assignment.csv')).set_index('image_id')
meta = pd.read_csv(args.meta).set_index('image_id')
assert (meta.loc[sa.index, 'dx'] == sa['dx']).all()


def load(model, s):
    if model == 'EfficientNet-B0':
        z = np.load(os.path.join(CNN_DIR, 'predictions', f'prob__cw-aug__lesion__seed{s}.npz'))
        prob = z['test']; pred = prob.argmax(1)
    else:
        z = np.load(os.path.join(BASE_DIR, 'predictions', f'pred__{model.replace(" ", "")}__lesion__seed{s}.npz'), allow_pickle=True)
        prob = None; t = z['test']   # baseline 只保存預測類別，沒有機率
        pred = np.array([CLS.index(v) for v in t]) if t.dtype.kind in 'UO' else t.astype(int)
    ids = z['test_image_id'].astype(str)
    assert (sa.loc[ids, f'seed{s}_lesion'] == 'test').all()
    y = np.array([CLS.index(d) for d in sa.loc[ids, 'dx']])
    return ids, y, pred, prob


f1m = lambda y, p, l=L: f1_score(y, p, labels=l, average='macro', zero_division=0)
ms = lambda v: f'{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}'
save = lambda df, name, **kw: df.to_csv(os.path.join(OUT, name), encoding='utf-8-sig', **kw)

# 0. 檢查：重算的 macro-F1 必須與 01、02 的正式結果一致
ref_cnn = pd.read_csv(os.path.join(CNN_DIR, 'all_runs.csv'))
ref_cnn = ref_cnn[(ref_cnn['實驗設定'] == 'cw+aug') & (ref_cnn['切分'] == 'lesion')].set_index('種子')['F1-score (macro)']
ref_b = pd.read_csv(os.path.join(BASE_DIR, 'all_runs.csv'))
ref_b = ref_b[ref_b['切分'] == 'lesion'].set_index(['模型', '種子'])['F1-score (macro)']
for m in MODELS:
    for s in SEEDS:
        _, y, p, _ = load(m, s); r = ref_cnn[s] if m == 'EfficientNet-B0' else ref_b[(m, s)]
        assert abs(f1m(y, p) - r) < 1e-9, (m, s, f1m(y, p), r)
print('重算 macro-F1 與正式結果一致')


def permutation(level):
    """level='image'：逐張打亂標籤。level='lesion'：打亂「病灶 → 診斷」，同一病灶的影像維持同一標籤"""
    rng = np.random.default_rng(0); rows = []
    for m in MODELS:
        for s in SEEDS:
            ids, y, p, _ = load(m, s); obs = f1m(y, p)
            if level == 'image':
                null = np.array([f1m(rng.permutation(y), p) for _ in range(args.n_perm)])
                extra = {}
            else:
                u, inv = np.unique(sa.loc[ids, 'lesion_id'].to_numpy(), return_inverse=True)
                lab = np.zeros(len(u), int); lab[inv] = y
                assert (lab[inv] == y).all()
                null = np.array([f1m(rng.permutation(lab)[inv], p) for _ in range(args.n_perm)])
                extra = {'測試影像數': len(y), '測試病灶數': len(u)}
            rows.append({'模型': m, '種子': s, **extra, '實際 macro-F1': round(obs, 3), '虛無分布平均': round(null.mean(), 3),
                         '虛無分布最大值': round(null.max(), 3), '≥ 實際值次數': int((null >= obs).sum()),
                         'p 值': round(((null >= obs).sum() + 1) / (args.n_perm + 1), 4)})
            print(level, rows[-1]); sys.stdout.flush()
    return pd.DataFrame(rows)


save(permutation('image'), 'S1_permutation_test.csv', index=False)
save(permutation('lesion'), 'S1b_permutation_test_lesion_level.csv', index=False)

# B. 各類別效能、AUC、黑色素瘤敏感度與特異度
per, auc, melss = [], [], []
for s in SEEDS:
    _, y, p, prob = load('EfficientNet-B0', s)
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=L, zero_division=0)
    for i, c in enumerate(CLS):
        per.append({'種子': s, '類別': c, '精確率': pr[i], '召回率': rc[i], 'F1': f[i], '影像數': sup[i]})
    auc.append({'種子': s, 'macro AUC': roc_auc_score(y, prob, multi_class='ovr', average='macro', labels=L),
                '黑色素瘤 AUC': roc_auc_score((y == MEL).astype(int), prob[:, MEL])})
for m in MODELS:
    for s in SEEDS:
        _, y, p, _ = load(m, s); pos = y == MEL
        melss.append({'模型': m, '種子': s, '敏感度': (p[pos] == MEL).mean(), '特異度': (p[~pos] != MEL).mean()})
per, auc, melss = pd.DataFrame(per), pd.DataFrame(auc), pd.DataFrame(melss)
tb = per.groupby('類別', sort=False).agg(**{k: (k, ms) for k in ['精確率', '召回率', 'F1']}, 影像數=('影像數', 'mean')).reset_index()
tb['影像數'] = tb['影像數'].round(1)
tb.loc[len(tb)] = ['macro 平均'] + [ms(per.groupby('種子')[k].mean()) for k in ['精確率', '召回率', 'F1']] + [round(per.groupby('種子')['影像數'].sum().mean(), 1)]
save(tb, 'S2_per_class_cnn.csv', index=False)
save(pd.DataFrame([{'指標': k, '平均 ± 標準差': ms(auc[k]), '各種子': '、'.join(f'{v:.3f}' for v in auc[k])} for k in ['macro AUC', '黑色素瘤 AUC']]),
     'S3_auc_cnn.csv', index=False)
save(melss.groupby('模型', sort=False).agg(敏感度=('敏感度', ms), 特異度=('特異度', ms)).reset_index(), 'S4_melanoma_sens_spec.csv', index=False)

# C. 子群組
sex_g = lambda v: v if v in ('male', 'female') else '未知'
def age_g(v):
    if pd.isna(v): return '未知'
    return '未滿 40 歲' if v < 40 else ('40 至 59 歲' if v < 60 else '60 歲以上')
sub = []
for s in SEEDS:
    ids, y, p, _ = load('EfficientNet-B0', s); md = meta.loc[ids]
    for var, g in [('性別', md['sex'].map(sex_g).to_numpy()), ('年齡', md['age'].map(age_g).to_numpy())]:
        for lvl in np.unique(g):
            k = g == lvl; labs = sorted(set(y[k])); nm = (y[k] == MEL).sum()
            sub.append({'變項': var, '組別': lvl, '種子': s, '影像數': k.sum(), '類別數': len(labs), 'macro-F1': f1m(y[k], p[k], labs),
                        '黑色素瘤張數': nm, '黑色素瘤召回率': (p[k][y[k] == MEL] == MEL).mean() if nm > 0 else np.nan})
sub = pd.DataFrame(sub)
order = {'male': 0, 'female': 1, '未滿 40 歲': 0, '40 至 59 歲': 1, '60 歲以上': 2, '未知': 9}
res = []
for (var, lvl), d in sub.groupby(['變項', '組別']):
    n, nm = d['影像數'].mean(), d['黑色素瘤張數'].mean()
    res.append({'變項': var, '組別': lvl, '平均影像數': round(n, 1), '類別數（最少至最多）': f"{d['類別數'].min()} 至 {d['類別數'].max()}",
                'macro-F1': ms(d['macro-F1']), '平均黑色素瘤張數': round(nm, 1),
                '黑色素瘤召回率': ms(d['黑色素瘤召回率'].dropna()) if nm >= 10 and d['黑色素瘤召回率'].notna().sum() >= 2 else '不報（少於 10 張）',
                '備註': '僅供參考（少於 30 張）' if n < 30 else '', '_o': (var != '性別', order.get(lvl, 5))})
save(pd.DataFrame(res).sort_values('_o').drop(columns='_o'), 'S5_subgroup_cnn.csv', index=False)
save(sub, 'S5_subgroup_cnn_per_seed.csv', index=False)
comp = []
for s in SEEDS:
    ids, y, _, _ = load('EfficientNet-B0', s); g = meta.loc[ids]['age'].map(age_g).to_numpy()
    for lvl in np.unique(g):
        for i, c in enumerate(CLS):
            comp.append({'組別': lvl, '種子': s, '類別': c, 'n': ((g == lvl) & (y == i)).sum()})
save(pd.DataFrame(comp).groupby(['組別', '類別'])['n'].mean().unstack().round(1), 'S5_age_class_mix.csv')

# 審查後補充：R1 各模型預測為各類別的張數，R2 各年齡組黑色素瘤被判為各類別的張數
rows = []
for m in MODELS:
    for s in SEEDS:
        _, y, p, _ = load(m, s)
        rows.append({'模型': m, '種子': s, **{f'預測為{c}的張數': int((p == i).sum()) for i, c in enumerate(CLS)}})
save(pd.DataFrame(rows), 'R1_predicted_class_counts.csv', index=False)
rows = []
for s in SEEDS:
    ids, y, p, _ = load('EfficientNet-B0', s); g = meta.loc[ids, 'age'].map(age_g).to_numpy()
    for lvl in ['未滿 40 歲', '40 至 59 歲', '60 歲以上']:
        k = (g == lvl) & (y == MEL)
        rows.append({'組別': lvl, '種子': s, '黑色素瘤張數': int(k.sum()), **{f'判為{c}': int((p[k] == i).sum()) for i, c in enumerate(CLS)}})
b = pd.DataFrame(rows); save(b, 'R2_melanoma_pred_by_age_per_seed.csv', index=False)
bs = b.groupby('組別', sort=False).sum(numeric_only=True).drop(columns='種子')
bs['判為痣的比例'] = (bs['判為nv'] / bs['黑色素瘤張數']).round(3)
save(bs, 'R2_melanoma_pred_by_age_pooled.csv')
# S6（2026-11-20 追加）：E2 五種設定的各類別召回率，回答 RQ2 對黑色素瘤以外少數類別的影響，只作描述
from sklearn.metrics import recall_score
CFGS = [('none', '不處理'), ('cw', '類別權重'), ('aug', '資料增強'), ('cw-aug', '類別權重＋資料增強（提出方法）'), ('os-aug', '過採樣＋資料增強')]
rr = []
for cfg, lab in CFGS:
    for s in SEEDS:
        z = np.load(os.path.join(CNN_DIR, 'predictions', f'prob__{cfg}__lesion__seed{s}.npz'))
        ids = z['test_image_id'].astype(str); p = z['test'].argmax(1)
        y = np.array([CLS.index(d) for d in sa.loc[ids, 'dx']])
        rr.append({'設定': lab, '種子': s, **dict(zip(CLS, recall_score(y, p, labels=L, average=None, zero_division=0)))})
rr = pd.DataFrame(rr)
out6 = rr.groupby('設定', sort=False)[CLS].agg(ms).reset_index()
base = rr[rr['設定'] == '不處理'].set_index('種子')[CLS]; prop = rr[rr['設定'] == '類別權重＋資料增強（提出方法）'].set_index('種子')[CLS]
dd = prop - base
out6.loc[len(out6)] = ['提出方法 − 不處理（配對差值）'] + [ms(dd[c]) + f'（{int((dd[c] > 0).sum())}／5）' for c in CLS]
save(out6, 'S6_per_class_recall_E2.csv', index=False)
print('完成，輸出於', OUT)
