# -*- coding: utf-8 -*-
"""展示 W(3x7) 完整训练过程：BT数据 → hinge loss → 梯度下降 → W"""
import json, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(errors="replace")
from iqa_agent.config import get_config

cfg = get_config()
POOL = ["S-TECH", "S-GLOBAL", "S-CONTENT"]
FEAT_KEYS = ["lap_var", "noise", "colorful", "bright", "logpix", "aspect", "spread"]
seed = cfg.seed


def jload(p):
    for enc in ("utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(p)


# ── 加载数据 ──
wd = os.path.join(cfg.runs_dir, "full_tournament")
ranking = jload(os.path.join(wd, "ranking_koniq_v3.json"))
ws = jload(os.path.join(cfg.runs_dir, "bt_pilot", "workset_scores.json"))
new_sc = jload(os.path.join(wd, "new_node_scores.json"))
feats = jload(os.path.join(wd, "route_features_koniq.json"))
feats_v3 = jload(os.path.join(wd, "features_koniq_v3.json"))
ws_by_path = {r["path"]: r for r in ws}

# 全量特征补全
data = []
from PIL import Image
for r in ranking:
    p = r["path"]
    if p in ws_by_path:
        ss = json.loads(ws_by_path[p]["skill_scores"]) if isinstance(ws_by_path[p]["skill_scores"], str) else ws_by_path[p]["skill_scores"]
        ft = feats.get(p)
    else:
        ss = new_sc.get(p)
        ft = feats_v3.get(str(r["node"]))
    if not ss or any(ss.get(sk) is None for sk in POOL):
        continue
    if not isinstance(ft, dict) or "noise" not in ft:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
        gray = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
        c = gray[1:-1,1:-1]
        lap = -4*c + gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
        lap_var = float(np.var(lap))
        box = (gray[:-2,:-2]+gray[:-2,1:-1]+gray[:-2,2:]+gray[1:-1,:-2]+c+gray[1:-1,2:]+gray[2:,:-2]+gray[2:,1:-1]+gray[2:,2:])/9.0
        res = c-box; gx=np.abs(gray[1:-1,2:]-gray[1:-1,:-2])
        thr=np.quantile(gx,0.25); flat=res[gx<=thr]
        noise=float(1.4826*np.median(np.abs(flat-np.median(flat)))) if flat.size else 0.0
        rg=arr[...,0]-arr[...,1]; yb=0.5*(arr[...,0]+arr[...,1])-arr[...,2]
        colorful=float(np.sqrt(rg.std()**2+yb.std()**2)+0.3*np.sqrt(rg.mean()**2+yb.mean()**2))
        h,w=gray.shape
        ft={"lap_var":lap_var,"noise":noise,"colorful":colorful,"bright":float(gray.mean()),"logpix":float(np.log(h*w)),"aspect":float(w/h)}
    data.append({"path":p,"bt":r["bt"],"skills":{sk:ss[sk] for sk in POOL},"feat":ft})

n = len(data)
print(f"训练集节点数: {n}")
print(f"BT分范围: [{min(d['bt'] for d in data):.3f}, {max(d['bt'] for d in data):.3f}]")

# ── 矩阵化 ──
X = np.array([[d["skills"][sk] for sk in POOL] for d in data])  # nx3
bt = np.array([d["bt"] for d in data])                           # n
spread = X.std(axis=1)
raw = np.array([[d["feat"]["lap_var"],d["feat"]["noise"],d["feat"]["colorful"],
                 d["feat"]["bright"],d["feat"]["logpix"],d["feat"]["aspect"],0.0] for d in data])
raw[:,6] = spread
for j, k in enumerate(FEAT_KEYS):
    if k in ("lap_var","noise","colorful"):
        raw[:,j] = np.log(np.maximum(raw[:,j], 1e-6))
mu = raw.mean(axis=0); sd = raw.std(axis=0) + 1e-9
F = (raw - mu) / sd  # nx7

# 训练/验证 split
rng = np.random.default_rng(seed)
perm = rng.permutation(n)
train_idx = perm[:n//2]; valid_idx = perm[n//2:]

# ── 训练循环 ──
steps, batch, lr, l2 = 800, 256, 0.05, 1e-3
W = np.zeros((X.shape[1], F.shape[1]))  # 3x7, 全零初始化
rng_train = np.random.default_rng(seed)

print(f"\n{'='*55}")
print("训练开始: W 全零 → 每次随机抽 128 对 → 梯度下降")
print(f"{'='*55}")

loss_history = []
for t in range(steps):
    idx = rng_train.choice(train_idx, min(batch, len(train_idx)), replace=False)
    grad = np.zeros_like(W)
    cnt = 0; pair_loss = 0.0
    for k in range(0, len(idx)-1, 2):
        i, j = idx[k], idx[k+1]
        if bt[i] == bt[j]:
            continue
        # BT分高的那边是"A应该更好"
        a, b = (i, j) if bt[i] > bt[j] else (j, i)
        # forward pass: 特征 → 权重 → 融合分
        ga = np.exp(W @ F[a]); ga /= ga.sum()
        gb = np.exp(W @ F[b]); gb /= gb.sum()
        sa = float(ga @ X[a])
        sb = float(gb @ X[b])
        d = sa - sb
        if d > 10:  # 已经对了，跳过
            continue
        # sigmoid hinge loss = -log(sigmoid(d))
        sig = 1/(1+np.exp(-d))
        pair_loss += -np.log(max(sig, 1e-10))
        coef = -sig  # = -sigmoid(-d)  — derivative of log(sigmoid(d))
        # 梯度累加
        grad += coef * (np.outer(ga*(X[a]-sa), F[a]) - np.outer(gb*(X[b]-sb), F[b]))
        cnt += 1
    if cnt:
        W -= lr * (grad/cnt + 2*l2*W) / np.sqrt(t/50 + 1)

    if t % 100 == 0 and cnt > 0:
        print(f" step {t:3d}: loss={pair_loss/cnt:.4f}  |W|={np.abs(W).sum():.1f}")

print(f"\n{'='*55}")
print("训练完成: 最终 W(3x7)")
print(f"{'='*55}")
for i, sk in enumerate(POOL):
    print(f"  {sk:12s}: [{', '.join(f'{w:+.4f}' for w in W[i])}]")

# ── 取一张验证集图，走一遍推理 ──
vi = valid_idx[0]
d = data[vi]
f_i = F[vi]
x_i = X[vi]

print(f"\n{'='*55}")
print(f"演示: 取一张验证集图，走一遍 W→logit→softmax→融合分")
print(f"{'='*55}")
print(f"图 {vi} (验证集第1张)")
print(f"标准化特征 z = [{', '.join(f'{v:+.3f}' for v in f_i)}]")
print(f"3个专家分: TECH={x_i[0]:.2f} GLOBAL={x_i[1]:.2f} CONTENT={x_i[2]:.2f}")
logits = W @ f_i
print(f"\n第1步: logits = W(3x7) @ z(7x1)")
print(f"  logit(TECH)   = ({W[0,0]:+.4f})*({f_i[0]:+.3f}) + ({W[0,1]:+.4f})*({f_i[1]:+.3f}) + ... = {logits[0]:+.4f}")
print(f"  logit(GLOBAL) = ({W[1,0]:+.4f})*({f_i[0]:+.3f}) + ({W[1,1]:+.4f})*({f_i[1]:+.3f}) + ... = {logits[1]:+.4f}")
print(f"  logit(CONTENT)= ({W[2,0]:+.4f})*({f_i[0]:+.3f}) + ({W[2,1]:+.4f})*({f_i[1]:+.3f}) + ... = {logits[2]:+.4f}")
e = np.exp(logits - logits.max())
g = e / e.sum()
print(f"\n第2步: softmax (把三个任意实数变成三个正数、和为1)")
print(f"  e^logit(TECH)    = e^({logits[0]:+.4f})  = {e[0]:.4f}")
print(f"  e^logit(GLOBAL)  = e^({logits[1]:+.4f})  = {e[1]:.4f}")
print(f"  e^logit(CONTENT) = e^({logits[2]:+.4f})  = {e[2]:.4f}")
print(f"  总和 = {e.sum():.4f}")
print(f"  → TECH={g[0]:.3f}  GLOBAL={g[1]:.3f}  CONTENT={g[2]:.3f}")
fus = g @ x_i
print(f"\n第3步: 融合分 = 权重·专家分")
print(f"  = {x_i[0]:.2f}*{g[0]:.3f} + {x_i[1]:.2f}*{g[1]:.3f} + {x_i[2]:.2f}*{g[2]:.3f}")
print(f"  = {fus:.4f}")
print(f"\nBT分(这个图在训练集上的BT排行榜分) = {bt[vi]:.4f}")

# 留出验证
vset = set(valid_idx.tolist())
rng_p = np.random.default_rng(seed+1)
pairs = []
while len(pairs) < 2000:
    i, j = rng_p.integers(0, n, 2)
    if i in vset and j in vset and abs(bt[i]-bt[j]) >= 0.5:
        pairs.append((i,j))

def softmax(z):
    z=z-z.max(); e=np.exp(z); return e/e.sum()

scores_dyn = np.array([softmax(W @ F[i]) @ X[i] for i in range(n)])
scores_eq = X.mean(axis=1)
dyn_hits = sum(1 for i,j in pairs if (scores_dyn[i]-scores_dyn[j])*(bt[i]-bt[j]) > 0)
eq_hits = sum(1 for i,j in pairs if (scores_eq[i]-scores_eq[j])*(bt[i]-bt[j]) > 0)
print(f"\n{'='*55}")
print(f"留出集验证 (2000对):")
print(f"  动态融合一致率 = {dyn_hits/len(pairs):.4f}")
print(f"  等权融合一致率 = {eq_hits/len(pairs):.4f}")
print(f"  → 动态融合 优于 等权: {'是' if dyn_hits>eq_hits else '否'}")
print(f"{'='*55}")
