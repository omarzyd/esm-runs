"""Evolving the Stain Map (ESM): core script for data preparation and training runs.

Run it in one of three ways:
    python esm_core.py --prepare --out DATA_DIR [--full]
    python esm_core.py --runs B-cmamae-s0,B-ga-s1 --data DATA_DIR --out OUT --gpu 0
    python esm_core.py --smoke          (tiny synthetic test, runs on the CPU)

Sources (official code and docs this file adapts):
- WILDS Camelyon17 loader: https://github.com/p-lambda/wilds (get_dataset, metadata_fields, split_array)
- torchvision models: https://pytorch.org/vision/stable/models.html (resnet18 ImageNet weights, densenet121)
- H&E colour deconvolution matrix: scikit-image skimage.color.rgb2hed (rgb_from_hed; Ruifrok & Johnston 2001)
- kornia: kornia.filters.filter2d / gaussian_blur2d / laplacian, kornia.enhance.jpeg_codec_differentiable
- pyribs: https://docs.pyribs.org (GridArchive, EvolutionStrategyEmitter for CMA-MAE, GaussianEmitter, Scheduler)
- mealpy 3: https://mealpy.readthedocs.io (FloatVar, Optimizer.solve, starting_solutions)
GLUE (custom code, flagged with "# GLUE"): the genome decoder, descriptors and realism bounds, archive
re-scoring, the exact evaluation budget, the run list and the worker split.
"""
import argparse, copy, glob, hashlib, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------
# Run list (fixed; every worker uses the same list)                                        # GLUE
# ----------------------------------------------------------------------------------------------
ARMS_B = ["erm", "hedjitter", "random13", "cmamae", "mapelites", "cmaes_hard",
          "ga", "de", "pso", "abc", "acor", "gwo"]
SEEDS_B = [0, 1, 2, 3, 4]
ARMS_C = ["erm", "hedjitter", "random13", "cmamae"]
SEEDS_C = [0, 1, 2]
RUNS_B = [f"B-{a}-s{s}" for s in SEEDS_B for a in ARMS_B]    # dev track (ResNet-18 ImageNet, subset)
RUNS_C = [f"C-{a}-s{s}" for s in SEEDS_C for a in ARMS_C]    # official-style track (DenseNet-121 scratch, full)


def parse_run(run_id):
    exp, arm, s = run_id.split("-")
    return exp, arm, int(s[1:])


def worker_share(runs, worker_id, n_workers):
    return [r for i, r in enumerate(runs) if i % n_workers == worker_id - 1]


# ----------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------
CFG = {
    "B": dict(arch="resnet18_imagenet", epochs=5, batch=128, lr=0.01, wd=1e-4, train_key="train"),
    "C": dict(arch="densenet121_scratch", epochs=5, batch=128, lr=0.004, wd=1e-2, train_key="trainfull"),
    "views": 2,            # archive views per patch per step (augmenting arms)
    "clean_frac": 0.25,    # share of views left clean
    "budget": 200,         # new candidate evaluations per archive update (every arm)
    "probe": 64,           # probe patches per candidate (same batch for all candidates of an update)
    "pool_k": 32,          # pool size for hardest-only arms
    "grid": (16, 16),
    "ema": 0.995,
    "mc_teacher_acc": 0.6, # minimal criterion: the teacher must still classify the transformed probe
    "tmin": -1.0,          # archive threshold_min (regret below this is rejected)
    "emitters": 5, "emitter_batch": 20, "pop": 20,
}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------------------------
# Data preparation (worker 1, once)
# ----------------------------------------------------------------------------------------------
def sha_of_arrays(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def prepare(out, root, full=False, n_per_hosp=10000, n_val=10000, seed=0):
    from concurrent.futures import ThreadPoolExecutor
    from wilds import get_dataset
    os.makedirs(out, exist_ok=True); os.makedirs(root, exist_ok=True)
    ds = get_dataset("camelyon17", download=True, root_dir=root)
    meta = ds.metadata_array.numpy()
    fields = list(ds.metadata_fields)
    hosp, slide = meta[:, fields.index("hospital")], meta[:, fields.index("slide")]
    y, split = ds.y_array.numpy(), np.asarray(ds.split_array)
    rng = np.random.default_rng(seed)
    tr = []
    for h in (0, 3, 4):
        for c in (0, 1):
            idx = np.where((split == 0) & (hosp == h) & (y == c))[0]
            tr.append(rng.choice(idx, n_per_hosp // 2, replace=False))
    sets = {"train": np.sort(np.concatenate(tr)),
            "val": np.sort(rng.choice(np.where(split == 3)[0], n_val, replace=False)),
            "test": np.where(split == 2)[0]}
    if full:
        sets["trainfull"] = np.where(split == 0)[0]
        sets["valfull"] = np.where(split == 3)[0]
    info = {"wilds_split_counts": {k: int((split == v).sum()) for k, v in ds.split_dict.items()}}
    for name, idx in sets.items():
        t0 = time.time()
        X = np.lib.format.open_memmap(f"{out}/{name}_x.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(idx), 96, 96, 3))
        def load(j):
            return np.asarray(ds.get_input(int(j)).convert("RGB"), dtype=np.uint8)
        with ThreadPoolExecutor(16) as ex:
            for i, arr in enumerate(ex.map(load, idx, chunksize=256)):
                X[i] = arr
        X.flush(); del X
        for k, v in (("idx", idx), ("y", y[idx]), ("hosp", hosp[idx]), ("slide", slide[idx])):
            np.save(f"{out}/{name}_{k}.npy", v.astype(np.int64))
        info[name] = {"n": int(len(idx)), "sha_idx": sha_of_arrays(idx), "sec": round(time.time() - t0)}
        print(name, info[name], flush=True)
    info["sha_all"] = sha_of_arrays(*[sets[k] for k in sorted(sets)])
    json.dump(info, open(f"{out}/esm_meta.json", "w"), indent=1)
    print("PREPARED", info["sha_all"], flush=True)


def find_data_dir(explicit=None):
    if explicit and os.path.exists(os.path.join(explicit, "esm_meta.json")):
        return explicit
    for pat in ["/kaggle/input/**/esm_meta.json", "/content/esm_data/**/esm_meta.json", "esm_data/esm_meta.json"]:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError("esm_meta.json not found: attach the prepare notebook output")


def load_split(d, name):
    meta = json.load(open(f"{d}/esm_meta.json"))
    idx = np.load(f"{d}/{name}_idx.npy")
    assert sha_of_arrays(idx) == meta[name]["sha_idx"], f"checksum mismatch for {name}: wrong data"
    X = np.load(f"{d}/{name}_x.npy", mmap_mode="r")
    return dict(x=X, y=np.load(f"{d}/{name}_y.npy"), hosp=np.load(f"{d}/{name}_hosp.npy"),
                slide=np.load(f"{d}/{name}_slide.npy"))


def smoke_data(n_train=192, n_val=64, n_test=96):                                     # GLUE
    rng = np.random.default_rng(0)
    def make(n, hosps):
        y = rng.integers(0, 2, n); h = rng.choice(hosps, n)
        base = np.array([[200, 120, 180], [230, 170, 210]])[y]          # purple-ish vs pink-ish
        x = np.clip(base[:, None, None, :] + rng.normal(0, 25, (n, 96, 96, 3)) + h[:, None, None, None] * 3, 0, 255)
        return dict(x=x.astype(np.uint8), y=y, hosp=h, slide=h * 10 + rng.integers(0, 3, n))
    return make(n_train, [0, 3, 4]), make(n_val, [1]), make(n_test, [2])


# ----------------------------------------------------------------------------------------------
# Stain + scanner genome (13 genes in [0,1])                                                # GLUE
# order: stain (optical density) -> gain, contrast, white balance -> scale -> blur -> sharpen -> noise -> JPEG
# ----------------------------------------------------------------------------------------------
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
_RGB_FROM_HED[2] = np.cross(_RGB_FROM_HED[0], _RGB_FROM_HED[1])   # as in skimage.color
RGB_FROM_HED = torch.tensor(_RGB_FROM_HED, dtype=torch.float32)
HED_FROM_RGB = torch.linalg.inv(RGB_FROM_HED)
GENES = ["logscale_H", "logscale_E", "shift_H", "shift_E", "rot_deg", "gain", "contrast", "cast",
         "blur_r", "sharpen", "noise", "jpeg_q", "scale"]
LO = np.array([math.log(0.5), math.log(0.5), -0.15, -0.15, -20, 0.65, 0.6, -0.06, 0.0, 0.0, 0.0, 50, 0.9])
HI = np.array([math.log(1.8), math.log(1.8), 0.15, 0.15, 20, 1.35, 1.4, 0.06, 2.0, 1.0, 0.04, 95, 1.1])
IDENTITY = np.clip((np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 95, 1.0]) - LO) / (HI - LO), 0, 1)


def decode(g):
    lo, hi = torch.tensor(LO, dtype=torch.float32, device=g.device), torch.tensor(HI, dtype=torch.float32, device=g.device)
    return lo + g.clamp(0, 1) * (hi - lo)


def to_conc(x):
    od = -torch.log(x.clamp(min=1e-6))
    return torch.einsum("bchw,cs->bshw", od, HED_FROM_RGB.to(x))


def apply_stain(x, sH, sE, bH, bE, rot):
    conc = to_conc(x)
    cH = conc[:, 0] * sH[:, None, None] + bH[:, None, None]
    cE = conc[:, 1] * sE[:, None, None] + bE[:, None, None]
    conc = torch.stack([cH, cE, conc[:, 2]], 1)
    M = RGB_FROM_HED.to(x)
    h, e = M[0], M[1]
    hn = h / h.norm(); ep = e - (e @ hn) * hn; ep = ep / ep.norm()
    hrot = torch.cos(rot)[:, None] * h[None] + torch.sin(rot)[:, None] * ep[None] * h.norm()
    Mb = M[None].repeat(x.shape[0], 1, 1); Mb[:, 0] = hrot
    return torch.exp(-torch.einsum("bshw,bsc->bchw", conc, Mb)).clamp(0, 1)


def disk_kernels(r, k=5):
    c = torch.arange(k, device=r.device, dtype=torch.float32) - k // 2
    d = torch.sqrt(c[None, :] ** 2 + c[:, None] ** 2)[None]
    w = (r[:, None, None] + 0.5 - d).clamp(0, 1)
    return w / w.sum(dim=(1, 2), keepdim=True)


def apply_genome(x, g):
    import kornia
    p = decode(g)
    y = apply_stain(x, torch.exp(p[:, 0]), torch.exp(p[:, 1]), p[:, 2], p[:, 3], p[:, 4] * math.pi / 180)
    y = y * p[:, 5, None, None, None]
    m = y.mean(dim=(1, 2, 3), keepdim=True)
    y = (y - m) * p[:, 6, None, None, None] + m
    wb = torch.stack([1 + p[:, 7], torch.ones_like(p[:, 7]), 1 - p[:, 7]], 1)[:, :, None, None]
    y = (y * wb).clamp(0, 1)
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0] = 1 / p[:, 12]; th[:, 1, 1] = 1 / p[:, 12]
    y = F.grid_sample(y, F.affine_grid(th, y.shape, align_corners=False), mode="bilinear",
                      padding_mode="reflection", align_corners=False)
    y = kornia.filters.filter2d(y, disk_kernels(p[:, 8]), border_type="reflect")
    bl = kornia.filters.gaussian_blur2d(y, (5, 5), (1.0, 1.0))
    y = (y + p[:, 9, None, None, None] * (y - bl)).clamp(0, 1)
    y = (y + torch.randn_like(y) * p[:, 10, None, None, None] * torch.sqrt(y.clamp(min=1e-4))).clamp(0, 1)
    y = kornia.enhance.jpeg_codec_differentiable(y, p[:, 11].contiguous())
    return y.clamp(0, 1)


def hed_jitter(x, sigma=0.1):
    """Hand-made H&E jitter baseline (Tellez 2019 / Gao 2023 style): scale ~U(1±s), shift ~U(±s)."""
    B = x.shape[0]; u = lambda: (torch.rand(B, device=x.device) * 2 - 1) * sigma
    return apply_stain(x, 1 + u(), 1 + u(), u(), u(), torch.zeros(B, device=x.device))


def geo(x):
    """Flips and 90-degree rotations (label-preserving; used by every arm)."""
    if random.random() < 0.5: x = x.flip(3)
    if random.random() < 0.5: x = x.flip(2)
    return torch.rot90(x, random.randint(0, 3), (2, 3))


# ----------------------------------------------------------------------------------------------
# Descriptors and realism bounds (training hospitals only)                                  # GLUE
# ----------------------------------------------------------------------------------------------
def image_stats(x):
    """Per-image stats: [p95 H concentration, log Laplacian variance, mean E concentration, R-B cast]."""
    import kornia
    c = to_conc(x)
    d1 = torch.quantile(c[:, 0].flatten(1), 0.95, dim=1)
    gray = x.mean(1, keepdim=True)
    d2 = torch.log(kornia.filters.laplacian(gray, 3).flatten(1).var(dim=1) + 1e-8)
    d3 = c[:, 1].flatten(1).mean(1)
    d4 = (x[:, 0] - x[:, 2]).flatten(1).mean(1)
    return torch.stack([d1, d2, d3, d4], 1)


def realism_bounds(train_x, device, n=2000, q=0.003, widen=0.25):
    idx = np.random.default_rng(123).choice(len(train_x), min(n, len(train_x)), replace=False)
    st = []
    for i in range(0, len(idx), 256):
        xb = torch.from_numpy(np.ascontiguousarray(train_x[np.sort(idx[i:i + 256])])).to(device)
        st.append(image_stats(xb.permute(0, 3, 1, 2).float() / 255))
    st = torch.cat(st)
    lo, hi = torch.quantile(st, q, dim=0), torch.quantile(st, 1 - q, dim=0)
    w = hi - lo
    return (lo - widen * w).cpu().numpy(), (hi + widen * w).cpu().numpy()


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
def make_model(arch, smoke=False):
    import torchvision
    if arch == "resnet18_imagenet":
        w = None if smoke else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        m = torchvision.models.resnet18(weights=w); m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m = torchvision.models.densenet121(weights=None); m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m


def norm(x):
    return (x - MEAN.to(x)) / STD.to(x)


@torch.no_grad()
def ema_update(ema, model, d):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(device).permute(0, 3, 1, 2).float() / 255
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(torch.softmax(model(norm(xb)).float(), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------------------------
# Fitness evaluation shared by every search arm (exact budget counter)                       # GLUE
# ----------------------------------------------------------------------------------------------
class Evaluator:
    def __init__(self, student, teacher, probe_x, probe_y, bounds, device):
        self.s, self.t, self.px, self.py, self.dev = student, teacher, probe_x, probe_y, device
        self.lo, self.hi = bounds
        self.n_new = 0

    @torch.no_grad()
    def __call__(self, G, count=True):
        G = np.atleast_2d(np.asarray(G, dtype=np.float32))
        if count: self.n_new += len(G)
        obj, meas, ok = np.zeros(len(G)), np.zeros((len(G), 2)), np.zeros(len(G), bool)
        P = len(self.px)
        self.s.eval(); self.t.eval()
        per = max(1, 2048 // P)
        for i in range(0, len(G), per):
            g = torch.from_numpy(G[i:i + per]).to(self.dev)
            n = len(g)
            x = self.px.repeat(n, 1, 1, 1)
            gx = g.repeat_interleave(P, 0)
            xt = apply_genome(x, gx)
            st = image_stats(xt).view(n, P, 4).mean(1).cpu().numpy()
            y = self.py.repeat(n)
            with torch.autocast(self.dev.type, enabled=self.dev.type == "cuda"):
                ls = F.cross_entropy(self.s(norm(xt)).float(), y, reduction="none").view(n, P).mean(1)
                lt_all = self.t(norm(xt)).float()
            lt = F.cross_entropy(lt_all, y, reduction="none").view(n, P).mean(1)
            acc_t = (lt_all.argmax(1) == y).float().view(n, P).mean(1)
            reg = (ls - lt).cpu().numpy()
            plaus = np.all((st >= self.lo) & (st <= self.hi), axis=1)
            good = plaus & (acc_t.cpu().numpy() >= CFG["mc_teacher_acc"])
            obj[i:i + n] = np.where(good, reg, -10.0)
            meas[i:i + n] = st[:, :2]
            ok[i:i + n] = good
        return obj, meas, ok


class BudgetDone(Exception):
    pass


# ----------------------------------------------------------------------------------------------
# Search arms                                                                                # GLUE
# ----------------------------------------------------------------------------------------------
class Arm:
    kind = "none"        # none | hed | pool

    def __init__(self, name, seed, bounds):
        self.name, self.seed, self.bounds = name, seed, bounds
        self.pool = np.zeros((0, 13), np.float32)
        self.rng = np.random.default_rng(seed + 1000)

    def update(self, ev):
        pass

    def stats(self):
        return {"pool": int(len(self.pool))}


class ArmNone(Arm):
    kind = "none"


class ArmHED(Arm):
    kind = "hed"


class ArmRandom(Arm):
    kind = "pool"

    def update(self, ev):
        G = self.rng.random((CFG["budget"], 13)).astype(np.float32)
        _, _, ok = ev(G)
        self.pool = G[ok] if ok.any() else self.pool


class ArmTopK(Arm):
    """Hardest-only arms: keep the top-K plausible candidates (re-scored every update)."""
    kind = "pool"

    def _merge(self, ev, G, obj, ok):
        if len(self.pool):
            o2, _, k2 = ev(self.pool, count=False)
            G, obj, ok = np.vstack([G, self.pool]), np.concatenate([obj, o2]), np.concatenate([ok, k2])
        keep = np.where(ok)[0]
        if len(keep):
            keep = keep[np.argsort(-obj[keep])][:CFG["pool_k"]]
            self.pool = G[keep].astype(np.float32)


class ArmMealpy(ArmTopK):
    def __init__(self, name, seed, bounds):
        super().__init__(name, seed, bounds)
        self.start = None

    def update(self, ev):
        from mealpy import FloatVar, GA, DE, PSO, ABC, ACOR, GWO
        cls = {"ga": GA.BaseGA, "de": DE.OriginalDE, "pso": PSO.OriginalPSO, "abc": ABC.OriginalABC,
               "acor": ACOR.OriginalACOR, "gwo": GWO.OriginalGWO}[self.name]
        log_G, log_o, log_ok = [], [], []
        budget = CFG["budget"]

        def f(sol):
            if len(log_G) >= budget:
                raise BudgetDone()
            o, _, k = ev(sol)
            log_G.append(np.asarray(sol, np.float32)); log_o.append(o[0]); log_ok.append(k[0])
            return float(o[0])

        problem = {"obj_func": f, "bounds": FloatVar(lb=[0.0] * 13, ub=[1.0] * 13), "minmax": "max", "log_to": None}
        model = cls(epoch=10_000, pop_size=CFG["pop"])
        start = None
        if len(self.pool) >= CFG["pop"]:
            start = [p for p in self.pool[:CFG["pop"]]]
        try:
            model.solve(problem, termination={"max_fe": budget + CFG["pop"]}, starting_solutions=start,
                        seed=int(self.rng.integers(1e9)))
        except BudgetDone:
            pass
        if log_G:
            self._merge(ev, np.array(log_G), np.array(log_o), np.array(log_ok))


class ArmRibs(Arm):
    """pyribs arms: CMA-MAE (alpha>0, pool = result archive), CMA-ES-hard (alpha=0, pool = top-K),
    MAP-Elites (Gaussian emitters, pool = archive)."""
    kind = "pool"

    def __init__(self, name, seed, bounds):
        super().__init__(name, seed, bounds)
        from ribs.archives import GridArchive
        from ribs.emitters import EvolutionStrategyEmitter, GaussianEmitter
        from ribs.schedulers import Scheduler
        lo, hi = bounds
        ranges = [(float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))]
        kw = dict(solution_dim=13, dims=CFG["grid"], ranges=ranges, seed=seed)
        if name == "mapelites":
            self.archive = GridArchive(**kw, learning_rate=1.0, threshold_min=CFG["tmin"])
            self.result = None
        else:
            alpha = 0.01 if name == "cmamae" else 0.0
            self.archive = GridArchive(**kw, learning_rate=alpha, threshold_min=CFG["tmin"])
            self.result = GridArchive(**kw)
        self._Sched, self._ES, self._GE = Scheduler, EvolutionStrategyEmitter, GaussianEmitter
        self.n_restart = 0
        self._build()
        self.topk = ArmTopK(name, seed, bounds)

    def _build(self):
        b, s = [(0.0, 1.0)] * 13, self.seed + 100 * self.n_restart
        if self.name == "mapelites":
            em = [self._GE(self.archive, sigma=0.1, x0=IDENTITY, bounds=b, batch_size=CFG["emitter_batch"],
                           seed=s + i) for i in range(CFG["emitters"])]
        else:
            em = [self._ES(self.archive, x0=IDENTITY, sigma0=0.15, ranker="imp", selection_rule="mu",
                           restart_rule="basic", bounds=b, batch_size=CFG["emitter_batch"], seed=s + i)
                  for i in range(CFG["emitters"])]
        self.sched = self._Sched(self.archive, em, result_archive=self.result)

    def _store(self):
        return self.result if self.result is not None else self.archive

    def update(self, ev):
        if self.name == "cmaes_hard" and len(self.topk.pool):   # re-score the hardest-only pool
            o, _, k = ev(self.topk.pool, count=False)
            self.topk.pool, self._topk_obj = self.topk.pool[k], o[k]
        st = self._store()
        if len(st) > 0:                                   # re-score stored elites under the current model
            sol = st.data("solution")
            o, m, k = ev(sol, count=False)
            self.archive.clear()
            if self.result is not None: self.result.clear()
            if k.any():
                self.archive.add(sol[k], o[k], m[k])
                if self.result is not None: self.result.add(sol[k], o[k], m[k])
        n_iter = max(1, CFG["budget"] // (CFG["emitters"] * CFG["emitter_batch"]))
        for _ in range(n_iter):
            sol = self.sched.ask()
            o, m, k = ev(sol)
            try:
                self.sched.tell(o, m)
            except IndexError:            # every candidate rejected and the archive is empty: restart emitters
                self.n_restart += 1
                self._build()
            if self.name == "cmaes_hard":
                self._topk_add(sol, o, k)
        if self.name == "cmaes_hard":
            self.pool = self.topk.pool
        else:
            st = self._store()
            self.pool = st.data("solution").astype(np.float32) if len(st) else self.pool

    def _topk_add(self, sol, o, k):
        P = self.topk.pool
        keep = np.where(k)[0]
        G = np.vstack([P, sol[keep]]) if len(P) else sol[keep]
        if len(G) == 0: return
        self.topk.pool = G.astype(np.float32)
        self._topk_obj = np.concatenate([getattr(self, "_topk_obj", np.zeros(0))[:len(P)], o[keep]])
        order = np.argsort(-self._topk_obj)[:CFG["pool_k"]]
        self.topk.pool, self._topk_obj = self.topk.pool[order], self._topk_obj[order]

    def stats(self):
        st = self._store()
        return {"pool": int(len(self.pool)), "cells": int(len(st)),
                "coverage": round(len(st) / (CFG["grid"][0] * CFG["grid"][1]), 3)}


def make_arm(name, seed, bounds):
    if name == "erm": return ArmNone(name, seed, bounds)
    if name == "hedjitter": return ArmHED(name, seed, bounds)
    if name == "random13": return ArmRandom(name, seed, bounds)
    if name in ("cmamae", "cmaes_hard", "mapelites"): return ArmRibs(name, seed, bounds)
    return ArmMealpy(name, seed, bounds)


# ----------------------------------------------------------------------------------------------
# One training run
# ----------------------------------------------------------------------------------------------
def probe_batch(tr, n, rng, device):
    groups = [np.where((tr["hosp"] == h) & (tr["y"] == c))[0] for h in np.unique(tr["hosp"]) for c in (0, 1)]
    groups = [g for g in groups if len(g)]
    per = max(1, n // len(groups))
    idx = np.sort(np.concatenate([rng.choice(g, min(per, len(g)), replace=False) for g in groups]))
    x = torch.from_numpy(np.ascontiguousarray(tr["x"][idx])).to(device).permute(0, 3, 1, 2).float() / 255
    return x, torch.from_numpy(tr["y"][idx]).long().to(device)


def run_one(run_id, data, out_root, device, smoke=False):
    exp, arm_name, seed = parse_run(run_id)
    out = os.path.join(out_root, run_id)
    if os.path.exists(os.path.join(out, "metrics.json")):
        print("skip (done)", run_id, flush=True); return
    os.makedirs(out, exist_ok=True)
    cfg = dict(CFG[exp]); set_seed(seed)
    tr, va, te = data["train"], data["val"], data["test"]
    if smoke: cfg.update(epochs=2, batch=32)
    rng = np.random.default_rng(seed)
    bounds = data["bounds"]
    model = make_model(cfg["arch"], smoke).to(device).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["wd"], nesterov=True)
    n = len(tr["y"]); steps = cfg["epochs"] * math.ceil(n / cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    arm = make_arm(arm_name, seed, bounds)
    X_all = tr["x"]; Y_all = torch.from_numpy(tr["y"]).long()
    log = open(os.path.join(out, "log.jsonl"), "w")
    best = (-1, None, -1)
    t_start = time.time(); search_sec = 0.0
    for ep in range(cfg["epochs"]):
        if arm.kind == "pool" and ep >= 1:
            t0 = time.time()
            px, py = probe_batch(tr, CFG["probe"], rng, device)
            ev = Evaluator(model, ema, px, py, bounds, device)
            arm.update(ev)
            search_sec += time.time() - t0
        model.train()
        perm = rng.permutation(n); tot = 0.0; nb = 0
        for i in range(0, n, cfg["batch"]):
            bi = np.sort(perm[i:i + cfg["batch"]])
            xb = torch.from_numpy(np.ascontiguousarray(X_all[bi])).to(device, non_blocking=True)
            xb = xb.permute(0, 3, 1, 2).float() / 255
            yb = Y_all[bi].to(device)
            views = []
            if arm.kind == "none":
                views = [geo(xb)]
            else:
                for _ in range(CFG["views"]):
                    v = geo(xb)
                    if arm.kind == "hed" or ep == 0 or len(arm.pool) == 0:
                        v = hed_jitter(v, 0.1 if arm.kind == "hed" else 0.05)
                    else:
                        with torch.no_grad():
                            g = arm.pool[rng.integers(0, len(arm.pool), len(v))]
                            aug = apply_genome(v, torch.from_numpy(g).to(device))
                            keep_clean = torch.rand(len(v), device=device) < CFG["clean_frac"]
                            v = torch.where(keep_clean[:, None, None, None], v, aug)
                    views.append(v)
            xin = torch.cat(views); yin = yb.repeat(len(views))
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(norm(xin).contiguous(memory_format=torch.channels_last)), yin)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            ema_update(ema, model, CFG["ema"])
            tot += loss.item(); nb += 1
        pv = predict(ema, va["x"], device)
        vacc = float(((pv > 0.5) == va["y"]).mean())
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "val_acc_ema": vacc, "sec": round(time.time() - t_start),
               "search_sec": round(search_sec), **arm.stats()}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(run_id, rec, flush=True)
        if vacc > best[0]:
            best = (vacc, {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}, ep)
    ema.load_state_dict(best[1])
    pt = predict(ema, te["x"], device)
    pred = (pt > 0.5).astype(int)
    acc = float((pred == te["y"]).mean())
    slides = {int(s): float((pred[te["slide"] == s] == te["y"][te["slide"] == s]).mean()) for s in np.unique(te["slide"])}
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(te["y"], pt))
    except Exception:
        auc = None
    metrics = {"run_id": run_id, "exp": exp, "arm": arm_name, "seed": seed, "arch": cfg["arch"],
               "test_acc": acc, "val_acc": best[0], "best_epoch": best[2], "test_auc": auc,
               "worst_slide_acc": min(slides.values()), "slide_acc": slides,
               "minutes": round((time.time() - t_start) / 60, 2), "search_minutes": round(search_sec / 60, 2),
               "final_pool": int(len(arm.pool)), "arm_stats": arm.stats(), "cfg": cfg, "shared_cfg":
               {k: v for k, v in CFG.items() if k not in ("B", "C")}}
    np.savez_compressed(os.path.join(out, "preds_test.npz"), prob=pt.astype(np.float16),
                        y=te["y"].astype(np.int8), slide=te["slide"].astype(np.int16))
    np.save(os.path.join(out, "pool_final.npy"), arm.pool)
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best[1].items()},
               os.path.join(out, "model_ema_fp16.pt"))      # weights: GitHub Release + local output, not git
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print("DONE", run_id, "test_acc=%.4f val_acc=%.4f" % (acc, best[0]), flush=True)


def load_all(data_dir, exp, device, smoke=False):
    if smoke:
        tr, va, te = smoke_data()
    else:
        key = CFG[exp]["train_key"]
        tr = load_split(data_dir, key); va = load_split(data_dir, "val"); te = load_split(data_dir, "test")
        tr["x"] = np.ascontiguousarray(tr["x"])     # into RAM for fast random access
    return {"train": tr, "val": va, "test": te,
            "bounds": realism_bounds(tr["x"], device, widen=3.0 if smoke else 0.25)}


# ----------------------------------------------------------------------------------------------
# GitHub run queue: claim -> train -> push results -> upload weights                      # GLUE
# Sources: git CLI docs (clone/commit/push/pull --rebase); GitHub REST API "Releases"
# (https://docs.github.com/en/rest/releases): get release by tag, upload/delete release asset.
# ----------------------------------------------------------------------------------------------
RESULT_FILES = ("metrics.json", "log.jsonl", "preds_test.npz", "pool_final.npy")
STALE_MIN = {"B": 40, "C": 240}       # a claim older than this with no result is taken over


_SECRETS = []


def _scrub(text):
    for t in _SECRETS:
        text = text.replace(t, "***")
    return text


def _git(cwd, *args, check=True):
    r = __import__("subprocess").run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(_scrub(f"git {args[0]} failed: {r.stderr.strip()[-300:]}"))
    return r


class Queue:
    def __init__(self, repo, token, clone_dir, worker, ns=""):
        self.repo, self.token, self.dir, self.worker, self.ns = repo, token, clone_dir, worker, ns
        _SECRETS.append(token)
        url = f"https://x-access-token:{token}@github.com/{repo}.git"
        if not os.path.exists(os.path.join(clone_dir, ".git")):
            _git(".", "clone", "--quiet", url, clone_dir)
        _git(self.dir, "config", "user.name", f"esm-{worker}")
        _git(self.dir, "config", "user.email", "esm-worker@users.noreply.github.com")

    def sync(self):
        _git(self.dir, "fetch", "--quiet", "origin")
        _git(self.dir, "reset", "--quiet", "--hard", "origin/main")

    def state(self, plan):
        done = {r for r in plan if os.path.exists(os.path.join(self.dir, self.ns + "results", r, "metrics.json"))}
        claims = {}
        for p in glob.glob(os.path.join(self.dir, self.ns + "claims", "*.json")):
            try:
                c = json.load(open(p)); claims[c["run"]] = c
            except Exception:
                pass
        return done, claims

    def _push(self, msg, paths, tries=8):
        _git(self.dir, "add", *paths)
        if _git(self.dir, "diff", "--cached", "--quiet", check=False).returncode == 0:
            return True
        _git(self.dir, "commit", "--quiet", "-m", msg)
        for i in range(tries):
            if _git(self.dir, "push", "--quiet", "origin", "HEAD:main", check=False).returncode == 0:
                return True
            if _git(self.dir, "pull", "--quiet", "--rebase", "origin", "main", check=False).returncode != 0:
                _git(self.dir, "rebase", "--abort", check=False)
                return False
            time.sleep(2 + 3 * random.random() * (i + 1))
        return False

    def claim(self, plan):
        for _ in range(10):
            self.sync()
            done, claims = self.state(plan)
            now = time.time()
            free = [r for r in plan if r not in done and
                    (r not in claims or now - claims[r]["time"] > 60 * STALE_MIN[parse_run(r)[0]])]
            if not free:
                active = [r for r in plan if r not in done and r in claims]
                return None, len(done), active
            r = free[0]
            os.makedirs(os.path.join(self.dir, self.ns + "claims"), exist_ok=True)
            json.dump({"run": r, "worker": self.worker, "time": now},
                      open(os.path.join(self.dir, self.ns + "claims", f"{r}.json"), "w"))
            _git(self.dir, "add", f"{self.ns}claims/{r}.json")
            _git(self.dir, "commit", "--quiet", "-m", f"claim {r} by {self.worker}")
            if _git(self.dir, "push", "--quiet", "origin", "HEAD:main", check=False).returncode == 0:
                return r, len(done), []
            time.sleep(1 + 2 * random.random())          # someone pushed first: re-read and try again
        return None, -1, []

    def publish(self, run_id, out_root, depth=0):
        if depth > 3:
            return False
        src, dst = os.path.join(out_root, run_id), os.path.join(self.dir, self.ns + "results", run_id)
        os.makedirs(dst, exist_ok=True)
        import shutil
        for f in RESULT_FILES:
            if os.path.exists(os.path.join(src, f)):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        ok = self._push(f"result {run_id} by {self.worker}", [f"{self.ns}results/{run_id}"])
        if not ok:                                         # safest fallback: fresh state, copy again, push
            time.sleep(15)
            try:
                self.sync()
            except Exception:
                return False
            return self.publish(run_id, out_root, depth + 1)
        return ok

    def upload_weights(self, run_id, out_root, tag="weights"):
        import requests
        path = os.path.join(out_root, run_id, "model_ema_fp16.pt")
        if not os.path.exists(path):
            return False
        h = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json"}
        rel = requests.get(f"https://api.github.com/repos/{self.repo}/releases/tags/{tag}", headers=h, timeout=60).json()
        name = f"{self.ns.replace('/', '_')}{run_id}.pt"
        for a in rel.get("assets", []):
            if a["name"] == name:
                requests.delete(a["url"], headers=h, timeout=60)
        up = rel["upload_url"].split("{")[0]
        for i in range(4):
            try:
                with open(path, "rb") as fh:
                    r = requests.post(f"{up}?name={name}", headers={**h, "Content-Type": "application/octet-stream"},
                                      data=fh, timeout=600)
                if r.status_code in (200, 201):
                    return True
            except Exception:
                pass
            time.sleep(10 * (i + 1))
        return False


def queue_worker(a, device):
    token = os.environ.get("GH_TOKEN")
    assert token, "GH_TOKEN missing: add it in Kaggle Add-ons > Secrets (or Colab Secrets)"
    exps = a.exps.split(",")
    plan = [r for r in (RUNS_C + RUNS_B) if parse_run(r)[0] in exps]
    if a.smoke:                                   # queue test on random data, separate namespace
        plan = plan[:3]
    q = Queue(a.repo, token, a.clone, a.worker, ns=a.ns)
    data_cache, pending = {}, []
    while True:
        r, n_done, active = q.claim(plan)
        if r is None:
            print(f"[{a.worker}] queue empty: {n_done}/{len(plan)} done; still running elsewhere: {active}", flush=True)
            break
        print(f"[{a.worker}] claimed {r} ({n_done}/{len(plan)} done)", flush=True)
        exp = parse_run(r)[0]
        if exp not in data_cache:
            data_cache.clear()
            data_cache[exp] = load_all(None if a.smoke else find_data_dir(a.data), exp, device, a.smoke)
        run_one(r, data_cache[exp], a.out, device, smoke=a.smoke)
        if not q.publish(r, a.out):
            pending.append(r)
        if not q.upload_weights(r, a.out):
            print(f"[{a.worker}] weights upload failed for {r} (kept in local output)", flush=True)
    for r in pending:                                          # final safety: retry unpublished results
        print(f"[{a.worker}] retry publish {r}:", q.publish(r, a.out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true"); ap.add_argument("--full", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", default=""); ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="results"); ap.add_argument("--root", default="/kaggle/temp/wilds")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--queue", action="store_true"); ap.add_argument("--repo", default="omarzyd/esm-runs")
    ap.add_argument("--exps", default="B"); ap.add_argument("--worker", default="w")
    ap.add_argument("--clone", default="esm-runs"); ap.add_argument("--ns", default="")
    a = ap.parse_args()
    if a.prepare:
        prepare(a.out, a.root, full=a.full); return
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    runs = [r for r in a.runs.split(",") if r]
    if a.smoke:
        global CFG
        # smoke only: tiny budgets and loose filters so every code path runs on random data
        CFG.update(budget=20, probe=8, pool_k=6, grid=(4, 4), emitters=2, emitter_batch=3, pop=10,
                   mc_teacher_acc=0.0, tmin=-20.0)
        runs = runs or [f"B-{x}-s0" for x in ARMS_B]
    if a.queue:
        queue_worker(a, device); return
    data_cache = {}
    for r in runs:
        exp = parse_run(r)[0]
        if exp not in data_cache:
            data_cache[exp] = load_all(None if a.smoke else find_data_dir(a.data), exp, device, a.smoke)
        run_one(r, data_cache[exp], a.out, device, smoke=a.smoke)


if __name__ == "__main__":
    main()
