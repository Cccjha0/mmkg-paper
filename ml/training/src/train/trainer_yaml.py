import os
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ml.training.src.data.sampler import negative_sample
from ml.training.src.eval.filtered_ranking import filtered_ranking_eval, prepare_true_tails_index
from ml.training.src.utils.io import make_run_dir, append_csv, save_json, copy_file


class TrainerYAML:
    def __init__(self, model, train_triples, dev_triples, num_entities, true_tails, true_heads, cfg: dict):
        self.model = model
        self.train_triples = train_triples
        self.dev_triples = dev_triples
        self.num_entities = num_entities
        self.true_tails = true_tails
        self.true_heads = true_heads
        self.true_tails_index = prepare_true_tails_index(true_tails)
        self.cfg = cfg

        self.device = cfg["system"].get("device", "cuda")
        tr = cfg["training"]
        ev = cfg["evaluation"]
        out = cfg["output"]

        self.lr = tr.get("lr", 1e-3)
        self.batch_size = tr.get("batch_size", 1024)
        self.neg_ratio = tr.get("neg_ratio", 10)
        self.fusion_warmup_epochs = tr.get("fusion_warmup_epochs", 0)
        self.epochs = tr.get("epochs", 200)
        self.eval_every = tr.get("eval_every", 5)
        self.patience = tr.get("early_stop_patience", 10)

        self.dev_eval_limit = ev.get("dev_eval_limit", len(dev_triples))
        self.chunk_size = ev.get("chunk_size", 10000)
        self.query_batch_size = ev.get("query_batch_size", 1)

        seed = cfg["system"].get("seed", 1)
        root_dir = out.get("root_dir", "ml/artifacts/outputs")
        exp_name = out.get("exp_name", "experiment")
        self.run_dir = make_run_dir(root_dir, exp_name, seed)

        # save config snapshot for reproducibility
        save_json(os.path.join(self.run_dir, "config_merged.json"), cfg)
        # also copy original yaml files if provided
        paths = cfg.get("_config_paths", {})
        if paths.get("common"):
            copy_file(paths["common"], self.run_dir, "common.yaml")
        if paths.get("exp"):
            copy_file(paths["exp"], self.run_dir, "experiment.yaml")

        self.metrics_csv = os.path.join(self.run_dir, "metrics_seed1.csv")
        self.ckpt_path = os.path.join(self.run_dir, "best.ckpt")

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.base_use_fusion = getattr(self.model, "use_fusion", None)

    @torch.no_grad()
    def _compute_gate_stats(self, sample_size: int = 5000):
        """
        Sample some entity ids and compute gate statistics:
        - overall gate mean/std
        - has_img gate mean/std
        - no_img gate mean/std
        """
        # 如果模型没实现该接口（比如 Early Fusion），就返回空
        if not hasattr(self.model, "gate_for_entities"):
            return {}

        N = self.num_entities
        device = self.device

        # sample entity ids on device
        eids = torch.randint(0, N, (sample_size,), device=device)

        g = self.model.gate_for_entities(eids).detach().cpu()  # [S]
        has_img = self.model.has_img[eids].detach().cpu()      # [S] bool

        def mean_std(x: torch.Tensor):
            if x.numel() == 0:
                return 0.0, 0.0
            m = float(x.mean().item())
            s = float(x.std(unbiased=False).item())
            if not math.isfinite(m) or not math.isfinite(s):
                return 0.0, 0.0
            return m, s

        g_all = g
        g_img = g[has_img]
        g_noimg = g[~has_img]

        m_all, s_all = mean_std(g_all)
        m_img, s_img = mean_std(g_img)
        m_no, s_no = mean_std(g_noimg)

        return {
            "g_mean_all": m_all,
            "g_std_all": s_all,
            "g_mean_img": m_img,
            "g_std_img": s_img,
            "g_mean_noimg": m_no,
            "g_std_noimg": s_no,
            "g_frac_img_in_sample": float(has_img.float().mean().item()),
        }

    def train(self):
        best_mrr = -1.0
        bad_epochs = 0

        train_tensor = torch.tensor(self.train_triples, dtype=torch.long)
        dev_tensor = torch.tensor(self.dev_triples[: self.dev_eval_limit], dtype=torch.long)

        for epoch in range(1, self.epochs + 1):
            if self.base_use_fusion is not None and self.base_use_fusion and self.fusion_warmup_epochs > 0:
                use_fusion_now = epoch > self.fusion_warmup_epochs
                self.model.use_fusion = use_fusion_now
                if epoch == 1 or epoch == (self.fusion_warmup_epochs + 1):
                    phase = "warmup(residual-only)" if not use_fusion_now else "joint(fusion+residual)"
                    print(f"[Train] phase={phase} epoch={epoch}")

            self.model.train()

            perm = torch.randperm(train_tensor.size(0))
            total_loss = 0.0
            steps = 0

            for i in tqdm(range(0, train_tensor.size(0), self.batch_size), desc=f"epoch {epoch}"):
                idx = perm[i : i + self.batch_size]
                pos = train_tensor[idx].to(self.device)
                neg = negative_sample(pos, num_entities=self.num_entities, neg_ratio=self.neg_ratio)

                loss = self.model(pos, neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += float(loss.item())
                steps += 1

            avg_loss = total_loss / max(1, steps)
            print(f"[Train] epoch={epoch} avg_loss={avg_loss:.6f}")

            # eval
            if epoch % self.eval_every == 0:
                self.model.eval()
                metrics = filtered_ranking_eval(
                    model=self.model,
                    triples=dev_tensor,
                    true_tails=self.true_tails_index,
                    true_heads=self.true_heads,
                    num_entities=self.num_entities,
                    chunk_size=self.chunk_size,
                    query_batch_size=self.query_batch_size,
                    device=self.device,
                    ks=(1, 3, 10),
                )
                row = {
                    "epoch": epoch,
                    "avg_loss": avg_loss,
                    "mrr": metrics["mrr"],
                    "hits@1": metrics["hits@1"],
                    "hits@3": metrics["hits@3"],
                    "hits@10": metrics["hits@10"],
                }

                # gate stats (only for models that support it, e.g., Gated Fusion)
                gate_stats = self._compute_gate_stats(sample_size=5000)
                row.update(gate_stats)

                append_csv(
                    self.metrics_csv,
                    row,
                    header_order=[
                        "epoch", "avg_loss", "mrr", "hits@1", "hits@3", "hits@10",
                        "g_mean_all", "g_std_all",
                        "g_mean_img", "g_std_img",
                        "g_mean_noimg", "g_std_noimg",
                        "g_frac_img_in_sample",
                    ]
                )
                print("[Dev] " + " ".join([f"{k}={v:.6f}" for k, v in metrics.items()]))

                # print residual_scale if exists
                if hasattr(self.model, "residual_scale"):
                    rs = F.softplus(self.model.residual_scale).detach().cpu().item()
                    print(f"[Debug] residual_scale = {rs:.6f}")
                if hasattr(self.model, "use_normalized_mix") and getattr(self.model, "use_normalized_mix", False):
                    a = F.softplus(self.model.mix_fusion_raw).detach()
                    b = F.softplus(self.model.mix_residual_raw).detach()
                    denom = (a + b).clamp_min(1e-12)
                    wf = (a / denom).cpu().item()
                    wr = (b / denom).cpu().item()
                    print(f"[Debug] mix_w_fusion = {wf:.6f} mix_w_residual = {wr:.6f}")

                if metrics["mrr"] > best_mrr:
                    best_mrr = metrics["mrr"]
                    bad_epochs = 0
                    torch.save(self.model.state_dict(), self.ckpt_path)
                    print(f"[CKPT] saved best -> {self.ckpt_path}")
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        print("[EarlyStop] triggered.")
                        break

        print(f"[Done] best_dev_mrr={best_mrr:.6f} run_dir={self.run_dir}")
        return best_mrr
