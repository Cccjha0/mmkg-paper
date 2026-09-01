from __future__ import annotations

import os
import math
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ml.training.src.data.dataset_spec import MMKG_GENERAL_V1, OPENBG_LEGACY_V1
from ml.training.src.data.sampler import negative_sample
from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.data.sampler_recent import (
    bernoulli_filtered_negative_sample,
    build_relation_statistics,
)
from ml.training.src.eval.filtered_ranking import (
    filtered_ranking_eval,
    prepare_true_heads_index,
    prepare_true_tails_index,
)
from ml.training.src.utils.io import make_run_dir, append_csv, save_json, copy_file


class TrainerYAML:
    def __init__(self, model, train_triples, dev_triples, test_triples, num_entities, true_tails, true_heads, cfg: dict):
        self.model = model
        self.train_triples = train_triples
        self.dev_triples = dev_triples
        self.test_triples = test_triples
        self.num_entities = num_entities
        self.true_tails = true_tails
        self.true_heads = true_heads
        self.true_tails_index = prepare_true_tails_index(true_tails)
        self.true_heads_index = prepare_true_heads_index(true_heads)
        self.cfg = cfg
        self.entity_has_img = getattr(model, "has_img", None)
        self.entity_has_text = getattr(model, "has_text", None)
        processed_dir = cfg.get("dataset", {}).get("processed_dir")
        if processed_dir:
            if self.entity_has_img is None:
                general_img_mask = os.path.join(processed_dir, "has_img.pt")
                if os.path.exists(general_img_mask):
                    self.entity_has_img = torch.load(general_img_mask, map_location="cpu")
            if self.entity_has_text is None:
                general_text_mask = os.path.join(processed_dir, "has_text.pt")
                if os.path.exists(general_text_mask):
                    self.entity_has_text = torch.load(general_text_mask, map_location="cpu")
        if self.entity_has_img is None:
            cache_dir = cfg.get("dataset", {}).get("cache_dir")
            has_img_path = os.path.join(cache_dir, "has_img.pt") if cache_dir else None
            if has_img_path and os.path.exists(has_img_path):
                self.entity_has_img = torch.load(has_img_path, map_location="cpu")

        self.device = cfg["system"].get("device", "cuda")
        tr = cfg["training"]
        ev = cfg["evaluation"]
        out = cfg["output"]

        self.lr = tr.get("lr", 1e-3)
        self.batch_size = tr.get("batch_size", 1024)
        self.neg_ratio = tr.get("neg_ratio", 10)
        self.sampler_name = tr.get("sampler", "uniform").lower()
        if self.sampler_name not in {"uniform", "bernoulli_filtered", "none"}:
            raise ValueError(
                f"Unsupported training.sampler={self.sampler_name!r}. "
                "Expected 'uniform', 'bernoulli_filtered', or 'none'."
            )
        if self.sampler_name == "bernoulli_filtered":
            # Sampling must only use training facts.  The all-split truth maps
            # supplied to this trainer remain exclusively for filtered eval.
            self.train_true_tails, self.train_true_heads = build_true_facts(train_triples)
            self.relation_stats = build_relation_statistics(train_triples)
        self.fusion_warmup_epochs = tr.get("fusion_warmup_epochs", 0)
        self.residual_warmup_epochs = tr.get("residual_warmup_epochs", 0)
        self.epochs = tr.get("epochs", 200)
        self.eval_every = tr.get("eval_every", 5)
        patience = tr.get("early_stop_patience", 10)
        self.patience = None if patience is None else int(patience)
        self.protocol_version = cfg.get("protocol", {}).get("version", OPENBG_LEGACY_V1)
        self.termination_policy = tr.get("termination_policy")
        if self.protocol_version == MMKG_GENERAL_V1:
            if self.termination_policy not in {"dev_early_stop", "fixed_budget"}:
                raise ValueError(
                    "mmkg_general_v1 requires training.termination_policy to be "
                    "'dev_early_stop' or 'fixed_budget'."
                )
            if self.termination_policy == "fixed_budget" and self.patience is not None:
                raise ValueError("fixed_budget requires training.early_stop_patience: null")
            if self.termination_policy == "dev_early_stop" and self.patience is None:
                raise ValueError("dev_early_stop requires a finite training.early_stop_patience")
        self.device_type = torch.device(self.device).type
        self.pin_memory = bool(tr.get("pin_memory", self.device_type == "cuda"))
        self.profile_timing = bool(tr.get("profile_timing", False))
        self.profile_warmup_steps = int(tr.get("profile_warmup_steps", 2))
        self.profile_steps = int(tr.get("profile_steps", 20))
        self.profile_stop_after = bool(tr.get("profile_stop_after", False))
        if self.profile_warmup_steps < 0 or self.profile_steps <= 0:
            raise ValueError("profile_warmup_steps must be non-negative and profile_steps must be positive.")

        raw_dev_limit = ev.get("dev_eval_limit", len(dev_triples))
        raw_test_limit = ev.get("test_eval_limit", len(test_triples))
        self.dev_eval_limit = len(dev_triples) if raw_dev_limit is None else min(int(raw_dev_limit), len(dev_triples))
        self.test_eval_limit = (
            len(test_triples) if raw_test_limit is None else min(int(raw_test_limit), len(test_triples))
        )
        if self.dev_eval_limit <= 0:
            raise ValueError("evaluation.dev_eval_limit must select at least one validation triple")
        self.chunk_size = ev.get("chunk_size", 10000)
        self.query_batch_size = ev.get("query_batch_size", 1)
        self.eval_direction = ev.get("direction", "both")
        self.run_test = bool(ev.get("run_test", True))

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
        self.test_metrics_json = os.path.join(self.run_dir, "test_metrics.json")

        self.optimizer_name = str(tr.get("optimizer", "adam")).lower()
        self.optim = self._build_optimizer()
        self.base_use_fusion = getattr(self.model, "use_fusion", None)
        print(
            f"[Selection] protocol={self.protocol_version} "
            f"dev_queries={self.dev_eval_limit}/{len(self.dev_triples)} "
            f"termination={self.termination_policy or 'legacy_default'} "
            f"patience={self.patience}"
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the configured optimizer while preserving Adam as the default."""
        parameters = self.model.parameters()
        if self.optimizer_name == "adam":
            return torch.optim.Adam(parameters, lr=self.lr)
        if self.optimizer_name == "adagrad":
            return torch.optim.Adagrad(parameters, lr=self.lr)
        raise ValueError(
            f"Unsupported training.optimizer={self.optimizer_name!r}. Expected 'adam' or 'adagrad'."
        )

    def _profile_cuda_synchronize(self) -> None:
        """Synchronize only for explicitly enabled CUDA timing diagnostics."""
        if self.profile_timing and self.device_type == "cuda":
            torch.cuda.synchronize(torch.device(self.device))

    def _eval_cuda_synchronize(self) -> None:
        """Close CUDA's asynchronous timing window at evaluation boundaries."""
        if self.device_type == "cuda":
            torch.cuda.synchronize(torch.device(self.device))

    def _run_filtered_eval(self, triples: torch.LongTensor, split: str) -> dict:
        """Run the unchanged evaluator and report end-to-end wall-clock time."""
        self._eval_cuda_synchronize()
        started_at = time.perf_counter()
        metrics = filtered_ranking_eval(
            model=self.model,
            triples=triples,
            true_tails=self.true_tails_index,
            true_heads=self.true_heads_index,
            num_entities=self.num_entities,
            chunk_size=self.chunk_size,
            query_batch_size=self.query_batch_size,
            device=self.device,
            ks=(1, 3, 10),
            direction=self.eval_direction,
            entity_has_img=self.entity_has_img,
            entity_has_text=self.entity_has_text,
        )
        self._eval_cuda_synchronize()
        elapsed = time.perf_counter() - started_at
        triple_count = int(triples.shape[0])
        throughput = triple_count / elapsed if elapsed > 0.0 else float("inf")
        print(
            f"[EvalPerf] split={split} triples={triple_count} "
            f"direction={self.eval_direction} elapsed={elapsed:.3f}s "
            f"throughput={throughput:.2f} triples/s"
        )
        return metrics

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfer a CPU batch with optional pinned-memory async H2D."""
        if tensor.device.type != "cpu" or self.device_type == "cpu":
            return tensor.to(self.device)
        non_blocking = False
        if self.pin_memory and self.device_type == "cuda":
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()
            non_blocking = True
        return tensor.to(self.device, non_blocking=non_blocking)

    @staticmethod
    def _print_perf_summary(totals: dict[str, float], measured_steps: int, warmup_steps: int) -> None:
        if measured_steps <= 0:
            print("[Perf] no batches were measured")
            return
        averages_ms = {key: 1000.0 * value / measured_steps for key, value in totals.items()}
        total_ms = sum(averages_ms.values())
        print(f"[Perf] measured_batches={measured_steps} warmup_batches={warmup_steps}")
        for key in ("batch_indexing", "negative_sampling", "h2d_transfer", "gpu_step"):
            share = 100.0 * averages_ms[key] / total_ms if total_ms > 0 else 0.0
            print(f"[Perf] {key:<18} = {averages_ms[key]:9.3f} ms ({share:5.1f}%)")

    def _sample_negatives(self, pos: torch.LongTensor) -> torch.LongTensor | None:
        """Use legacy sampling by default, or explicit recent-baseline sampling."""
        if self.sampler_name == "none":
            return None
        if self.sampler_name == "bernoulli_filtered":
            return bernoulli_filtered_negative_sample(
                pos=pos,
                num_entities=self.num_entities,
                true_heads=self.train_true_heads,
                true_tails=self.train_true_tails,
                relation_stats=self.relation_stats,
                neg_ratio=self.neg_ratio,
            )
        return negative_sample(pos, num_entities=self.num_entities, neg_ratio=self.neg_ratio)

    def _compute_loss(self, pos: torch.LongTensor, neg: torch.LongTensor | None) -> torch.Tensor:
        """Legacy model-loss contract used only by the standard engine."""
        if neg is None:
            raise RuntimeError("The standard trainer requires sampled negative triples.")
        return self.model(pos, neg)

    def _train_step(self, pos: torch.LongTensor, neg: torch.LongTensor | None) -> dict[str, float]:
        """Run one optimizer step and return scalar diagnostics.

        Recent multi-optimizer engines override this hook.  The standard path
        retains its original model(pos, neg), backward, and Adam semantics.
        """
        loss = self._compute_loss(pos, neg)
        self.optim.zero_grad()
        loss.backward()
        grad_stats = self._compute_grad_group_stats()
        self.optim.step()
        return {"loss": float(loss.item()), **grad_stats}

    def _on_epoch_start(self, epoch: int) -> None:
        """Optional model hook; legacy models do not implement it."""
        on_epoch_start = getattr(self.model, "on_epoch_start", None)
        if on_epoch_start is not None:
            on_epoch_start(epoch)

    @staticmethod
    def _grad_norm_sq(param: torch.nn.Parameter) -> float:
        if param.grad is None:
            return 0.0
        g = param.grad.detach()
        if g.is_sparse:
            g = g.coalesce().values()
        return float(torch.sum(g * g).item())

    def _compute_grad_group_stats(self) -> dict:
        residual_sq = 0.0
        fusion_sq = 0.0
        projection_sq = 0.0
        missing_text_sq = 0.0
        missing_image_sq = 0.0

        for name, param in self.model.named_parameters():
            sq = self._grad_norm_sq(param)
            if sq <= 0.0:
                continue
            if name.endswith("t_missing"):
                missing_text_sq += sq
            if name.endswith("v_missing"):
                missing_image_sq += sq

            if name.startswith("entity_residual") or name == "residual_scale":
                residual_sq += sq
            elif (
                name.startswith("fusion")
                or name.startswith("t_adapter")
                or name.startswith("v_adapter")
                or name.startswith("t_missing")
                or name.startswith("v_missing")
            ):
                fusion_sq += sq
            elif name.startswith("text_proj") or name.startswith("img_proj"):
                projection_sq += sq

        return {
            "grad_residual": math.sqrt(residual_sq),
            "grad_fusion": math.sqrt(fusion_sq),
            "grad_projection": math.sqrt(projection_sq),
            "grad_missing_text": math.sqrt(missing_text_sq),
            "grad_missing_image": math.sqrt(missing_image_sq),
        }

    def _compute_scalar_stats(self) -> dict:
        out = {
            "residual_scale_value": 0.0,
            "mix_w_fusion": 0.0,
            "mix_w_residual": 0.0,
        }

        if hasattr(self.model, "residual_scale"):
            out["residual_scale_value"] = float(F.softplus(self.model.residual_scale).detach().cpu().item())

        if hasattr(self.model, "use_normalized_mix") and getattr(self.model, "use_normalized_mix", False):
            a = F.softplus(self.model.mix_fusion_raw).detach()
            b = F.softplus(self.model.mix_residual_raw).detach()
            denom = (a + b).clamp_min(1e-12)
            out["mix_w_fusion"] = float((a / denom).cpu().item())
            out["mix_w_residual"] = float((b / denom).cpu().item())

        return out

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
        global_step = 0
        profile_measured_steps = 0
        profile_stopped_early = False
        profile_totals = {
            "batch_indexing": 0.0,
            "negative_sampling": 0.0,
            "h2d_transfer": 0.0,
            "gpu_step": 0.0,
        }

        train_tensor = torch.tensor(self.train_triples, dtype=torch.long)
        dev_tensor = torch.tensor(self.dev_triples[: self.dev_eval_limit], dtype=torch.long)
        test_tensor = torch.tensor(self.test_triples[: self.test_eval_limit], dtype=torch.long)

        for epoch in range(1, self.epochs + 1):
            if self.base_use_fusion is not None and self.base_use_fusion and self.fusion_warmup_epochs > 0:
                use_fusion_now = epoch > self.fusion_warmup_epochs
                self.model.use_fusion = use_fusion_now
                if epoch == 1 or epoch == (self.fusion_warmup_epochs + 1):
                    phase = "warmup(residual-only)" if not use_fusion_now else "joint(fusion+residual)"
                    print(f"[Train] phase={phase} epoch={epoch}")

            if hasattr(self.model, "enable_residual") and self.residual_warmup_epochs > 0:
                enable_residual_now = epoch > self.residual_warmup_epochs
                self.model.enable_residual = enable_residual_now
                if epoch == 1 or epoch == (self.residual_warmup_epochs + 1):
                    phase = "warmup(fusion-only)" if not enable_residual_now else "joint(fusion+residual)"
                    print(f"[Train] residual_phase={phase} epoch={epoch}")

            self.model.train()
            self._on_epoch_start(epoch)

            perm = torch.randperm(train_tensor.size(0))
            epoch_totals: dict[str, float] = {}
            steps = 0
            epoch_grad_stats = {
                "grad_residual": 0.0,
                "grad_fusion": 0.0,
                "grad_projection": 0.0,
                "grad_missing_text": 0.0,
                "grad_missing_image": 0.0,
            }

            for i in tqdm(range(0, train_tensor.size(0), self.batch_size), desc=f"epoch {epoch}"):
                should_measure = (
                    self.profile_timing
                    and global_step >= self.profile_warmup_steps
                    and profile_measured_steps < self.profile_steps
                )
                if should_measure:
                    self._profile_cuda_synchronize()
                indexing_start = time.perf_counter()
                idx = perm[i : i + self.batch_size]
                pos_cpu = train_tensor[idx]
                indexing_elapsed = time.perf_counter() - indexing_start

                # Recent baselines sample on CPU before H2D. This removes the
                # former positive CPU -> GPU -> CPU round trip. Legacy uniform
                # sampling stays on its original device/RNG path.
                if self.sampler_name == "bernoulli_filtered":
                    sampling_start = time.perf_counter()
                    neg_cpu = self._sample_negatives(pos_cpu)
                    sampling_elapsed = time.perf_counter() - sampling_start

                    transfer_start = time.perf_counter()
                    pos = self._to_device(pos_cpu)
                    neg = self._to_device(neg_cpu)
                    if should_measure:
                        self._profile_cuda_synchronize()
                    transfer_elapsed = time.perf_counter() - transfer_start
                else:
                    transfer_start = time.perf_counter()
                    pos = self._to_device(pos_cpu)
                    if should_measure:
                        self._profile_cuda_synchronize()
                    transfer_elapsed = time.perf_counter() - transfer_start

                    if should_measure:
                        self._profile_cuda_synchronize()
                    sampling_start = time.perf_counter()
                    neg = self._sample_negatives(pos)
                    if should_measure:
                        self._profile_cuda_synchronize()
                    sampling_elapsed = time.perf_counter() - sampling_start

                if should_measure:
                    self._profile_cuda_synchronize()
                gpu_step_start = time.perf_counter()
                step_stats = self._train_step(pos, neg)
                if should_measure:
                    self._profile_cuda_synchronize()
                gpu_step_elapsed = time.perf_counter() - gpu_step_start

                if should_measure:
                    profile_totals["batch_indexing"] += indexing_elapsed
                    profile_totals["negative_sampling"] += sampling_elapsed
                    profile_totals["h2d_transfer"] += transfer_elapsed
                    profile_totals["gpu_step"] += gpu_step_elapsed
                    profile_measured_steps += 1
                global_step += 1
                for key, value in step_stats.items():
                    if key.startswith("grad_"):
                        continue
                    epoch_totals[key] = epoch_totals.get(key, 0.0) + float(value)
                epoch_grad_stats = {
                    key: float(step_stats.get(key, 0.0))
                    for key in epoch_grad_stats
                }
                steps += 1
                if (
                    self.profile_timing
                    and self.profile_stop_after
                    and profile_measured_steps >= self.profile_steps
                ):
                    profile_stopped_early = True
                    break

            epoch_averages = {
                key: value / max(1, steps)
                for key, value in epoch_totals.items()
            }
            avg_loss = epoch_averages.get("loss", 0.0)
            detail = " ".join(
                f"{key}={value:.6f}"
                for key, value in epoch_averages.items()
                if key != "loss"
            )
            print(f"[Train] epoch={epoch} avg_loss={avg_loss:.6f}" + (f" {detail}" if detail else ""))

            if profile_stopped_early:
                break

            # eval
            if epoch % self.eval_every == 0:
                self.model.eval()
                metrics = self._run_filtered_eval(dev_tensor, split="dev")
                row = {
                    "epoch": epoch,
                    "avg_loss": avg_loss,
                    "mrr": metrics["mrr"],
                    "hits@1": metrics["hits@1"],
                    "hits@3": metrics["hits@3"],
                    "hits@10": metrics["hits@10"],
                }
                row.update({key: value for key, value in epoch_averages.items() if key != "loss"})

                # gate stats (only for models that support it, e.g., Gated Fusion)
                gate_stats = self._compute_gate_stats(sample_size=5000)
                scalar_stats = self._compute_scalar_stats()
                row.update(gate_stats)
                row.update(epoch_grad_stats)
                row.update(scalar_stats)

                append_csv(
                    self.metrics_csv,
                    row,
                    header_order=[
                        "epoch", "avg_loss", "mrr", "hits@1", "hits@3", "hits@10",
                        "kgc_loss", "adversarial_loss", "generator_loss", "gradient_penalty",
                        "generator_grad_norm",
                        "grad_residual", "grad_fusion", "grad_projection",
                        "grad_missing_text", "grad_missing_image",
                        "residual_scale_value", "mix_w_fusion", "mix_w_residual",
                        "g_mean_all", "g_std_all",
                        "g_mean_img", "g_std_img",
                        "g_mean_noimg", "g_std_noimg",
                        "g_frac_img_in_sample",
                    ]
                )
                print("[Dev] " + " ".join([f"{k}={v:.6f}" for k, v in metrics.items()]))

                print(
                    "[Diag] "
                    f"grad_residual={epoch_grad_stats['grad_residual']:.6f} "
                    f"grad_fusion={epoch_grad_stats['grad_fusion']:.6f} "
                    f"grad_projection={epoch_grad_stats['grad_projection']:.6f} "
                    f"grad_missing_text={epoch_grad_stats['grad_missing_text']:.6f} "
                    f"grad_missing_image={epoch_grad_stats['grad_missing_image']:.6f}"
                )
                if hasattr(self.model, "residual_scale"):
                    print(f"[Diag] residual_scale = {scalar_stats['residual_scale_value']:.6f}")
                if hasattr(self.model, "use_normalized_mix") and getattr(self.model, "use_normalized_mix", False):
                    print(
                        f"[Diag] mix_w_fusion = {scalar_stats['mix_w_fusion']:.6f} "
                        f"mix_w_residual = {scalar_stats['mix_w_residual']:.6f}"
                    )

                if metrics["mrr"] > best_mrr:
                    best_mrr = metrics["mrr"]
                    bad_epochs = 0
                    torch.save(self.model.state_dict(), self.ckpt_path)
                    print(f"[CKPT] saved best -> {self.ckpt_path}")
                else:
                    bad_epochs += 1
                    if self.patience is not None and bad_epochs >= self.patience:
                        print("[EarlyStop] triggered.")
                        break

        if self.profile_timing:
            self._print_perf_summary(
                profile_totals,
                profile_measured_steps,
                min(global_step, self.profile_warmup_steps),
            )

        test_metrics = None
        if profile_stopped_early:
            print("[Test] skipped: profiling run stopped after the requested measured batches.")
        elif not self.run_test:
            print("[Test] skipped: evaluation.run_test=false (Dev-only configuration selection).")
        elif os.path.exists(self.ckpt_path) and test_tensor.numel() > 0:
            state = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            test_metrics = self._run_filtered_eval(test_tensor, split="test")
            save_json(self.test_metrics_json, test_metrics)
            print("[Test] " + " ".join([f"{k}={v:.6f}" for k, v in test_metrics.items()]))
            print(f"[Test] saved -> {self.test_metrics_json}")
        elif test_tensor.numel() == 0:
            print("[Test] skipped: no labeled 3-column test triples were provided.")

        print(f"[Done] best_dev_mrr={best_mrr:.6f} run_dir={self.run_dir}")
        return {"best_dev_mrr": best_mrr, "test_metrics": test_metrics, "run_dir": self.run_dir}
