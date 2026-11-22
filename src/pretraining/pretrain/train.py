"""Supervised pre-training: policy (played move and/or Stockfish best move) + value (game result blended with
Stockfish evaluation).

    python -m pretrain.train --shards data/labeled --out runs/pre1 --epochs 4 --batch 512 --workers 6
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pretrain.dataset import PositionStream, collate
from pretrain.project import load_project


def losses(net, batch, device, args):
    state = batch["state"].to(device, non_blocking=True).float()
    legal = batch["legal"].to(device).long()
    pad = legal < 0
    logits, value = net(state)
    chosen = logits.gather(1, legal.clamp(min=0)).float().masked_fill(pad, float("-inf"))   # legal logits only
    target, sf_t = batch["target"].to(device), batch["sf_target"].to(device)
    pol_w = batch["pol_w"].to(device)
    ce = F.cross_entropy(chosen, target, reduction="none")
    pol = (ce * pol_w).sum() / pol_w.sum().clamp(min=1)
    has_sf = (sf_t >= 0).float()
    ce_sf = F.cross_entropy(chosen, sf_t.clamp(min=0), reduction="none")
    pol_sf = (ce_sf * has_sf).sum() / has_sf.sum().clamp(min=1)
    v_target = batch["value"].to(device)
    val = F.mse_loss(value.float().reshape(-1), v_target)
    total = args.w_played * pol + args.w_sf * pol_sf + args.w_value * val
    return total, {"pol": pol.detach(), "pol_sf": pol_sf.detach(), "val": val.detach(), "chosen": chosen.detach(),
                   "target": target, "pol_w": pol_w, "value_pred": value.detach().float().reshape(-1),
                   "z": batch["z"].to(device)}


@torch.no_grad()
def evaluate(net, loader, device, args, max_batches):
    net.eval()
    n = top1 = top3 = 0.0
    sums = {"pol": 0.0, "val": 0.0, "pol_sf": 0.0}
    sign_hit = sign_n = 0.0
    batches = 0
    for batch in loader:
        _, m = losses(net, batch, device, args)
        w = m["pol_w"] > 0
        k = min(3, m["chosen"].shape[1])
        top = m["chosen"].topk(k, dim=1).indices
        hit = top == m["target"][:, None]
        top1 += hit[:, 0][w].sum().item()
        top3 += hit.any(1)[w].sum().item()
        n += w.sum().item()
        nz = m["z"] != 0
        sign_hit += ((m["value_pred"] > 0) == (m["z"] > 0))[nz].sum().item()
        sign_n += nz.sum().item()
        for key in sums:
            sums[key] += m[key].item()
        batches += 1
        if batches >= max_batches:
            break
    net.train()
    d = max(batches, 1)
    return {"top1": top1 / max(n, 1), "top3": top3 / max(n, 1), "pol": sums["pol"] / d, "val": sums["val"] / d,
            "pol_sf": sums["pol_sf"] / d, "result_sign_acc": sign_hit / max(sign_n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demo", action="store_true", help="use the bundled stand-in encoding/model (smoke test)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-pct", type=int, default=3)
    ap.add_argument("--policy-min-elo", type=int, default=0, help="only learn moves of players at/above this Elo")
    ap.add_argument("--sf-value-weight", type=float, default=0.7, help="value target = (1-w)*result + w*tanh(cp/543)")
    ap.add_argument("--w-played", type=float, default=1.0)
    ap.add_argument("--w-sf", type=float, default=0.0, help="weight of CE towards Stockfish's best move")
    ap.add_argument("--w-value", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (0 = full schedule)")
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    proj = load_project(args.demo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = proj.PolicyValueNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    mk = lambda split, **kw: PositionStream(args.shards, split, args.val_pct, demo=args.demo,
                                            policy_min_elo=args.policy_min_elo,
                                            sf_value_weight=args.sf_value_weight, seed=args.seed, **kw)
    train_ds, val_ds = mk("train"), mk("val", shuffle=False, shuffle_buffer=1)
    n_train = train_ds.count_positions()
    steps_per_epoch = math.ceil(n_train / args.batch)
    total = args.max_steps or steps_per_epoch * args.epochs
    print(f"train positions: {n_train}  steps/epoch: {steps_per_epoch}  total steps: {total}  device: {device}")
    if n_train == 0:
        raise SystemExit("no training positions (check filters / --val-pct)")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup) / max(1, total - args.warmup)))))

    os.makedirs(args.out, exist_ok=True)
    mkloader = lambda ds, shuffle_workers: DataLoader(ds, batch_size=args.batch, num_workers=shuffle_workers,
                                                      collate_fn=collate, pin_memory=device == "cuda")
    amp = device == "cuda"
    step, best, t0, seen = 0, float("inf"), time.time(), 0

    def save(name, metrics):
        torch.save({"format": 3, "algorithm": "pretrain", "name": "pretrain",
                    "encoding": {"action_size": proj.ACTION_SIZE, "input_channels": proj.INPUT_CHANNELS},
                    "net": net.state_dict(), "opt": opt.state_dict(),
                    "extra": {"pretrain": {"step": step, "metrics": metrics, "args": vars(args)}}},
                   os.path.join(args.out, name))

    net.train()
    for epoch in range(args.epochs):
        for batch in mkloader(train_ds.set_epoch(epoch), args.workers):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                loss, m = losses(net, batch, device, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step, seen = step + 1, seen + len(batch["target"])
            if step % args.log_every == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} pol {m['pol'].item():.3f} val {m['val'].item():.4f} "
                      f"lr {lr_at(step):.2e} {seen / (time.time() - t0):.0f} pos/s", flush=True)
            if step % args.eval_every == 0 or step >= total:
                metrics = evaluate(net, mkloader(val_ds, 0), device, args, args.val_batches)
                score = metrics["pol"] + metrics["val"]
                print(f"[val] step {step}: " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()), flush=True)
                save("last.pt", metrics)
                if score < best:
                    best = score
                    save("best.pt", metrics)
            if step >= total:
                break
        if step >= total:
            break
    print(f"done in {time.time() - t0:.0f}s; best val (pol+val) {best:.4f}; skipped samples: {train_ds.skipped}")


if __name__ == "__main__":
    main()
