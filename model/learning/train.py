"""Supervised policy/value learning with a held-out game-level test split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model.learning.common import fresh_directory, setup, write_json
from model.learning.data import ChessDataset, collate, split_records
from model.learning.models import ModelConfig, PolicyValueNet, load_model, masked_logits, save_model


def epoch_pass(model, loader, device, optimizer=None, value_weight=1.0):
    model.train(optimizer is not None)
    totals = dict(loss=0., policy_loss=0., value_loss=0., policy_accuracy=0.)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, target, value_target, mask in loader:
            x, target = x.to(device), target.to(device)
            value_target, mask = value_target.to(device), mask.to(device)
            logits, value = model(x)
            logits = masked_logits(logits, mask)
            policy_loss = F.cross_entropy(logits, target)
            value_loss = F.mse_loss(value, value_target)
            loss = policy_loss + value_weight * value_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n = len(x)
            count += n
            for key, v in [('loss', loss.item()), ('policy_loss', policy_loss.item()),
                           ('value_loss', value_loss.item()),
                           ('policy_accuracy', (logits.argmax(-1) == target).float().mean().item())]:
                totals[key] += v * n
    return {k: v / count for k, v in totals.items()}


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.value_weight < 0:
        raise ValueError('Thông số huấn luyện không hợp lệ.')
    device = setup(args.seed, args.device, args.threads)
    source = Path(args.data)
    raw = source.read_bytes()
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    folds, manifest = split_records(records, args.seed)
    datasets = [ChessDataset(fold) for fold in folds]
    out = fresh_directory(args.out)
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=i == 0,
                          num_workers=0, collate_fn=collate,
                          generator=torch.Generator().manual_seed(args.seed))
               for i, ds in enumerate(datasets)]
    config = ModelConfig(args.model, args.width, args.hidden)
    model = PolicyValueNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    run = {
        'args': vars(args), 'device': device, 'torch': torch.__version__,
        'python': platform.python_version(), 'data_sha256': hashlib.sha256(raw).hexdigest(),
        'parameters': sum(p.numel() for p in model.parameters()), 'split': manifest,
    }
    write_json(out / 'run.json', run)
    best_loss = float('inf')
    best_epoch = 0
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        names = ['epoch'] + [f'{split}_{metric}' for split in ('train', 'validation')
                            for metric in ('loss', 'policy_loss', 'value_loss', 'policy_accuracy')]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = epoch_pass(model, loaders[0], device, optimizer, args.value_weight)
            val_metrics = epoch_pass(model, loaders[1], device, value_weight=args.value_weight)
            row = {'epoch': epoch, **{f'train_{k}': v for k, v in train_metrics.items()},
                   **{f'validation_{k}': v for k, v in val_metrics.items()}}
            writer.writerow(row)
            stream.flush()
            metadata = dict(epoch=epoch, policy_trained=True, method='supervised',
                            seed=args.seed, data_sha256=run['data_sha256'], validation=val_metrics)
            if val_metrics['loss'] < best_loss:
                best_loss, best_epoch = val_metrics['loss'], epoch
                save_model(out / 'best.pt', model, **metadata)
            print(json.dumps(row), flush=True)
        save_model(out / 'last.pt', model, **metadata)
    best, _ = load_model(out / 'best.pt', device)
    test_metrics = epoch_pass(best, loaders[2], device, value_weight=args.value_weight)
    write_json(out / 'test.json', {'best_epoch': best_epoch, 'rows': len(datasets[2]), **test_metrics})
    return {'out': str(out), 'device': device, 'best_epoch': best_epoch, 'test': test_metrics}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model', choices=['mlp', 'cnn'], default='cnn')
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--value-weight', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    print(json.dumps(train(p.parse_args()), indent=2))


if __name__ == '__main__':
    main()
