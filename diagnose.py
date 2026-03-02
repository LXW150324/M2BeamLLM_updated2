"""
Diagnostic script: identify why performance is capped at 32%.
Run from M2BeamLLM_updated directory: python diagnose.py
"""
import os, sys, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import get_config

cfg = get_config()

print("=" * 60)
print("  M²BeamLLM Diagnostic Report")
print("=" * 60)

# ---------------------------------------------------------------
# 1. Check if encoder checkpoint exists and loads
# ---------------------------------------------------------------
ckpt_path = os.path.join(cfg.train.checkpoint_dir, "encoder_pretrained.pt")
print(f"\n[1] Encoder checkpoint: {ckpt_path}")
if os.path.isfile(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keys = list(ckpt["model_state_dict"].keys())
    print(f"  ✓ Found! {len(keys)} keys, epoch={ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', '?')}")
    print(f"  First 5 keys: {keys[:5]}")

    # Check key compatibility with MultimodalEncoder
    from models.encoders import MultimodalEncoder

    enc = MultimodalEncoder(64, pretrained=False)
    enc_keys = set(enc.state_dict().keys())
    ckpt_keys = set(keys)
    matched = enc_keys & ckpt_keys
    missing = enc_keys - ckpt_keys
    unexpected = ckpt_keys - enc_keys
    print(f"  Matched: {len(matched)}/{len(enc_keys)} keys")
    if missing: print(f"  ✗ MISSING from checkpoint: {list(missing)[:5]}...")
    if unexpected: print(f"  Extra keys in checkpoint: {list(unexpected)[:5]}...")
else:
    print(f"  ✗ NOT FOUND! Encoder pretraining weights are NOT being loaded!")
    print(f"  → Phase 2 uses random FC heads + ImageNet backbones only")

# ---------------------------------------------------------------
# 2. Beam distribution analysis
# ---------------------------------------------------------------
print(f"\n[2] Beam distribution analysis")
train_beams = np.load(os.path.join(cfg.data.data_root, "train_beams.npy"))
test_beams = np.load(os.path.join(cfg.data.data_root, "test_beams.npy"))
print(f"  Train beams shape: {train_beams.shape}, dtype: {train_beams.dtype}")
print(f"  Beam range: [{train_beams.min()}, {train_beams.max()}]")

# Distribution
from collections import Counter

all_train = train_beams.flatten()
counts = Counter(all_train)
total = len(all_train)
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
print(f"  Unique beams used: {len(counts)}/64")
print(f"  Top-5 most common beams:")
for beam, cnt in sorted_counts[:5]:
    print(f"    Beam {beam}: {cnt} ({100 * cnt / total:.1f}%)")
print(f"  Bottom-5 rarest beams:")
for beam, cnt in sorted_counts[-5:]:
    print(f"    Beam {beam}: {cnt} ({100 * cnt / total:.1f}%)")

# ---------------------------------------------------------------
# 3. "Repeat last beam" baseline
# ---------------------------------------------------------------
print(f"\n[3] 'Repeat last beam' baseline (predict future = last historical)")
H, T = 8, 5
correct_per_step = []
for t_future in range(T):
    last_hist = test_beams[:, H - 1]  # last historical beam
    true_future = test_beams[:, H + t_future]  # actual future beam
    acc = (last_hist == true_future).mean()
    correct_per_step.append(acc)
    print(f"  Step {t_future + 1}: {100 * acc:.1f}%")
mean_baseline = np.mean(correct_per_step)
print(f"  Mean 'repeat last': {100 * mean_baseline:.1f}%")

# Also check "most common beam" baseline
most_common_beam = sorted_counts[0][0]
test_future = test_beams[:, H:H + T]
majority_acc = (test_future == most_common_beam).mean()
print(f"  'Always predict beam {most_common_beam}' baseline: {100 * majority_acc:.1f}%")

# ---------------------------------------------------------------
# 4. Beam temporal autocorrelation
# ---------------------------------------------------------------
print(f"\n[4] Beam temporal autocorrelation")
for lag in [1, 2, 3, 5]:
    same = 0
    total_pairs = 0
    for i in range(test_beams.shape[1] - lag):
        same += (test_beams[:, i] == test_beams[:, i + lag]).sum()
        total_pairs += test_beams.shape[0]
    print(f"  Lag {lag}: {100 * same / total_pairs:.1f}% same beam")

# ---------------------------------------------------------------
# 5. Feature collapse check (if checkpoint exists)
# ---------------------------------------------------------------
print(f"\n[5] Checking for feature collapse...")
best_ckpt = os.path.join(cfg.train.checkpoint_dir, "best_standard_gpt2.pt")
if os.path.isfile(best_ckpt):
    from models.m2beamllm import M2BeamLLM
    from utils.dataset import DeepSenseDataset
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    model = M2BeamLLM(
        feature_dim=64, num_beams=64, llm_name="gpt2",
        llm_hidden_dim=768, num_unfrozen_layers=2,
        temperature=0.07, num_attention_heads=4,
        ffn_hidden_dim=256, fusion_dropout=0.1, T=5, H=8,
    ).to(device)
    ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    # Load one batch
    test_ds = DeepSenseDataset(cfg.data.data_root, "test", H=8, T=5)
    loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    with torch.no_grad():
        raw = model.encoder(
            batch["images"], batch["radars"], batch["lidars"], batch["gps"]
        )

    # Check cosine similarity between modalities
    from torch.nn.functional import normalize

    mods = list(raw.keys())
    print(f"  Modality feature stats (first batch):")
    for m in mods:
        feat = raw[m]  # (B, H, 64)
        print(f"    {m:>6s}: mean={feat.mean():.4f}, std={feat.std():.4f}, "
              f"norm={feat.norm(dim=-1).mean():.4f}")

    print(f"  Cross-modal cosine similarity:")
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            a = normalize(raw[mods[i]], dim=-1)
            b = normalize(raw[mods[j]], dim=-1)
            sim = (a * b).sum(dim=-1).mean()
            print(f"    {mods[i]:>6s} ↔ {mods[j]:<6s}: {sim:.4f}")
else:
    print(f"  ✗ No trained model checkpoint found at {best_ckpt}")

print(f"\n{'=' * 60}")
print(f"  Diagnostic Complete")
print(f"{'=' * 60}")