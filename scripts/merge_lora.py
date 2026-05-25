"""
Merge LoRA deltas into base Conformer weights so WER scripts can load the result
as a plain ConformerASR checkpoint.

Reads:  data/models/conformer_v2/best_model_v2_80M.pt   (pre-FT base)
        data/models/conformer_v2/lora_only.pt           (LoRA A/B per block)
Writes: data/models/conformer_v2/best_lora_merged.pt    (drop-in ConformerASR checkpoint)
"""
import os, torch, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "data/models/conformer_v2/best_model_v2_80M.pt"))
    ap.add_argument("--lora", default=os.path.join(ROOT, "data/models/conformer_v2/lora_only.pt"))
    ap.add_argument("--out",  default=os.path.join(ROOT, "data/models/conformer_v2/best_lora_merged.pt"))
    args = ap.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    lora = torch.load(args.lora, map_location="cpu", weights_only=False)

    state = base["model_state"]
    lora_state = lora["lora_state"]
    cfg = lora.get("config", base["config"])
    rank = lora["lora"]["rank"]
    alpha = lora["lora"]["alpha"]
    scale = alpha / rank
    n_layers = cfg["n_layers"]

    print(f"LoRA rank={rank} alpha={alpha} scale={scale:.3f}  n_layers={n_layers}")

    for i in range(n_layers):
        A_qkv = lora_state[f"layers.{i}.attn.A_qkv"]
        B_qkv = lora_state[f"layers.{i}.attn.B_qkv"]
        A_o   = lora_state[f"layers.{i}.attn.A_o"]
        B_o   = lora_state[f"layers.{i}.attn.B_o"]
        dW_in = (B_qkv @ A_qkv) * scale
        dW_o  = (B_o   @ A_o)   * scale
        in_key  = f"layers.{i}.attn.in_proj_weight"
        out_key = f"layers.{i}.attn.out_proj.weight"
        state[in_key]  = state[in_key]  + dW_in
        state[out_key] = state[out_key] + dW_o
        print(f"  layer {i}: merged in_proj |dW|={dW_in.norm().item():.4f}  out_proj |dW|={dW_o.norm().item():.4f}")

    torch.save({"model_state": state, "config": cfg,
                "merged_from": {"base": os.path.basename(args.base), "lora": os.path.basename(args.lora)},
                "lora_meta": lora["lora"], "epoch": lora.get("epoch"), "val_loss": lora.get("val_loss")},
               args.out)
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
