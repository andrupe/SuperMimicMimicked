"""Export a trained QuadcopterPolicy checkpoint to C header + inference code for Crazyflie 2.1 (STM32F405)."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch


def format_float_array(name: str, tensor: torch.Tensor) -> str:
  flat = tensor.detach().cpu().float().contiguous().view(-1).tolist()
  vals = ", ".join(f"{v:.7f}f" for v in flat)
  return f"static const float {name}[{len(flat)}] = {{{vals}}};\n"


def export_checkpoint_to_c(ckpt_path: Path, out_header: Path) -> None:
  ckpt = torch.load(str(ckpt_path), map_location="cpu")
  actor_sd = ckpt["actor"]
  hidden_dims = tuple(ckpt["actor_hidden_dims"])
  obs_dim = int(actor_sd["mlp.0.weight"].shape[1])

  lines = [
    "// Auto-generated Crazyflie 2.1 policy weights for STM32F405\n",
    f"// Source checkpoint: {ckpt_path}\n",
    f"// Architecture: obs({obs_dim}) -> {hidden_dims} (Linear + LayerNorm + GELU) -> Linear(4) -> tanh\n",
    "#ifndef SUPERMIMIC_POLICY_H\n",
    "#define SUPERMIMIC_POLICY_H\n\n",
    "#include <math.h>\n\n",
    f"#define SM_OBS_DIM {obs_dim}\n",
    "#define SM_FRAME_DIM 23\n",
    f"#define SM_HIST_LEN {max(1, obs_dim // 23)}\n",
    "#define SM_ACT_DIM 4\n",
    "#define SM_THRUST_OFFSET 0.0809325f\n",
    "#define SM_THRUST_SCALE  0.0732675f\n\n",
  ]

  layer_idx = 0
  # Each hidden block in build_mlp is: 0=Linear, 1=LayerNorm, 2=GELU (3 modules)
  for h_i, h_dim in enumerate(hidden_dims):
    w_key = f"mlp.{layer_idx}.weight"
    b_key = f"mlp.{layer_idx}.bias"
    ln_w_key = f"mlp.{layer_idx + 1}.weight"
    ln_b_key = f"mlp.{layer_idx + 1}.bias"
    lines.append(format_float_array(f"sm_w{h_i}", actor_sd[w_key]))
    lines.append(format_float_array(f"sm_b{h_i}", actor_sd[b_key]))
    lines.append(format_float_array(f"sm_ln_w{h_i}", actor_sd[ln_w_key]))
    lines.append(format_float_array(f"sm_ln_b{h_i}", actor_sd[ln_b_key]))
    layer_idx += 3

  lines.append(format_float_array("sm_w_out", actor_sd[f"mlp.{layer_idx}.weight"]))
  lines.append(format_float_array("sm_b_out", actor_sd[f"mlp.{layer_idx}.bias"]))

  lines.append("""
static inline float sm_gelu(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

static inline void sm_linear_ln_gelu(
    const float *in, int in_dim,
    const float *w, const float *b,
    const float *ln_w, const float *ln_b,
    float *out, int out_dim) {
  float mean = 0.0f;
  for (int i = 0; i < out_dim; ++i) {
    float acc = b[i];
    const float *row = w + i * in_dim;
    for (int j = 0; j < in_dim; ++j) acc += row[j] * in[j];
    out[i] = acc;
    mean += acc;
  }
  mean /= (float)out_dim;
  float var = 0.0f;
  for (int i = 0; i < out_dim; ++i) {
    float d = out[i] - mean;
    var += d * d;
  }
  float inv_std = 1.0f / sqrtf(var / (float)out_dim + 1e-5f);
  for (int i = 0; i < out_dim; ++i) {
    float norm = (out[i] - mean) * inv_std * ln_w[i] + ln_b[i];
    out[i] = sm_gelu(norm);
  }
}
""")

  buf_decls = "\n".join(f"  static float buf{idx}[{h}];" for idx, h in enumerate(hidden_dims))
  layer_calls = []
  in_sym, in_dim = "obs", "SM_OBS_DIM"
  for idx, h in enumerate(hidden_dims):
    out_sym = f"buf{idx}"
    layer_calls.append(
      f"  sm_linear_ln_gelu({in_sym}, {in_dim}, sm_w{idx}, sm_b{idx}, sm_ln_w{idx}, sm_ln_b{idx}, {out_sym}, {h});"
    )
    in_sym, in_dim = out_sym, str(h)
  calls_str = "\n".join(layer_calls)
  last_h = hidden_dims[-1]
  lines.append(f"""
static inline void sm_policy_forward_with_raw(
    const float obs[SM_OBS_DIM], float thrust_n[SM_ACT_DIM], float raw_act[SM_ACT_DIM]) {{
{buf_decls}
{calls_str}
  for (int i = 0; i < SM_ACT_DIM; ++i) {{
    float acc = sm_b_out[i];
    const float *row = sm_w_out + i * {last_h};
    for (int j = 0; j < {last_h}; ++j) acc += row[j] * {in_sym}[j];
    float a = tanhf(acc);
    raw_act[i] = a;
    float f = SM_THRUST_OFFSET + SM_THRUST_SCALE * a;
    if (f < 0.0f) f = 0.0f;
    if (f > 0.1542f) f = 0.1542f;
    thrust_n[i] = f;
  }}
}}

static inline void sm_policy_forward(const float obs[SM_OBS_DIM], float thrust_n[SM_ACT_DIM]) {{
  float raw_act[SM_ACT_DIM];
  sm_policy_forward_with_raw(obs, thrust_n, raw_act);
}}
""")

  lines.append("\n#endif // SUPERMIMIC_POLICY_H\n")
  out_header.parent.mkdir(parents=True, exist_ok=True)
  out_header.write_text("".join(lines))
  print(f"[EXPORT] Wrote C header to {out_header} (obs_dim={obs_dim}, hidden={hidden_dims})")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--out", type=Path, default=Path("crazyflie_nn/firmware/supermimic_policy.h"))
  args = p.parse_args()
  export_checkpoint_to_c(args.checkpoint, args.out)


if __name__ == "__main__":
  main()
