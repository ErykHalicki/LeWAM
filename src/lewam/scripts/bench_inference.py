"""
Benchmark single forward pass: VJEPA2 encode + VLM encode + DiT forward.

Usage:
    python src/lelewam/scripts/bench_inference.py
    python src/lelewam/scripts/bench_inference.py --config configs/model/baby.yaml --cameras 1
    python src/lelewam/scripts/bench_inference.py --config configs/model/base.yaml --cameras 2 --runs 10
    python src/lelewam/scripts/bench_inference.py --no-compile
"""
import argparse
import os
import time

import torch
import yaml

from lewam.models.lewam import LeWAM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/base.yaml")
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ode-steps", type=int, default=0,
                        help="If >0, run ode_solve instead of a single forward pass")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    patch_h = 14
    patch_w = 14 * args.cameras

    model = LeWAM(
        frame_latent_h=patch_h,
        frame_latent_w=patch_w,
        fps=5.0,
        action_fps=30.0,
        norm_stats=LeWAM._dummy_norm_stats(cfg.get("action_dim", 6), cfg.get("state_dim", 6)),
        **cfg,
    )
    model.eval()
    model = model.to(device)

    total_params = model.count_params(trainable_only=False)
    trainable_params = model.count_params(trainable_only=True)
    N_ctx = model.N_ctx
    N_fut = model.N_fut
    N_act = model.N_act
    num_context_frames = model.num_context_frames
    num_future_frames = model.num_future_frames
    action_dim = model.action_dim
    action_horizon = model.action_horizon
    action_fps = model.action_fps
    has_vlm = model.vlm_encoder is not None

    use_compile = not args.no_compile and device == "cuda"
    if use_compile:
        cache_dir = os.path.join(
            os.environ.get("LE_WAM_ROOT", "."), ".cache", "torch_compile"
        )
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        print(f"Compiling model (cache: {cache_dir})...")
        model = torch.compile(model, dynamic=True)  # type: ignore[assignment]

    print(f"Config:   {args.config}")
    print(f"Device:   {device}")
    print(f"Cameras:  {args.cameras}")
    print(f"Compile:  {use_compile}")
    print(f"Params:   {total_params}M total, {trainable_params}M trainable")
    print(f"Tokens:   N_ctx={N_ctx}, N_fut={N_fut}, N_act={N_act}")
    if args.ode_steps > 0:
        print(f"Mode:     ODE solve ({args.ode_steps} steps)")
    else:
        print(f"Mode:     single forward pass")
    print()

    B = 1
    frames_ctx = torch.randn(B, args.cameras, num_context_frames, 3, 224, 224, device=device)
    frames_fut = torch.randn(B, args.cameras, num_future_frames, 3, 224, 224, device=device)
    state = torch.randn(B, 6, device=device)
    text = ["pick up the red block"]

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    autocast_device = "cuda" if device == "cuda" else "cpu"

    def run_once():
        sync()
        t0 = time.perf_counter()

        ctx_tok = model.encode_video(frames_ctx)
        sync()
        t_enc = time.perf_counter()

        lang_tok, lang_mask = None, None
        if has_vlm:
            lang_tok, lang_mask = model.encode_language(text, images=frames_ctx[:, 0, -1])
        sync()
        t_vlm = time.perf_counter()

        if args.ode_steps > 0:
            _ = model.ode_solve(
                ctx_tok, state, lang_tok, lang_mask, num_steps=args.ode_steps,
            )
        else:
            fut_tok = model.encode_video(frames_fut)
            x_vid = torch.randn_like(fut_tok)
            x_act = torch.randn(B, N_act, action_dim, device=device)
            t_step = torch.tensor([0.5], device=device)
            _ = model(x_vid, x_act, ctx_tok, t_step, state, lang_tok, lang_mask)
        sync()
        t_dit = time.perf_counter()

        return (t_enc - t0), (t_vlm - t_enc), (t_dit - t_vlm), (t_dit - t0)

    times = []
    with torch.no_grad(), torch.autocast(autocast_device, dtype=amp_dtype):
        print("Warmup (compile + first run)...")
        run_once()
        print()

        dit_label = "ode" if args.ode_steps > 0 else "dit"
        for i in range(args.runs):
            dt_enc, dt_vlm, dt_dit, dt_total = run_once()
            total_ms = dt_total * 1000
            times.append(total_ms)
            print(
                f"Run {i+1}/{args.runs}: "
                f"vjepa={dt_enc*1000:.0f}ms  "
                f"vlm={dt_vlm*1000:.0f}ms  "
                f"{dit_label}={dt_dit*1000:.0f}ms  "
                f"total={total_ms:.0f}ms"
            )

    horizon_s = action_horizon / action_fps
    mean_ms = sum(times) / len(times)
    hz = 1000.0 / mean_ms if mean_ms > 0 else 0
    print(f"\nAction horizon: {action_horizon} steps @ {action_fps}Hz = {horizon_s:.2f}s")
    print(f"Mean latency:   {mean_ms:.0f}ms ({hz:.2f} Hz)")
    print(f"Realtime ratio: {horizon_s / (mean_ms / 1000):.2f}x")


if __name__ == "__main__":
    main()
