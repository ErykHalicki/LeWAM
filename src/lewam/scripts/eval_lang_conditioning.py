"""
Evaluate whether language conditioning affects model predictions.

Loads a few samples from a LeRobot dataset, runs ODE inference with and
without language, and reports MSE between the two sets of predictions
plus MSE against ground truth for each.

Usage:
    python src/lewam/scripts/eval_lang_conditioning.py \
        --checkpoint runs/my-run/my-run_latest.pt \
        --repo ehalicki/so101_multitask \
        --samples 8
"""
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms

from lewam.models.lewam import LeWAM

PATCH_SIZE = LeWAM.VJEPA_PATCH_SIZE


def load_samples(repo_id, model, num_samples, crop_size, num_cameras):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = model.fps
    action_fps = model.action_fps
    num_context = model.num_context_frames
    num_future = model.num_future_frames
    action_horizon = model.action_horizon

    past_ts = [-(num_context - 1 - i) / fps for i in range(num_context)]
    future_ts = [(i + 1) / fps for i in range(num_future)]
    action_ts = [i / action_fps for i in range(action_horizon + 1)]

    ds = LeRobotDataset(repo_id=repo_id, revision="main")
    camera_keys = sorted(k for k in ds.meta.features if k.startswith("observation.images."))
    assert len(camera_keys) == num_cameras

    image_tx = transforms.Resize((crop_size, crop_size), antialias=True)
    delta_timestamps = {k: past_ts + future_ts for k in camera_keys}
    delta_timestamps["observation.state"] = [0.0]
    delta_timestamps["action"] = action_ts

    ds = LeRobotDataset(
        repo_id=repo_id,
        revision="main",
        delta_timestamps=delta_timestamps,
        image_transforms=image_tx,
    )

    indices = torch.linspace(0, len(ds) - 1, num_samples).long().tolist()
    batches = [ds[i] for i in indices]

    frames_list, states, actions_list, tasks = [], [], [], []
    for b in batches:
        cam_frames = torch.stack([b[k] for k in camera_keys], dim=0)
        frames_list.append(cam_frames)
        states.append(b["observation.state"].squeeze(0))
        actions_list.append(b["action"])
        tasks.append(b["task"])

    all_frames = torch.stack(frames_list)
    all_states = torch.stack(states)
    all_actions = torch.stack(actions_list)

    return all_frames, all_states, all_actions, tasks, camera_keys


@torch.no_grad()
def run_eval(model, frames, states, actions_gt, tasks, num_cameras, device, ode_steps):
    model.eval()
    model.to(device)

    crop_size = model.video_encoder.preprocessor.crop_size
    frame_latent_h = crop_size // PATCH_SIZE
    frame_latent_w = (crop_size // PATCH_SIZE) * num_cameras
    if frame_latent_h != model.frame_latent_h or frame_latent_w != model.frame_latent_w:
        model.set_patch_grid(frame_latent_h, frame_latent_w, num_cameras)

    N = frames.shape[0]
    num_context = model.num_context_frames
    num_fut_t = model.num_future_tubelets
    spatial = model.frame_latent_h * model.frame_latent_w
    dt = 1.0 / model.action_fps
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    per_sample = []
    agg = {k: [] for k in [
        "vid_lang_gt", "vid_none_gt", "vid_wrong_gt", "vid_lang_none", "vid_lang_wrong",
        "act_lang_gt", "act_none_gt", "act_wrong_gt", "act_lang_none", "act_lang_wrong",
        "cos_vid", "cos_act",
        "gt_vdiff", "gt_vnorm", "lang_vdiff", "lang_vnorm", "none_vdiff", "none_vnorm",
        "gt_adiff", "lang_adiff", "none_adiff",
    ]}

    for i in range(N):
        print(f"  sample {i+1}/{N}...", end=" ", flush=True)
        f = frames[i:i+1].to(device)
        ctx = f[:, :, :num_context]
        fut = f[:, :, num_context:]
        st = states[i:i+1].to(device)
        act = actions_gt[i:i+1].to(device)
        task = [tasks[i]]

        context_tokens = model.encode_video(ctx)
        future_tokens_gt_i = model.encode_video(fut)
        norm_state = model.normalize_state(st)
        rel_vel = (act[:, 1:] - act[:, :-1]) / dt
        norm_act_gt = model.normalize_actions(rel_vel)

        last_ctx = torch.cat([ctx[:, c, -1] for c in range(num_cameras)], dim=-1)
        lang_tok, lang_msk = model.encode_language(task, images=last_ctx)

        with amp_ctx:
            pv_lang, pa_lang = model.ode_solve(context_tokens, norm_state, lang_tok, lang_msk, num_steps=ode_steps)
            pv_none, pa_none = model.ode_solve(context_tokens, norm_state, None, None, num_steps=ode_steps)

        wrong_tok, wrong_msk = model.encode_language([task[0] + " backwards while spinning"], images=last_ctx)
        with amp_ctx:
            pv_wrong, pa_wrong = model.ode_solve(context_tokens, norm_state, wrong_tok, wrong_msk, num_steps=ode_steps)

        def mse(a, b):
            return F.mse_loss(a.float(), b.float()).item()

        agg["vid_lang_gt"].append(mse(pv_lang, future_tokens_gt_i))
        agg["vid_none_gt"].append(mse(pv_none, future_tokens_gt_i))
        agg["vid_wrong_gt"].append(mse(pv_wrong, future_tokens_gt_i))
        agg["vid_lang_none"].append(mse(pv_lang, pv_none))
        agg["vid_lang_wrong"].append(mse(pv_lang, pv_wrong))
        agg["act_lang_gt"].append(mse(pa_lang, norm_act_gt))
        agg["act_none_gt"].append(mse(pa_none, norm_act_gt))
        agg["act_wrong_gt"].append(mse(pa_wrong, norm_act_gt))
        agg["act_lang_none"].append(mse(pa_lang, pa_none))
        agg["act_lang_wrong"].append(mse(pa_lang, pa_wrong))
        agg["cos_vid"].append(F.cosine_similarity(pv_lang.float().reshape(1, -1), pv_none.float().reshape(1, -1)).item())
        agg["cos_act"].append(F.cosine_similarity(pa_lang.float().reshape(1, -1), pa_none.float().reshape(1, -1)).item())

        def tdyn_vid(v):
            v = v[0].float().reshape(num_fut_t, spatial, -1)
            return (v[1:] - v[:-1]).norm(dim=-1).mean().item(), v.norm(dim=-1).mean().item()

        def tdyn_act(a):
            return (a[0, 1:] - a[0, :-1]).float().norm(dim=-1).mean().item()

        gvd, gvn = tdyn_vid(future_tokens_gt_i)
        lvd, lvn = tdyn_vid(pv_lang)
        nvd, nvn = tdyn_vid(pv_none)
        agg["gt_vdiff"].append(gvd); agg["gt_vnorm"].append(gvn)
        agg["lang_vdiff"].append(lvd); agg["lang_vnorm"].append(lvn)
        agg["none_vdiff"].append(nvd); agg["none_vnorm"].append(nvn)
        agg["gt_adiff"].append(tdyn_act(norm_act_gt))
        agg["lang_adiff"].append(tdyn_act(pa_lang))
        agg["none_adiff"].append(tdyn_act(pa_none))

        per_sample.append({
            "task": tasks[i],
            "act_mse_lang": agg["act_lang_gt"][-1],
            "act_mse_none": agg["act_none_gt"][-1],
            "act_mse_wrong": agg["act_wrong_gt"][-1],
            "act_diff_lang_none": agg["act_lang_none"][-1],
            "vid_frame_diff_pred": lvd,
            "vid_frame_diff_gt": gvd,
            "act_step_diff_pred": agg["lang_adiff"][-1],
            "act_step_diff_gt": agg["gt_adiff"][-1],
        })
        print("done")
        del f, ctx, fut, st, act, context_tokens, future_tokens_gt_i, pv_lang, pa_lang, pv_none, pa_none, pv_wrong, pa_wrong
        torch.cuda.empty_cache()

    def mean(lst):
        return sum(lst) / len(lst)

    gt_vd, lang_vd, none_vd = mean(agg["gt_vdiff"]), mean(agg["lang_vdiff"]), mean(agg["none_vdiff"])
    gt_ad, lang_ad, none_ad = mean(agg["gt_adiff"]), mean(agg["lang_adiff"]), mean(agg["none_adiff"])

    results = {
        "video": {
            "lang_vs_gt": mean(agg["vid_lang_gt"]),
            "none_vs_gt": mean(agg["vid_none_gt"]),
            "wrong_vs_gt": mean(agg["vid_wrong_gt"]),
            "lang_vs_none": mean(agg["vid_lang_none"]),
            "lang_vs_wrong": mean(agg["vid_lang_wrong"]),
            "cos_lang_none": mean(agg["cos_vid"]),
        },
        "action": {
            "lang_vs_gt": mean(agg["act_lang_gt"]),
            "none_vs_gt": mean(agg["act_none_gt"]),
            "wrong_vs_gt": mean(agg["act_wrong_gt"]),
            "lang_vs_none": mean(agg["act_lang_none"]),
            "lang_vs_wrong": mean(agg["act_lang_wrong"]),
            "cos_lang_none": mean(agg["cos_act"]),
        },
        "temporal": {
            "gt_vid_frame_diff": gt_vd,
            "lang_vid_frame_diff": lang_vd,
            "none_vid_frame_diff": none_vd,
            "gt_vid_mean_norm": mean(agg["gt_vnorm"]),
            "lang_vid_mean_norm": mean(agg["lang_vnorm"]),
            "none_vid_mean_norm": mean(agg["none_vnorm"]),
            "vid_diff_ratio_lang": lang_vd / gt_vd if gt_vd > 0 else float("inf"),
            "vid_diff_ratio_none": none_vd / gt_vd if gt_vd > 0 else float("inf"),
            "gt_act_step_diff": gt_ad,
            "lang_act_step_diff": lang_ad,
            "none_act_step_diff": none_ad,
            "act_diff_ratio_lang": lang_ad / gt_ad if gt_ad > 0 else float("inf"),
            "act_diff_ratio_none": none_ad / gt_ad if gt_ad > 0 else float("inf"),
        },
    }

    return results, per_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo", default="ehalicki/so101_multitask")
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--ode-steps", type=int, default=4)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LeWAM.from_checkpoint(ckpt,
        num_context_frames=ckpt["config"].get("num_context_frames", 16),
        num_future_frames=ckpt["config"].get("num_future_frames", 12),
    )
    step = ckpt.get("step", "?")
    print(f"  step={step}  dim={model.model_dim}  depth={len(model.blocks)}  vlm={'yes' if model.vlm_encoder else 'no'}")
    del ckpt

    print(f"Loading {args.samples} samples from {args.repo}...")
    crop_size = model.video_encoder.preprocessor.crop_size
    frames, states, actions, tasks, cam_keys = load_samples(
        args.repo, model, args.samples, crop_size, args.cameras,
    )
    print(f"  cameras={cam_keys}  frames={frames.shape}  unique tasks={len(set(tasks))}")

    print(f"\nRunning inference ({args.ode_steps} ODE steps) on {device}...")
    results, per_sample = run_eval(model, frames, states, actions, tasks, args.cameras, device, args.ode_steps)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    for domain in ["video", "action"]:
        r = results[domain]
        print(f"\n  {domain.upper()}:")
        print(f"    MSE vs GT  (with lang):    {r['lang_vs_gt']:.6f}")
        print(f"    MSE vs GT  (no lang):      {r['none_vs_gt']:.6f}")
        print(f"    MSE vs GT  (wrong lang):   {r['wrong_vs_gt']:.6f}")
        print(f"    MSE lang vs no-lang:       {r['lang_vs_none']:.6f}")
        print(f"    MSE lang vs wrong-lang:    {r['lang_vs_wrong']:.6f}")
        print(f"    Cosine sim lang vs none:   {r['cos_lang_none']:.6f}")

    t = results["temporal"]
    print("\n" + "=" * 70)
    print("TEMPORAL DYNAMICS (frame-to-frame change magnitude)")
    print("=" * 70)
    print(f"\n  VIDEO (L2 norm of tubelet-to-tubelet diff, averaged over patches):")
    print(f"    GT:           {t['gt_vid_frame_diff']:.4f}   (mean token norm: {t['gt_vid_mean_norm']:.4f})")
    print(f"    Pred (lang):  {t['lang_vid_frame_diff']:.4f}   (mean token norm: {t['lang_vid_mean_norm']:.4f})   ratio to GT: {t['vid_diff_ratio_lang']:.2f}x")
    print(f"    Pred (none):  {t['none_vid_frame_diff']:.4f}   (mean token norm: {t['none_vid_mean_norm']:.4f})   ratio to GT: {t['vid_diff_ratio_none']:.2f}x")
    print(f"\n  ACTIONS (L2 norm of step-to-step diff):")
    print(f"    GT:           {t['gt_act_step_diff']:.4f}")
    print(f"    Pred (lang):  {t['lang_act_step_diff']:.4f}   ratio to GT: {t['act_diff_ratio_lang']:.2f}x")
    print(f"    Pred (none):  {t['none_act_step_diff']:.4f}   ratio to GT: {t['act_diff_ratio_none']:.2f}x")
    if t["vid_diff_ratio_lang"] < 0.1:
        print(f"\n  ** WARNING: Predicted video has <10% of GT temporal variation, likely collapsing to static prediction **")

    print("\n" + "=" * 70)
    print("PER-SAMPLE ACTION MSE")
    print("=" * 70)
    for i, s in enumerate(per_sample):
        delta = s["act_mse_none"] - s["act_mse_lang"]
        direction = "lang helps" if delta > 0 else "lang hurts"
        print(f"  [{i}] task=\"{s['task'][:50]}\"")
        print(f"      lang={s['act_mse_lang']:.6f}  none={s['act_mse_none']:.6f}  "
              f"wrong={s['act_mse_wrong']:.6f}  diff={s['act_diff_lang_none']:.6f}  ({direction})")
        vr = s["vid_frame_diff_pred"] / s["vid_frame_diff_gt"] if s["vid_frame_diff_gt"] > 0 else float("inf")
        print(f"      vid dt: pred={s['vid_frame_diff_pred']:.4f} gt={s['vid_frame_diff_gt']:.4f} ({vr:.2f}x)  "
              f"act dt: pred={s['act_step_diff_pred']:.4f} gt={s['act_step_diff_gt']:.4f}")

    lang_none_diff = results["action"]["lang_vs_none"]
    if lang_none_diff < 1e-4:
        print("\n  ** Language conditioning has NEGLIGIBLE effect on actions **")
    elif results["action"]["lang_vs_gt"] < results["action"]["none_vs_gt"]:
        print("\n  ** Language conditioning IMPROVES action accuracy **")
    else:
        print("\n  ** Language conditioning HURTS action accuracy **")


if __name__ == "__main__":
    main()
