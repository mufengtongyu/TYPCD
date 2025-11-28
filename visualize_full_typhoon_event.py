"""Visualize all predicted frames for a single typhoon event.

The script mirrors the rendering style of ``visualize_typhoon_track.py`` but
iterates over every predicted frame for a chosen typhoon event (scene) and
outputs both a Northwest Pacific view and a zoomed-in view for each frame.
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib

# Use a non-interactive backend to avoid GUI issues on headless systems.
matplotlib.use("Agg")

import numpy as np
import torch

from dataset.preprocessing import get_timesteps_data
from evaluation.trajectory_utils import prediction_output_to_trajectories
from visualize_typhoon_track import build_inference_components, load_config, plot_paths


FramePrediction = Tuple[int, np.ndarray, np.ndarray, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize all predicted frames for a single typhoon event",
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Config file used to build the model",
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/WP_GRU_ConvNeXt_AdaLN_DiT_loss_1125/WP_epoch190.pt",
        help="Checkpoint containing the trained weights",
    )
    parser.add_argument(
        "--output-dir",
        default="fig/typhoon_event",
        help="Directory to save the full-view visualizations",
    )
    parser.add_argument(
        "--zoom-output-dir",
        default="fig/typhoon_event_zoom",
        help="Directory to save the zoomed-in visualizations",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Index of the evaluation scene to visualize",
    )
    parser.add_argument(
        "--scene-name",
        default=None,
        help="Name of the evaluation scene to visualize (overrides --scene-index if provided)",
    )
    parser.add_argument(
        "--timestep-stride",
        type=int,
        default=5,
        help="Stride when scanning timesteps in the selected scene",
    )
    parser.add_argument(
        "--sampling",
        choices=["ddpm", "ddim"],
        default="ddim",
        help="Sampling strategy for the diffusion model",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Number of denoising steps for DDIM sampling",
    )
    return parser.parse_args()


def _sanitize_label(label: str) -> str:
    """Generate a filesystem-friendly label for output file names."""

    safe_label = label.replace("/", "-").replace("\\", "-")
    safe_label = safe_label.replace(" ", "_")
    return safe_label


def _select_scene_index(eval_env, scene_name: str | None, fallback_index: int) -> int:
    if scene_name is None:
        return fallback_index

    for idx, scene in enumerate(eval_env.scenes):
        if scene.name == scene_name:
            return idx

    available = ", ".join(scene.name for scene in eval_env.scenes)
    raise ValueError(
        f"Scene named '{scene_name}' not found. Available scenes: {available}",
    )


def collect_frame_predictions(
    model,
    eval_env,
    hyperparams: Dict,
    sampling: str,
    step: int,
    scene_index: int,
    timestep_stride: int,
) -> List[FramePrediction]:
    node_type = "PEDESTRIAN"
    ph = hyperparams["prediction_horizon"]
    max_hl = hyperparams["maximum_history_length"]
    scene = eval_env.scenes[scene_index]

    frame_predictions: Dict[int, FramePrediction] = {}

    for t in range(0, scene.timesteps, timestep_stride):
        timesteps = np.arange(t, min(t + timestep_stride, scene.timesteps))
        batch = get_timesteps_data(
            env=eval_env,
            scene=scene,
            t=timesteps,
            node_type=node_type,
            state=hyperparams["state"],
            pred_state=hyperparams["pred_state"],
            edge_types=eval_env.get_edge_types(),
            min_ht=7,
            max_ht=max_hl,
            min_ft=ph,
            max_ft=ph,
            hyperparams=hyperparams,
        )
        if batch is None:
            continue

        test_batch, nodes, timesteps_o = batch
        traj_pred = model.generate(
            test_batch,
            node_type,
            num_points=ph,
            sample=6,
            bestof=True,
            sampling=sampling,
            step=step,
        )

        predictions_dict: Dict[int, Dict] = {}
        for i, ts in enumerate(timesteps_o):
            predictions_dict.setdefault(ts, {})[nodes[i]] = np.transpose(
                traj_pred[:, [i]], (1, 0, 2, 3)
            )

        prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(
            predictions_dict,
            scene.dt,
            max_hl,
            ph,
            map=None,
            prune_ph_to_future=True,
        )

        for ts_key, node_predictions in prediction_dict.items():
            if ts_key in frame_predictions:
                continue

            node_key = next(iter(node_predictions.keys()))
            predicted_trajs = node_predictions[node_key][0, :, :, :2]
            history_raw = histories_dict[ts_key][node_key][:, :2]
            future_raw = futures_dict[ts_key][node_key][:, :2]

            frame_predictions[ts_key] = (
                int(ts_key),
                predicted_trajs,
                history_raw[-8:],
                future_raw[:4],
            )

    if not frame_predictions:
        raise RuntimeError("No valid batches found for visualization.")

    return [frame_predictions[k] for k in sorted(frame_predictions.keys())]


def save_frame_visualizations(
    frame_predictions: List[FramePrediction],
    scene_label: str,
    output_dir: str,
    zoom_output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(zoom_output_dir, exist_ok=True)

    for ts, predicted_trajs, history_raw, future_raw in frame_predictions:
        base_name = f"{scene_label}_t{ts:04d}"
        output_path = os.path.join(output_dir, f"{base_name}.png")
        zoom_output_path = os.path.join(zoom_output_dir, f"{base_name}.png")
        plot_paths(predicted_trajs, history_raw, future_raw, output_path, zoom_output_path)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config, args.checkpoint)
    model, eval_env, hyperparams = build_inference_components(
        config, args.checkpoint, device
    )

    scene_index = _select_scene_index(eval_env, args.scene_name, args.scene_index)
    frame_predictions = collect_frame_predictions(
        model,
        eval_env,
        hyperparams,
        args.sampling,
        args.step,
        scene_index,
        args.timestep_stride,
    )

    scene = eval_env.scenes[scene_index]
    scene_label = _sanitize_label(args.scene_name or scene.name or f"scene_{scene_index}")

    save_frame_visualizations(
        frame_predictions,
        scene_label,
        args.output_dir,
        args.zoom_output_dir,
    )

    print(
        f"Saved {len(frame_predictions)} frame(s) for scene '{scene_label}' "
        f"to '{args.output_dir}' and '{args.zoom_output_dir}'."
    )


if __name__ == "__main__":
    main()
