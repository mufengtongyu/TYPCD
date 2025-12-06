"""Visualize all predicted frames for a single typhoon event.

The script mirrors the rendering style of ``visualize_typhoon_track.py`` but
iterates over every predicted frame for a chosen typhoon event (scene) and
outputs both a Northwest Pacific view and a zoomed-in view for each frame.
It also produces full-event track summaries (history in black, predictions in
green) plus per-horizon summaries (6/12/18/24h) that connect the best
predicted points into a trajectory.
"""

import argparse
import os
from typing import Dict, List, Tuple, Optional

import matplotlib

# Use a non-interactive backend to avoid GUI issues on headless systems.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio.v2

from dataset.preprocessing import get_timesteps_data
from evaluation.trajectory_utils import prediction_output_to_trajectories
from visualize_typhoon_track import (
    build_inference_components,
    denormalize_positions,
    load_config,
    plot_paths,
    _get_axes_with_basemap,
)


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
        "--exp-name",
        default="WP_GRU_ConvNeXt_AdaLN_DiT_loss_1125",
        help="Experiment name for the primary TYPCD model",
    )
    parser.add_argument(
        "--test-epochs",
        type=int,
        nargs="+",
        default=[190],
        help="Epoch indices to evaluate for the primary TYPCD model",
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
        "--summary-output",
        default="fig/typhoon_event_track.png",
        help="Path to save the full-event track visualization",
    )
    parser.add_argument(
        "--horizon-output-dir",
        default="fig/typhoon_event_horizons",
        help="Directory to save per-horizon track visualizations",
    )
    parser.add_argument(
        "--event-gif",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Generate a GIF from per-frame typhoon_event outputs (enabled by default). "
            "Use --no-event-gif to skip."
        ),
    )
    parser.add_argument(
        "--event-gif-interval",
        type=float,
        default=1.5,
        help="Frame interval (seconds) for the typhoon_event GIF",
    )
    parser.add_argument(
        "--horizon-gif",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Generate a GIF from typhoon_event_horizons outputs (enabled by default). "
            "Use --no-horizon-gif to skip."
        ),
    )
    parser.add_argument(
        "--horizon-gif-interval",
        type=float,
        default=2,
        help="Frame interval (seconds) for the typhoon_event_horizons GIF",
    )
    parser.add_argument(
        "--horizon-reduction",
        choices=["min", "mean"],
        default="mean",
        help="How to aggregate ensemble errors when drawing per-horizon summaries.",
    )
    parser.add_argument(
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable overlaying TC-Diffuser results for typhoon_event and typhoon_event_horizons plots.",
    )
    parser.add_argument(
        "--compare-config",
        default="configs/baseline.yaml",
        help="Config file for the comparison TC-Diffuser model",
    )
    parser.add_argument(
        "--compare-exp-name",
        default="WP_epoch160",
        help="Experiment name for the comparison TC-Diffuser model",
    )
    parser.add_argument(
        "--compare-test-epochs",
        type=int,
        nargs="+",
        default=[160],
        help="Epoch indices to evaluate for the comparison TC-Diffuser model",
    )
    parser.add_argument(
        "--compare-checkpoint",
        default="experiments/WP_epoch160/WP_epoch160.pt",
        help="Checkpoint containing the TC-Diffuser trained weights",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=1,
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


def _best_predictions_by_horizon(
    predicted_trajs: np.ndarray, future: np.ndarray, reduction: str = "min"
) -> np.ndarray:
    """Return per-horizon representative predictions using the requested reduction."""

    future = future[: predicted_trajs.shape[1]]
    if future.shape[0] == 0:
        return np.empty((0, 2))
    
    if reduction == "mean":
        mean_points = np.nanmean(predicted_trajs, axis=0)
        return mean_points[: future.shape[0]]

    if reduction != "min":
        raise ValueError(f"Unsupported reduction '{reduction}'. Use 'min' or 'mean'.")

    if reduction == "mean":
        mean_points = np.nanmean(predicted_trajs, axis=0)
        return mean_points[: future.shape[0]]

    if reduction != "min":
        raise ValueError(f"Unsupported reduction '{reduction}'. Use 'min' or 'mean'.")

    # predicted_trajs: (num_samples, horizon, 2)
    diff = predicted_trajs - future[None, ...]
    errors = np.linalg.norm(diff, axis=-1)
    best_indices = np.nanargmin(errors, axis=0)
    return predicted_trajs[best_indices, np.arange(future.shape[0])]


def _get_axes_with_raster_basemap(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    raster_path: str,
):
    try:
        import cartopy.crs as ccrs
        import rasterio

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        with rasterio.open(raster_path) as src:
            image = src.read()
            image = np.moveaxis(image, 0, -1)
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            ax.imshow(image, origin="upper", extent=extent, transform=ccrs.PlateCarree())

        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False
        plot_kwargs = {"transform": ccrs.PlateCarree()}
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        plot_kwargs = {}

    return fig, ax, plot_kwargs


def _plot_track_pair(
    history_track: np.ndarray,
    predicted_track: np.ndarray,
    output_path: str,
    title: str,
    axes_builder=_get_axes_with_basemap,
    *,
    compare_track: Optional[np.ndarray] = None,
    compare_label: str = "TC-Diffuser",
    compare_color: str = "darkorange",
) -> None:
    fig, ax, plot_kwargs = axes_builder(100, 180, 0, 50)
    ax.plot(
        history_track[:, 0],
        history_track[:, 1],
        "o-",
        color="black",
        markersize=3,
        linewidth=1.2,
        label="Historical track",
        **plot_kwargs,
    )
    ax.plot(
        predicted_track[:, 0],
        predicted_track[:, 1],
        "o--",
        color="green",
        markersize=3,
        linewidth=1.2,
        label="TYPCD predicted track",
        **plot_kwargs,
    )
    if compare_track is not None and len(compare_track) > 0:
        ax.plot(
            compare_track[:, 0],
            compare_track[:, 1],
            "o--",
            color=compare_color,
            markersize=3,
            linewidth=1.2,
            label=f"{compare_label} predicted track",
            **plot_kwargs,
        )
    ax.set_title(title)
    ax.legend(loc="best")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _select_scene_index(eval_env, scene_name: Optional[str], fallback_index: int) -> int:
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
) -> Tuple[List[FramePrediction], object]:
    node_type = "PEDESTRIAN"
    ph = hyperparams["prediction_horizon"]
    max_hl = hyperparams["maximum_history_length"]
    scene = eval_env.scenes[scene_index]

    frame_predictions: Dict[int, FramePrediction] = {}
    target_node = None

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
            if target_node is None:
                target_node = nodes[i]

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

    if not frame_predictions or target_node is None:
        raise RuntimeError("No valid batches found for visualization.")

    return [frame_predictions[k] for k in sorted(frame_predictions.keys())], target_node


def save_frame_visualizations(
    frame_predictions: List[FramePrediction],
    scene_label: str,
    output_dir: str,
    zoom_output_dir: str,
    *,
    compare_predictions: Optional[List[FramePrediction]] = None,
    compare_label: str = "TC-Diffuser",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(zoom_output_dir, exist_ok=True)

    compare_lookup: Dict[int, FramePrediction] = {}
    if compare_predictions:
        compare_lookup = {
            ts: (predicted, history_raw, future_raw)
            for ts, predicted, history_raw, future_raw in compare_predictions
        }

    for ts, predicted_trajs, history_raw, future_raw in frame_predictions:
        base_name = f"{scene_label}_t{ts:04d}"
        output_path = os.path.join(output_dir, f"{base_name}.png")
        zoom_output_path = os.path.join(zoom_output_dir, f"{base_name}.png")
        compare_trajs = None
        if compare_lookup and ts in compare_lookup:
            compare_trajs = compare_lookup[ts][0]
        plot_paths(
            predicted_trajs,
            history_raw,
            future_raw,
            output_path,
            zoom_output_path,
            compare_prediction=compare_trajs,
            compare_label=compare_label,
        )


def _get_full_track(node) -> np.ndarray:
    """Extract the full ground-truth track for the given node."""

    tr_scene = np.array([node.first_timestep, node.last_timestep])
    full_track = node.get(tr_scene, {"position": ["x", "y"]})
    return full_track[:, :2]


def _prepare_tracks(
    frame_predictions: List[FramePrediction],
    horizon_reduction: str,
    full_track_raw: np.ndarray,
    # horizon_reduction: str,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    history_track_deg = denormalize_positions(full_track_raw)

    predicted_by_horizon: List[List[np.ndarray]] = []
    for ts, predicted_trajs, _history_raw, future_raw in frame_predictions:
        best_points = _best_predictions_by_horizon(predicted_trajs, future_raw, reduction=horizon_reduction)
        if best_points.size == 0:
            continue

        if not predicted_by_horizon:
            predicted_by_horizon = [[] for _ in range(best_points.shape[0])]

        for idx in range(len(predicted_by_horizon)):
            predicted_by_horizon[idx].append(best_points[idx])

    if not predicted_by_horizon:
        raise RuntimeError("Unable to derive predicted tracks for any horizon.")

    predicted_tracks_deg = [denormalize_positions(np.vstack(track)) for track in predicted_by_horizon]
    return history_track_deg, predicted_tracks_deg


def _save_gif(image_paths: List[str], gif_path: str, duration: float) -> bool:
    """Create a GIF from a list of image paths if any images are available.

    Returns True if the GIF was created; False if skipped because no images were found.
    """

    if not image_paths:
        print(f"Skipping GIF creation for '{gif_path}': no input images found.")
        return False

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    frames = [imageio.v2.imread(path) for path in image_paths]

    # Pillow expects GIF durations in milliseconds; convert from seconds while guarding against
    # zero-duration frames that would result in effectively instant playback.
    frame_duration_ms = max(int(duration * 1000), 1)
    imageio.v2.mimsave(gif_path, frames, duration=frame_duration_ms)
    print(
        f"Saved GIF to '{gif_path}' with {len(frames)} frame(s) at {duration}s per frame."
    )
    return True


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(
        args.config,
        args.checkpoint,
        exp_name_override=args.exp_name,
        test_epochs=args.test_epochs,
    )
    model, eval_env, hyperparams = build_inference_components(
        config, args.checkpoint, device
    )

    scene_index = _select_scene_index(eval_env, args.scene_name, args.scene_index)
    frame_predictions, target_node = collect_frame_predictions(
        model,
        eval_env,
        hyperparams,
        args.sampling,
        args.step,
        scene_index,
        args.timestep_stride,
    )

    compare_frame_predictions: Optional[List[FramePrediction]] = None
    compare_predicted_tracks_deg: Optional[List[np.ndarray]] = None
    if args.compare:
        compare_config = load_config(
            args.compare_config,
            args.compare_checkpoint,
            exp_name_override=args.compare_exp_name,
            test_epochs=args.compare_test_epochs,
        )
        compare_model, compare_eval_env, compare_hypers = build_inference_components(
            compare_config,
            args.compare_checkpoint,
            device,
            diffusion_module_path="models.backup.diffusion_tcdiff",
        )
        compare_scene_index = _select_scene_index(
            compare_eval_env,
            args.scene_name or eval_env.scenes[scene_index].name,
            args.scene_index,
        )
        compare_frame_predictions, _ = collect_frame_predictions(
            compare_model,
            compare_eval_env,
            compare_hypers,
            args.sampling,
            args.step,
            compare_scene_index,
            args.timestep_stride,
        )

    scene = eval_env.scenes[scene_index]
    scene_label = _sanitize_label(args.scene_name or scene.name or f"scene_{scene_index}")

    save_frame_visualizations(
        frame_predictions,
        scene_label,
        args.output_dir,
        args.zoom_output_dir,
        compare_predictions=compare_frame_predictions,
    )

    event_gif_created = False
    if args.event_gif:
        frame_paths = sorted(
            os.path.join(args.output_dir, fname)
            for fname in os.listdir(args.output_dir)
            if fname.lower().endswith(".png")
        )
        event_gif_path = os.path.join(args.output_dir, f"{scene_label}.gif")
        event_gif_created = _save_gif(frame_paths, event_gif_path, args.event_gif_interval)

    full_track_raw = _get_full_track(target_node)
    history_track_deg, predicted_tracks_deg = _prepare_tracks(
        frame_predictions, args.horizon_reduction, full_track_raw
    )

    if compare_frame_predictions:
        _, compare_predicted_tracks_deg = _prepare_tracks(
            compare_frame_predictions, args.horizon_reduction, full_track_raw
        )

    summary_predicted = predicted_tracks_deg[0]
    _plot_track_pair(
        history_track_deg,
        summary_predicted,
        args.summary_output,
        "Full typhoon event track (best 6h forecast)",
        compare_track=(
            compare_predicted_tracks_deg[0]
            if compare_predicted_tracks_deg and len(compare_predicted_tracks_deg) > 0
            else None
        ),
    )

    horizon_hours = [6, 12, 18, 24]
    os.makedirs(args.horizon_output_dir, exist_ok=True)
    horizon_outputs: List[str] = []
    for idx, hours in enumerate(horizon_hours[: len(predicted_tracks_deg)]):
        horizon_output = os.path.join(
            args.horizon_output_dir, f"{scene_label}_{hours}h_track.png"
        )
        _plot_track_pair(
            history_track_deg,
            predicted_tracks_deg[idx],
            horizon_output,
            f"{hours}h best forecast track",
            axes_builder=lambda a, b, c, d: _get_axes_with_raster_basemap(
                a, b, c, d, "/mnt/e/data/HYP_LR_SR_OB_DR/HYP_LR_SR_OB_DR.tif"
            ),
            compare_track=(
                compare_predicted_tracks_deg[idx]
                if compare_predicted_tracks_deg
                and len(compare_predicted_tracks_deg) > idx
                else None
            ),
        )
        horizon_outputs.append(horizon_output)

    horizon_gif_created = False
    if args.horizon_gif:
        horizon_gif_path = os.path.join(args.horizon_output_dir, f"{scene_label}_horizons.gif")
        horizon_gif_created = _save_gif(
            sorted(horizon_outputs), horizon_gif_path, args.horizon_gif_interval
        )

    print(
        f"Saved {len(frame_predictions)} frame(s) for scene '{scene_label}' "
        f"to '{args.output_dir}' and '{args.zoom_output_dir}'."
    )
    print(
        f"Saved full-event tracks to '{args.summary_output}' and horizon plots to "
        f"'{args.horizon_output_dir}'."
    )
    if args.event_gif:
        if not event_gif_created:
            print("Event GIF generation was requested but no frames were available to combine.")
    else:
        print(
            "GIF creation for per-frame outputs is disabled because --no-event-gif was used. "
            "This is enabled by default; omit the flag to generate a typhoon_event GIF."
        )

    if args.horizon_gif:
        if not horizon_gif_created:
            print(
                "Horizon GIF generation was requested but no per-horizon images were found."
            )
    else:
        print(
            "GIF creation for horizon summaries is disabled because --no-horizon-gif was used. "
            "This is enabled by default; omit the flag to generate a typhoon_event_horizons GIF."
        )


if __name__ == "__main__":
    main()



