import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import dill

import matplotlib
matplotlib.use("Agg")  # 使用无界面后端，避免 Qt/Wayland 问题

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from easydict import EasyDict


from dataset.preprocessing import get_timesteps_data
from evaluation.trajectory_utils import prediction_output_to_trajectories
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Typhoon track visualization")
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
        "--output",
        default="fig/typhoon_prediction.png",
        help="Path to save the visualization",
    )
    parser.add_argument(
        "--zoom-output",
        default="fig/typhoon_prediction_zoom.png",
        help="Path to save the zoomed-in visualization",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Index of the evaluation scene to visualize",
    )
    parser.add_argument(
        "--timestep-stride",
        type=int,
        default=10,
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


def load_config(config_path: str, checkpoint_path: str) -> EasyDict:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    config.eval_mode = True
    config.eval_dataset = config.get("eval_dataset", config.get("train_dataset", "WP"))
    config.train_dataset = config.get("train_dataset", "WP")
    checkpoint_name = Path(checkpoint_path)
    config.exp_name = checkpoint_name.parent.name
    if checkpoint_name.stem.startswith(f"{config.train_dataset}_epoch"):
        try:
            config.eval_at = int(checkpoint_name.stem.split("_epoch")[-1])
        except ValueError:
            pass
    return config


def build_inference_components(config: EasyDict, checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found. Please download the weights before running the script."
        )

    hyperparams = get_traj_hypers()
    hyperparams["enc_rnn_dim_edge"] = config.encoder_dim // 2
    hyperparams["enc_rnn_dim_edge_influence"] = config.encoder_dim // 2
    hyperparams["enc_rnn_dim_history"] = config.encoder_dim // 2
    hyperparams["enc_rnn_dim_future"] = config.encoder_dim // 2

    registrar = ModelRegistrar(str(Path(checkpoint_path).parent), device.type)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    registrar.load_models(checkpoint["encoder"])

    train_data_path = os.path.join(config.data_dir, f"{config.train_dataset}_train.pkl")
    eval_data_path = os.path.join(config.data_dir, f"{config.eval_dataset}_test.pkl")
    with open(train_data_path, "rb") as f:
        train_env = dill.load(f, encoding="latin1")
    with open(eval_data_path, "rb") as f:
        eval_env = dill.load(f, encoding="latin1")

    encoder = Trajectron(registrar, hyperparams, device.type)
    encoder.set_environment(train_env)
    encoder.set_annealing_params()

    model = AutoEncoder(config, encoder=encoder).to(device)
    model.load_state_dict(checkpoint["ddpm"])
    model.eval()
    return model, eval_env, hyperparams


def denormalize_positions(points: np.ndarray) -> np.ndarray:
    # lon = points[..., 0] / 10.0 * 500.0 + 1300.0
    # lat = points[..., 1] / 6.0 * 300.0 + 300
    # return np.stack([lon / 10.0, lat / 10.0], axis=-1)
    """Convert serialized coordinates back to geographic degrees.

        The training data stores longitude/latitude using the same affine
        transform applied during evaluation (see ``compute_ade_x_y_traj_each_time``):

        * normalized_x -> (x / 10 * 500 + 1800)  (tenth degrees of longitude)
        * normalized_y -> (y / 6  * 300)         (tenth degrees of latitude)

        Some intermediate tensors (e.g., outputs from ``compute_ade_x_y_traj_each_time``)
        are already in tenth-degree units. We therefore detect large values and only
        apply the affine transform when the inputs are still normalized.
    """

    try:
        points = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        # Some call sites may pass objects that include non-numeric values (e.g.,
        # accidental string inputs). Coerce those entries to NaN so plotting can
        # proceed using only the valid coordinates.
        raw = np.asarray(points, dtype=object)
        coerced = np.full(raw.shape, np.nan, dtype=np.float64)
        it = np.nditer(raw, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
        for value in it:
            try:
                coerced[it.multi_index] = float(value)
            except (TypeError, ValueError):
                coerced[it.multi_index] = np.nan
        points = coerced

    if np.isnan(points).all():
        return points

    # If values are already in tenth-degrees (e.g., 1000-1800 for longitude),
    # simply convert to degrees.
    if np.nanmax(np.abs(points)) > 400:
        return points / 10.0

    lon_tenth = points[..., 0] / 10.0 * 500.0 + 1300.0
    lat_tenth = points[..., 1] / 6.0 * 300.0 + 300.0
    return np.stack([lon_tenth / 10.0, lat_tenth / 10.0], axis=-1)


def prepare_batch_prediction(
    model: AutoEncoder,
    eval_env,
    hyperparams: Dict,
    sampling: str,
    step: int,
    scene_index: int,
    timestep_stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_type = "PEDESTRIAN"
    ph = hyperparams["prediction_horizon"]
    max_hl = hyperparams["maximum_history_length"]
    scene = eval_env.scenes[scene_index]

    for t in range(0, scene.timesteps, timestep_stride):
        timesteps = np.arange(t, t + timestep_stride)
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
            predictions_dict.setdefault(ts, {})[nodes[i]] = np.transpose(traj_pred[:, [i]], (1, 0, 2, 3))

        prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(
            predictions_dict,
            scene.dt,
            max_hl,
            ph,
            map=None,
            prune_ph_to_future=True,
        )
        ts_key = list(prediction_dict.keys())[0]
        node_key = list(prediction_dict[ts_key].keys())[0]

        predicted_trajs = prediction_dict[ts_key][node_key][0, :, :, :2]
        history_raw = histories_dict[ts_key][node_key][:, :2]
        future_raw = futures_dict[ts_key][node_key][:, :2]

        # Keep the last 8 history points and the next 4 ground-truth points to
        # match the visualization specification.
        return predicted_trajs, history_raw[-8:], future_raw[:4]

    raise RuntimeError("No valid batches found for visualization.")


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Compute the convex hull of a set of 2D points using Andrew's monotone chain."""

    points = np.asarray(points)
    if len(points) <= 1:
        return points
    
    # Sort points lexicographically by x, then y.
    points_sorted = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    # Concatenate lower and upper to get full hull; last element of each list is omitted
    # because it is repeated at the beginning of the other list.
    hull = np.vstack((lower[:-1], upper[:-1]))
    return hull


def _get_axes_with_basemap(lon_min: float, lon_max: float, lat_min: float, lat_max: float):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines(resolution="110m", linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
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


def _get_axes_with_raster_basemap(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    raster_path: str,
    figsize: tuple = (10, 8),
):
    """Return axes configured with a raster basemap when available."""

    try:
        import cartopy.crs as ccrs
        import rasterio

        fig = plt.figure(figsize=figsize)
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
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        plot_kwargs = {}

    return fig, ax, plot_kwargs


def _plot_tracks_on_axis(
    ax,
    predicted_deg: np.ndarray,
    history_deg: np.ndarray,
    future_deg: np.ndarray,
    plot_kwargs: Dict,
):
    current_point = history_deg[-1]

    cone_points = np.vstack([current_point, predicted_deg.reshape(-1, 2)])
    cone_hull = _convex_hull(cone_points)
    if len(cone_hull) >= 3:
        ax.fill(
            cone_hull[:, 0],
            cone_hull[:, 1],
            color="limegreen",
            alpha=0.5,
            zorder=1,
            label="Probability cone",
            **plot_kwargs,
        )

    ax.plot(
        history_deg[:, 0],
        history_deg[:, 1],
        "o-",
        color="black",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=2,
        label="History",
        zorder=3,
        **plot_kwargs,
    )
    future_with_current = np.vstack([current_point, future_deg])

    ax.plot(
        future_with_current[:, 0],
        future_with_current[:, 1],
        "o-",
        color="red",
        markerfacecolor="red",
        markeredgecolor="red",
        markersize=2,
        label="Future ground truth",
        zorder=3,
        **plot_kwargs,
    )
    ax.plot(
        current_point[0],
        current_point[1],
        "o",
        color="blue",
        label="Current position",
        markersize=4,
        zorder=4,
        **plot_kwargs,
    )

    for idx, traj in enumerate(predicted_deg):
        predicted_with_current = np.vstack([current_point, traj])
        ax.plot(
            predicted_with_current[:, 0],
            predicted_with_current[:, 1],
            "o:",
            color="green",
            markerfacecolor="green",
            markeredgecolor="green",
            markersize=2,
            linewidth=1,
            label="Predicted trajectories" if idx == 0 else None,
            zorder=6,
            **plot_kwargs,
        )

    ensemble_mean = np.nanmean(predicted_deg, axis=0)
    ensemble_with_current = np.vstack([current_point, ensemble_mean])
    ax.plot(
        ensemble_with_current[:, 0],
        ensemble_with_current[:, 1],
        "o-.",
        color="green",
        markerfacecolor="green",
        markeredgecolor="green",
        markersize=5,
        linewidth=1.5,
        label="Ensemble mean trajectory",
        zorder=7,
        **plot_kwargs,
    )

    ax.legend(loc="best")


def plot_paths(
    predicted_trajs: np.ndarray,
    history: np.ndarray,
    future: np.ndarray,
    output_path: str,
    zoom_output_path: str,
):
    raster_path = "/mnt/e/data/HYP_LR_SR_OB_DR/HYP_LR_SR_OB_DR.tif"
    predicted_deg = denormalize_positions(np.asarray(predicted_trajs))
    history_deg = denormalize_positions(np.asarray(history))
    future_deg = denormalize_positions(np.asarray(future))

    # Full Northwest Pacific view
    fig, ax, plot_kwargs = _get_axes_with_raster_basemap(
        100, 180, 0, 50, raster_path
    )
    _plot_tracks_on_axis(ax, predicted_deg, history_deg, future_deg, plot_kwargs)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=500, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Zoomed view centered on the track to keep the trajectory near 80% of the frame
    all_points = np.vstack([history_deg, future_deg, predicted_deg.reshape(-1, 2)])
    lon_min, lat_min = np.nanmin(all_points, axis=0)
    lon_max, lat_max = np.nanmax(all_points, axis=0)

    lon_span = max(lon_max - lon_min, 1e-6)
    lat_span = max(lat_max - lat_min, 1e-6)
    target_width = lon_span / 0.6
    target_height = lat_span / 0.6

    desired_height_from_width = target_width * 3 / 4
    desired_width_from_height = target_height * 4 / 3

    final_width = max(target_width, desired_width_from_height)
    final_height = max(target_height, desired_height_from_width)

    lon_center = (lon_max + lon_min) / 2
    lat_center = (lat_max + lat_min) / 2

    lon_min_zoom = lon_center - final_width / 2
    lon_max_zoom = lon_center + final_width / 2
    lat_min_zoom = lat_center - final_height / 2
    lat_max_zoom = lat_center + final_height / 2

    fig_zoom, ax_zoom, plot_kwargs_zoom = _get_axes_with_raster_basemap(
        lon_min_zoom, lon_max_zoom, lat_min_zoom, lat_max_zoom, raster_path, figsize=(12, 9)
    )
    _plot_tracks_on_axis(ax_zoom, predicted_deg, history_deg, future_deg, plot_kwargs_zoom)
    zoom_dir = os.path.dirname(zoom_output_path)
    if zoom_dir:
        os.makedirs(zoom_dir, exist_ok=True)
    fig_zoom.tight_layout()
    fig_zoom.savefig(zoom_output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig_zoom)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config, args.checkpoint)
    model, eval_env, hyperparams = build_inference_components(config, args.checkpoint, device)
    predicted_trajs, history_raw, future_raw = prepare_batch_prediction(
        model,
        eval_env,
        hyperparams,
        args.sampling,
        args.step,
        args.scene_index,
        args.timestep_stride,
    )
    plot_paths(predicted_trajs, history_raw, future_raw, args.output, args.zoom_output)
    print(f"Visualization saved to {args.output} and {args.zoom_output}")


if __name__ == "__main__":
    main()

