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
from evaluation.evaluation import compute_ade_x_y_traj_each_time
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
    lon = points[..., 0] / 10.0 * 500.0 + 1300.0
    lat = points[..., 1] / 6.0 * 300.0 + 300
    return np.stack([lon / 10.0, lat / 10.0], axis=-1)


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

        _, _, _, predicted_trajs, gt_traj = compute_ade_x_y_traj_each_time(
            prediction_dict[ts_key][node_key], futures_dict[ts_key][node_key]
        )

        history_raw = histories_dict[ts_key][node_key]
        future_raw = futures_dict[ts_key][node_key]
        return predicted_trajs, history_raw, future_raw

    raise RuntimeError("No valid batches found for visualization.")


def select_probability_cone(pred_trajs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if pred_trajs.shape[0] < 2:
        return pred_trajs[0], pred_trajs[0]
    final_points = pred_trajs[:, -1, :]
    dists = np.linalg.norm(final_points[None, ...] - final_points[:, None, :], axis=-1)
    max_pair = np.unravel_index(np.argmax(dists), dists.shape)
    return pred_trajs[max_pair[0]], pred_trajs[max_pair[1]]


def plot_paths(
    predicted_trajs: np.ndarray,
    history: np.ndarray,
    future: np.ndarray,
    output_path: str,
):
    predicted_deg = denormalize_positions(np.asarray(predicted_trajs))
    history_deg = denormalize_positions(np.asarray(history)[:, :2])
    future_deg = denormalize_positions(np.asarray(future)[:, :2])

    cone_a, cone_b = select_probability_cone(predicted_deg)
    cone_coords = np.vstack([cone_a, cone_b[::-1]])

    lon_min, lon_max = 100, 180
    lat_min, lat_max = 0, 50

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

    ax.fill(
        cone_coords[:, 0],
        cone_coords[:, 1],
        color="limegreen",
        alpha=0.5,
        zorder=1,
        label="Probability cone",
        **plot_kwargs,
    )

    ax.plot(
        history_deg[:, 0],
        history_deg[:, 1],
        "o--",
        color="red",
        label="History",
        zorder=3,
        **plot_kwargs,
    )
    ax.plot(
        future_deg[:, 0],
        future_deg[:, 1],
        "o--",
        color="red",
        label="Future ground truth",
        zorder=3,
        **plot_kwargs,
    )
    ax.plot(
        history_deg[-1, 0],
        history_deg[-1, 1],
        "o",
        color="blue",
        label="Current position",
        markersize=8,
        zorder=4,
        **plot_kwargs,
    )

    for idx, traj in enumerate(predicted_deg):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            "o--",
            color="green",
            markersize=5,
            linewidth=1,
            label="Predicted trajectories" if idx == 0 else None,
            zorder=6,
            **plot_kwargs,
        )

    ax.legend(loc="best")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


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
    plot_paths(predicted_trajs, history_raw, future_raw, args.output)
    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()

