import logging
import hydra
import time
import json

from pathlib import Path
from pprint import pformat
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
from torch.cuda.amp import GradScaler

from datasets import Dataset
from safetensors.torch import load_file, safe_open

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle, hf_transform_to_torch
from lerobot.common.datasets.transforms import get_image_transforms
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
    set_global_seed,
)

from lerobot.scripts.train import (
    make_optimizer_and_scheduler,
    update_policy,
    log_train_info,
)


def load_stats_from_safetensors(filename):
    result = {}
    with safe_open(filename, framework="pt", device="cpu") as f:
        for k in f.keys():
            # NOTE: Assume k is of the form "key/stat"
            key = k.split("/")[0]
            if key not in result:
                result[key] = {}

            stat = k.split("/")[1]
            result[key][stat] = f.get_tensor(k)
    return result


def make_dataset_from_local(cfg, root_dir=".", split: str = "train") -> LeRobotDataset:
    resolve_delta_timestamps(cfg)

    image_transforms = None
    if cfg.training.image_transforms.enable:
        cfg_tf = cfg.training.image_transforms
        image_transforms = get_image_transforms(
            brightness_weight=cfg_tf.brightness.weight,
            brightness_min_max=cfg_tf.brightness.min_max,
            contrast_weight=cfg_tf.contrast.weight,
            contrast_min_max=cfg_tf.contrast.min_max,
            saturation_weight=cfg_tf.saturation.weight,
            saturation_min_max=cfg_tf.saturation.min_max,
            hue_weight=cfg_tf.hue.weight,
            hue_min_max=cfg_tf.hue.min_max,
            sharpness_weight=cfg_tf.sharpness.weight,
            sharpness_min_max=cfg_tf.sharpness.min_max,
            max_num_transforms=cfg_tf.max_num_transforms,
            random_order=cfg_tf.random_order,
        )

    hf_dataset = Dataset.load_from_disk(Path(f"{cfg.dataset_repo_id}/hf_data"))
    hf_dataset.set_transform(hf_transform_to_torch)

    info = json.load(open(Path(f"{cfg.dataset_repo_id}/meta_data/info.json")))
    stats = load_stats_from_safetensors(Path(f"{cfg.dataset_repo_id}/meta_data/stats.safetensors"))
    episode_data_index = load_file(
        Path(f"{cfg.dataset_repo_id}/meta_data/episode_data_index.safetensors")
    )

    dataset = LeRobotDataset.from_preloaded(
        transform=image_transforms,
        delta_timestamps=cfg.training.get("delta_timestamps"),
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        stats=stats,
        info=info,
        videos_dir=Path(f"{cfg.dataset_repo_id}/videos/"),
        video_backend=cfg.video_backend,
    )

    if cfg.get("override_dataset_stats"):
        for key, stats_dict in cfg.override_dataset_stats.items():
            for stats_type, listconfig in stats_dict.items():
                # example of stats_type: min, max, mean, std
                stats = OmegaConf.to_container(listconfig, resolve=True)
                dataset.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def train(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()
    logging.info(pformat(OmegaConf.to_container(cfg)))

    if cfg.training.online_steps > 0:  # and isinstance(cfg.dataset_repo_id, ListConfig):
        raise NotImplementedError("Online training not implemented.")

    if cfg.resume:
        raise NotImplementedError("Resuming training not implemented.")
    elif Logger.get_last_checkpoint_dir(out_dir).exists():
        raise RuntimeError(
            f"The configured output directory {Logger.get_last_checkpoint_dir(out_dir)} already exists. If "
            "you meant to resume training, please use `resume=true` in your command or yaml configuration."
        )

    if cfg.eval.batch_size > cfg.eval.n_episodes:
        raise ValueError(
            "The eval batch size is greater than the number of eval episodes "
            f"({cfg.eval.batch_size} > {cfg.eval.n_episodes}). As a result, {cfg.eval.batch_size} "
            f"eval environments will be instantiated, but only {cfg.eval.n_episodes} will be used. "
            "This might significantly slow down evaluation. To fix this, you should update your command "
            f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={cfg.eval.batch_size}`), "
            f"or lower the batch size (e.g. `eval.batch_size={cfg.eval.n_episodes}`)."
        )

    # log metrics to terminal and wandb
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)

    set_global_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_dataset")
    offline_dataset = make_dataset_from_local(cfg)
    if isinstance(offline_dataset, MultiLeRobotDataset):
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(offline_dataset.repo_id_to_index , indent=2)}"
        )

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.training.eval_freq > 0:
        raise NotImplementedError("Eval not implemented.")
        # logging.info("make_env")
        # eval_env = make_env(cfg)

    logging.info("make_policy")
    policy = make_policy(
        hydra_cfg=cfg,
        dataset_stats=offline_dataset.stats if not cfg.resume else None,
        pretrained_policy_name_or_path=str(logger.last_pretrained_model_dir)
        if cfg.resume
        else None,
    )
    assert isinstance(policy, nn.Module)
    # Create optimizer and scheduler
    # Temporary hack to move optimizer out of policy
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(enabled=cfg.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    # if cfg.resume:
    #     step = logger.load_last_training_state(optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.training.offline_steps=} ({format_big_number(cfg.training.offline_steps)})")
    logging.info(f"{cfg.training.online_steps=}")
    logging.info(
        f"{offline_dataset.num_samples=} ({format_big_number(offline_dataset.num_samples)})"
    )
    logging.info(f"{offline_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Note: this helper will be used in offline and online training loops.
    def evaluate_and_checkpoint_if_needed(step, is_online):
        _num_digits = max(6, len(str(cfg.training.offline_steps + cfg.training.online_steps)))
        step_identifier = f"{step:0{_num_digits}d}"

        # if cfg.training.eval_freq > 0 and step % cfg.training.eval_freq == 0:
        #     logging.info(f"Eval policy at step {step}")
        #     with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
        #         assert eval_env is not None
        #         eval_info = eval_policy(
        #             eval_env,
        #             policy,
        #             cfg.eval.n_episodes,
        #             videos_dir=Path(out_dir) / "eval" / f"videos_step_{step_identifier}",
        #             max_episodes_rendered=4,
        #             start_seed=cfg.seed,
        #         )
        #     log_eval_info(logger, eval_info["aggregated"], step, cfg, offline_dataset, is_online=is_online)
        #     if cfg.wandb.enable:
        #         logger.log_video(eval_info["video_paths"][0], step, mode="eval")
        #     logging.info("Resume training")

        if cfg.training.save_checkpoint and (
            step % cfg.training.save_freq == 0
            or step == cfg.training.offline_steps + cfg.training.online_steps
        ):
            logging.info(f"Checkpoint policy after step {step}")
            # Note: Save with step as the identifier, and format it to have at least 6 digits but more if
            # needed (choose 6 as a minimum for consistency without being overkill).
            logger.save_checkpont(
                step,
                policy,
                optimizer,
                lr_scheduler,
                identifier=step_identifier,
            )
            logging.info("Resume training")

    # create dataloader for offline training
    if cfg.training.get("drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.training.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()
    offline_step = 0
    for _ in range(step, cfg.training.offline_steps):
        if offline_step == 0:
            logging.info("Start offline training on a fixed dataset")

        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_s = time.perf_counter() - start_time

        for key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

        train_info = update_policy(
            policy,
            batch,
            optimizer,
            cfg.training.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.use_amp,
        )

        train_info["dataloading_s"] = dataloading_s

        if step % cfg.training.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset, is_online=False)

        # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
        # so we pass in step + 1.
        evaluate_and_checkpoint_if_needed(step + 1, is_online=False)

        step += 1
        offline_step += 1  # noqa: SIM113

    if cfg.training.online_steps == 0:
        if eval_env:
            eval_env.close()
        logging.info("End of training")
        return


@hydra.main(version_base="1.2", config_name="default", config_path="./configs")
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


if __name__ == "__main__":
    train_cli()
