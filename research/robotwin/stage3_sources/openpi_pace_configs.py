"""The 50 paper tasks; upstream OpenPI configurations remain available."""

# Imports in make_configs avoid the upstream registration cycle.
# ruff: noqa: PLC0415

TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)


def make_configs():
    # Called after upstream config.py has defined its types and built-in configs.
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.training import config as cfg
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    return [
        cfg.TrainConfig(
            name=f"pi05_robotwin_{task}",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
            data=cfg.LeRobotAlohaDataConfig(
                repo_id=f"robotwin/{task}",
                assets=cfg.AssetsConfig(asset_id="robotwin"),
                adapt_to_pi=False,
                use_delta_joint_actions=True,
                repack_transforms=transforms.Group(
                    inputs=[
                        transforms.RepackTransform(
                            {
                                "images": {
                                    "cam_high": "observation.images.cam_high",
                                    "cam_left_wrist": "observation.images.cam_left_wrist",
                                    "cam_right_wrist": "observation.images.cam_right_wrist",
                                },
                                "state": "observation.state",
                                "actions": "action",
                                "prompt": "prompt",
                            }
                        )
                    ]
                ),
                base_config=cfg.DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=optimizer.CosineDecaySchedule(
                warmup_steps=1000, peak_lr=2.5e-5, decay_steps=30000, decay_lr=2.5e-6
            ),
            optimizer=optimizer.AdamW(),
            ema_decay=0.99,
            seed=42,
            batch_size=32,
            num_train_steps=30005,
            fsdp_devices=1,
            save_interval=5000,
            keep_period=5000,
            num_workers=8,
            log_interval=100,
            assets_base_dir="./pace/runs/training-assets",
            checkpoint_base_dir="./pace/runs/checkpoints",
            policy_metadata={
                "task": task,
                "config": f"pi05_robotwin_{task}",
                "action_horizon": 50,
                "action_dim": 14,
                "asset_id": "robotwin",
                "action_space": "absolute_joint",
            },
        )
        for task in TASKS
    ]
