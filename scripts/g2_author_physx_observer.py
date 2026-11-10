"""Passive PhysX evidence recorder. No control, guard or policy dependencies."""

import json
import numpy as np
from sapien.physx import PhysxRigidDynamicComponent


FINGERS = ("fl_link7", "fl_link8", "fr_link7", "fr_link8")


class ObserveScene:
    """Only read the state after the original step returns. Never veto a step."""

    def __init__(self, scene, env, stream):
        self.original, self.env, self.stream = scene, env, stream
        self.tick = 0
        self.actor = env.box.actor
        self.block = self.actor.find_component_by_type(PhysxRigidDynamicComponent)
        self.entity = env.robot.left_entity
        self.links = {
            link.entity.name: link.entity
            for link in self.entity.links
            if link.entity.name in FINGERS
        }
        if set(self.links) != set(FINGERS):
            raise ValueError("missing native finger links")
        self.record()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def record(self):
        pose = self.actor.get_pose()
        block = np.asarray(pose.to_transformation_matrix(), dtype=float)
        contacts = []
        for contact in self.original.get_contacts():
            names = [b.entity.name for b in contact.bodies]
            if self.actor.name not in names:
                continue
            contacts.append(
                dict(
                    pair=names,
                    points=[
                        dict(
                            separation=float(p.separation),
                            normal=p.normal.tolist(),
                            impulse=p.impulse.tolist(),
                            position=p.position.tolist(),
                        )
                        for p in contact.points
                    ],
                )
            )
        joints = [j.get_name() for j in self.entity.get_active_joints()]
        qpos = np.asarray(self.entity.get_qpos())
        qvel = np.asarray(self.entity.get_qvel())
        row = dict(
            tick=self.tick,
            action=self.env.take_action_cnt,
            native_success=bool(self.env.eval_success),
            joint_names=joints if self.tick == 0 else None,
            qpos=qpos.tolist(),
            qvel=qvel.tolist(),
            robot_root=np.asarray(
                self.entity.get_root_pose().to_transformation_matrix()
            ).tolist(),
            block_pose=block.tolist(),
            block_linear_velocity=self.block.linear_velocity.tolist(),
            block_angular_velocity=self.block.angular_velocity.tolist(),
            contacts=contacts,
            finger_relative_to_block={
                name: (
                    np.linalg.inv(block) @ entity.get_pose().to_transformation_matrix()
                ).tolist()
                for name, entity in self.links.items()
            },
        )
        if self.tick == 0:
            row["target_pose"] = np.asarray(
                self.env.target_box.actor.get_pose().to_transformation_matrix()
            ).tolist()
            row["block_mass_kg"] = float(self.block.mass)
            row["step_period_s"] = float(self.original.get_timestep())
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")

    def step(self, *args, **kwargs):
        result = self.original.step(*args, **kwargs)
        self.tick += 1
        self.record()
        return result
