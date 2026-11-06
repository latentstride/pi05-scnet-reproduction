"""A separate CPU Aloha collision model sharing the actual task's actor geometry."""
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree as ET

from g2_collision import JointCollisionWorld


def collision_only_urdf(source, expected_sha256, output):
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("proxy URDF identity mismatch")
    root = ET.fromstring(raw)
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ("visual", "material"):
                parent.remove(child)
    for mesh in root.iter("mesh"):
        path = (source.parent / mesh.get("filename")).resolve()
        if not path.is_file():
            raise ValueError(f"missing proxy collision mesh: {path}")
        mesh.set("filename", str(path))
    output.write_bytes(ET.tostring(root))
    return output


class ProxyCollisionWorld:
    def __init__(self, actual_scene, source_urdf, expected_sha256, source_base, arms, output):
        import sapien
        from mplib.sapien_utils import SapienPlanningWorld
        from mplib.sapien_utils.conversion import convert_object_name
        self.actual_scene = actual_scene
        self.source_scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        loader = self.source_scene.create_urdf_loader()
        loader.fix_root_link = True
        path = collision_only_urdf(source_urdf, expected_sha256, output)
        self.entity = loader.load(str(path))
        self.entity.set_name("g2_aloha_proxy")
        self.entity.set_root_pose(source_base)
        self.world = SapienPlanningWorld(self.source_scene, [self.entity])
        self.model = self.world.get_articulation(convert_object_name(self.entity))
        names = [joint.get_name() for joint in self.entity.get_active_joints()]
        self.checker = JointCollisionWorld(self.world, [self.model, self.model], [names, names],
                                           [arm.chain.names for arm in arms], [arm.grip.names for arm in arms])
        self.refresh()

    def refresh(self):
        from sapien.physx import PhysxRigidBaseComponent
        from mplib.sapien_utils import SapienPlanningWorld
        self.world.update_from_simulation()
        for actor in self.actual_scene.get_all_actors():
            component = actor.find_component_by_type(PhysxRigidBaseComponent)
            if component is None:
                raise ValueError("task actor has no physical component")
            obj = SapienPlanningWorld.convert_physx_component(component)
            if obj is not None:
                self.world.add_object(obj)
        # The actual actor list must be fixed during an episode.
        names = tuple(sorted(actor.per_scene_id for actor in self.actual_scene.get_all_actors()))
        if hasattr(self, "actor_ids") and names != self.actor_ids:
            raise ValueError("task actors changed after collision-world construction")
        self.actor_ids = names

    @property
    def last_collisions(self):
        return self.checker.last_collisions

    def __call__(self, q, fingers):
        return self.checker(q, fingers)
