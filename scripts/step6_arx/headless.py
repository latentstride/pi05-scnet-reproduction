"""CPU acceptance only: omit Vulkan and visual components, not PhysX/task logic."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from g2_proxy_collision import collision_only_urdf


@contextmanager
def presentation_off(env, source_urdf, source_sha256, collision_urdf):
    import sapien
    from sapien.wrapper.actor_builder import ActorBuilder
    from sapien.wrapper.urdf_loader import URDFLoader

    cpu_urdf = collision_only_urdf(source_urdf, source_sha256, collision_urdf)
    original_load = URDFLoader.load

    def load(loader, filename, *args, **kwargs):
        if Path(filename).resolve() != source_urdf.resolve():
            raise ValueError("headless fixture must load the audited ARX URDF")
        return original_load(loader, str(cpu_urdf), *args, **kwargs)

    def scene():
        env.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        env.scene.set_timestep(1 / 250)
        env.scene.add_ground(0, render=False)
        env.scene.default_physical_material = env.scene.create_physical_material(0.5, 0.5, 0)
        env.direction_light_lst = []
        env.point_light_lst = []

    class NoVisual:
        def __init__(self, *args, **kwargs):
            pass

        def attach(self, shape):
            pass

    original_add_component = sapien.Entity.add_component

    def add_component(entity, component):
        if not isinstance(component, NoVisual):
            return original_add_component(entity, component)
        return entity

    with ExitStack() as stack:
        stack.enter_context(patch.object(URDFLoader, "load", load))
        # Native RoboTwin primitives directly attach RenderBodyComponents.
        # Keep the original actor creation, physical material, pose and mass code.
        stack.enter_context(patch.object(sapien.Entity, "add_component", add_component))
        for name in ("RenderMaterial", "RenderBodyComponent", "RenderShapeBox"):
            stack.enter_context(patch.object(sapien.render, name, NoVisual))
        stack.enter_context(patch.object(env, "setup_scene", scene))
        # Native camera setup advances one physics step before homing.
        stack.enter_context(patch.object(env, "load_camera", lambda **kwargs: env.scene.step()))
        stack.enter_context(patch.object(env, "_update_render", lambda: None))
        # ActorBuilder.build_entity skips the graphics component if no visuals exist.
        for name in dir(ActorBuilder):
            if name.startswith("add_") and (name.endswith("_visual") or name == "add_visual_from_file"):
                stack.enter_context(patch.object(ActorBuilder, name, lambda *args, **kwargs: None))
        yield
