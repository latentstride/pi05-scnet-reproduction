"""CPU dependency and native library validation. No GPU rendering claims."""
from pathlib import Path
import argparse
import ctypes
import importlib
import importlib.metadata as md
import json
import hashlib
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
versions = {p:md.version(p) for p in ['numpy','scipy','torch','torchvision','sapien',
            'mplib','trimesh','open3d','nvidia-curobo','pytorch3d','warp-lang']}
assert versions['sapien'] == '3.0.0b1'
assert versions['torch'].split('+')[0] == '2.4.1'
assert versions['nvidia-curobo'] == '0.7.8'
modules = ['torch','torchvision','sapien','sapien.core','mplib','open3d','pytorch3d',
           'cv2','h5py','zarr','av','transforms3d','gymnasium','trimesh','quaternion',
           'curobo.curobolib.kinematics_fused_cu',
           'curobo.curobolib.geom_cu','curobo.curobolib.lbfgs_step_cu',
           'curobo.curobolib.line_search_cu','curobo.curobolib.tensor_step_cu','warp']
for name in modules:
    importlib.import_module(name)
ctypes.CDLL('libvulkan.so.1')
subprocess.run(['vulkaninfo','--help'],check=True,stdout=subprocess.DEVNULL)
import torch
import sapien
from pytorch3d.ops import knn_points
knn = knn_points(torch.tensor([[[0.,0.,0.]]]), torch.tensor([[[1.,0.,0.]]]), K=1)
assert torch.allclose(knn.dists, torch.ones_like(knn.dists))
# No render system: create a CPU PhysX scene and a primitive rigid body.
scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.1,0.1,0.1])
actor = builder.build(name='cpu_dependency_probe')
scene.step()
assert actor.get_pose() is not None
from curobo.util_file import get_robot_configs_path,join_path,load_yaml
robot = Path(join_path(get_robot_configs_path(),'ur5e.yml'))
assert robot.is_file()
assert 'robot_cfg' in load_yaml(str(robot))
result = {'cpu_imports_passed':True,'native_vulkan_loader_passed':True,
 'vulkan_tool_present':True,'cpu_physx_scene_passed':True,'curobo_config_present':True,
 'pytorch3d_cpu_native_op_passed':True,
 'versions':versions,'python':sys.version,'cuda_build':torch.version.cuda,
 'compiled_arch_list':os.environ.get('TORCH_CUDA_ARCH_LIST','8.9+PTX'),
 'gpu_runtime_validated':False,'rendering_validated':False,'planning_validated':False,
 'motion_gen_import_deferred': 'CuRobo 0.7.8 allocates a CUDA Pose in a class default at import; requires the later GPU allocation'}
import curobo
import pytorch3d
native = [*Path(curobo.__file__).parent.rglob('*.so'), *Path(pytorch3d.__file__).parent.glob('*.so')]
assert len(native) >= 6, 'Expected five CuRobo CUDA extensions and the PyTorch3D extension'
result['native_extensions'] = [{
    'path':str(p), 'bytes':p.stat().st_size,
    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in native]
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
