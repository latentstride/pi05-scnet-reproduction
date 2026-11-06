"""CPU FCL collision checker built from both actual SAPIEN articulations and scene."""
from importlib.metadata import version
from hashlib import sha256
from pathlib import Path
from contextlib import contextmanager
import json

import numpy as np

from robotwin_control_geometry import finite_array


def audited_arx_self_profile(assets_root):
    """Retain the fixed asset's self-ignore policy, scoped to each articulation."""
    import yaml
    path=Path(assets_root)/"embodiments/ARX-X5/curobo_tmp.yml"
    expected="07c7faf92f6c9c7e3b7c370cb4487494d236935910cf6d89333f89f0b56c0ddb"
    if sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError("ARX self-collision policy asset drift")
    mapping=yaml.safe_load(path.read_text())["robot_cfg"]["kinematics"]["self_collision_ignore"]
    pairs=sorted({tuple(sorted((left,right))) for left,rights in mapping.items() for right in rights})
    return dict(source=str(path),sha256=expected,scope="same_articulation_only",pairs=pairs)


def physics_pair_enabled(a,b):
    """SAPIEN 3 PhysxCollisionShape filtering contract."""
    return bool(((a[0]&b[1]) or (a[1]&b[0])) and
                not ((a[2]&b[2]) and (a[3]&65535)==(b[3]&65535)))


def apply_audited_arx_physics(scene,entities,assets_root):
    """Match the existing native self profile in PhysX, with exact pair scope.

    One ignore bit per audited pair, distinct IDs per articulation; verify the
    complete scene shape-pair matrix before applying any change. Unexpected
    preconfigured ignore groups are rejected instead of overwritten.
    """
    profile=audited_arx_self_profile(assets_root)
    pairs={tuple(pair) for pair in profile['pairs']}
    if len(pairs)>32:raise ValueError('self profile exceeds PhysX ignore-bit capacity')
    def key(art):return tuple(sorted(link.entity.per_scene_id for link in art.get_links()))
    requested={key(art) for art in entities}
    rows=[];present=set()
    for art in scene.get_all_articulations():
        identity=key(art)
        if identity in requested:present.add(identity)
        names={link.name for link in art.get_links()}
        if identity in requested and any(not set(pair)<=names for pair in pairs):
            raise ValueError('native self profile references missing links')
        for link in art.get_links():
            for shape in link.collision_shapes:
                rows.append(dict(shape=shape,art=identity,name=link.name,old=list(shape.get_collision_groups())))
    if present!=requested:raise ValueError('requested articulation absent from scene')
    for actor in scene.get_all_actors():
        for component in actor.components:
            for shape in getattr(component,'collision_shapes',[]):
                rows.append(dict(shape=shape,art=None,name=actor.name,old=list(shape.get_collision_groups())))
    occupied={row['old'][3]&65535 for row in rows}
    ids={}
    for identity in sorted(requested):
        available=next((value for value in range(1,65536) if value not in occupied),None)
        if available is None:raise ValueError('no collision IDs available')
        ids[identity]=available;occupied.add(available)
    masks={name:0 for pair in pairs for name in pair}
    for index,pair in enumerate(sorted(pairs)):
        for name in pair:masks[name]|=1<<index
    for row in rows:
        row['new']=row['old'].copy()
        if row['art'] in requested:
            if row['old'][2]!=0:raise ValueError('unexpected existing physical ignore profile')
            row['new'][2]=masks.get(row['name'],0)
            row['new'][3]=(row['old'][3]&0xffff0000)|ids[row['art']]
    checked=0
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            exempt=(a['art'] in requested and a['art']==b['art'] and
                    tuple(sorted((a['name'],b['name']))) in pairs)
            expected=physics_pair_enabled(a['old'],b['old']) and not exempt
            if physics_pair_enabled(a['new'],b['new'])!=expected:
                raise ValueError('physical self profile would alter an unlisted collision pair')
            checked+=1
    changed=[]
    try:
        for row in rows:
            if row['old']!=row['new']:
                changed.append(row);row['shape'].set_collision_groups(row['new'])
                if list(row['shape'].get_collision_groups())!=row['new']:
                    raise ValueError('physical collision filter readback mismatch')
    except Exception:
        for row in reversed(changed):row['shape'].set_collision_groups(row['old'])
        raise
    return dict(profile_sha256=profile['sha256'],scope=profile['scope'],pairs=profile['pairs'],
                verified_scene_shape_pairs=checked,shapes_changed=len(changed),
                articulations=[dict(link_ids=list(identity),group_id=value) for identity,value in ids.items()],
                shapes=[dict(link=row['name'],articulation=list(row['art']),before=row['old'],after=row['new'])
                        for row in changed])


class JointCollisionWorld:
    def __init__(self, world, models, joint_names, arm_names, finger_names):
        if len(models) != 2 or any(len(group) != 2 for group in (joint_names, arm_names, finger_names)):
            raise ValueError("two role-specific arm mappings are required")
        self.world, self.models = world, tuple(models)
        self.indices = []
        for actual, arm, fingers in zip(joint_names, arm_names, finger_names):
            if len(set(actual)) != len(actual):
                raise ValueError("duplicate active joint name within an articulation")
            if len(set(arm + fingers)) != 8:
                raise ValueError("six arm joints and two independent finger joints required")
            self.indices.append([actual.index(name) for name in arm + fingers])
        if models[0] is models[1] and set(self.indices[0]).intersection(self.indices[1]):
            raise ValueError("shared articulation arm mappings overlap")
        self.last_collisions = []
        self.last_raw_collisions = []
        self.last_ignored_collisions = []
        self.self_profile = None
        self.audit_path = None
        self.calls = 0

    @classmethod
    def from_scene(cls, scene, entities, arm_names, finger_names):
        if version("mplib") != "0.2.1" or version("sapien") != "3.0.0b1":
            raise ValueError("collision backend version differs from source audit")
        from mplib.sapien_utils import SapienPlanningWorld
        from mplib.sapien_utils.conversion import convert_object_name
        world = SapienPlanningWorld(scene, list({id(entity): entity for entity in entities}.values()))
        models = [world.get_articulation(convert_object_name(entity)) for entity in entities]
        names = [[joint.get_name() for joint in entity.get_active_joints()] for entity in entities]
        return cls(world, models, names, arm_names, finger_names)

    def use_self_profile(self,profile):
        if profile["scope"]!="same_articulation_only" or len(profile["sha256"])!=64:
            raise ValueError("invalid scoped self-collision policy")
        self.self_profile=profile
        self.self_pairs={tuple(sorted(pair)) for pair in profile["pairs"]}
        self.model_names={model.get_name() for model in self.models}

    def refresh(self):
        """Must run before planning each prefix: objects and roots may have moved."""
        self.world.update_from_simulation()

    @contextmanager
    def candidate(self,q,fingers):
        q,fingers=finite_array(q,(2,6)),finite_array(fingers,(2,2))
        unique={id(model):model for model in self.models}
        old={key:np.asarray(model.get_qpos()).copy() for key,model in unique.items()}
        candidate={key:state.copy() for key,state in old.items()}
        try:
            for model,indices,values in zip(self.models,self.indices,np.c_[q,fingers]):
                candidate[id(model)][indices]=values
            for key,model in unique.items():
                model.set_qpos(candidate[key],full=True)
            yield
        finally:
            for key,model in unique.items():
                model.set_qpos(old[key],full=True)

    def classify(self,typed,contact=None):
        self.last_raw_collisions=[]
        self.last_collisions=[]
        self.last_ignored_collisions=[]
        for value in typed:
            row=dict(collision_type=str(value.collision_type),
                     object_name1=value.object_name1,link_name1=value.link_name1,
                     object_name2=value.object_name2,link_name2=value.link_name2)
            if contact is not None:
                row["max_penetration_m"]=contact.depth(value.res)
            self.last_raw_collisions.append(row)
            internal=(self.self_profile is not None and row["object_name1"]==row["object_name2"]
                      and row["object_name1"] in self.model_names
                      and tuple(sorted((row["link_name1"],row["link_name2"]))) in self.self_pairs)
            if internal:
                self.last_ignored_collisions.append(dict(row,rule_sha256=self.self_profile["sha256"]))
            elif contact is not None and contact.allows(row):
                self.last_ignored_collisions.append(dict(row,rule="bounded_task_contact",
                                                         profile=contact.profile_id))
            else:
                self.last_collisions.append(row)
        return not bool(self.last_collisions)

    def evaluate(self,q,fingers,contact=None):
        q,fingers=finite_array(q,(2,6)),finite_array(fingers,(2,2))
        from contextlib import nullcontext
        with self.candidate(q,fingers):
            context=contact.object_candidate(q) if contact is not None else nullcontext()
            with context:
                typed=(self.world.check_collision(contact.request) if contact is not None
                       else self.world.check_collision())
                if contact is not None:
                    typed+=contact.held_environment_collisions()
                self.calls+=1
                verdict=self.classify(typed,contact)
                if self.audit_path is not None:
                    record=dict(call=self.calls,q=q.tolist(),fingers=fingers.tolist(),
                                raw=self.last_raw_collisions,ignored=self.last_ignored_collisions,
                                blocked=self.last_collisions,
                                self_profile_sha256=self.self_profile["sha256"] if self.self_profile else None)
                    with Path(self.audit_path).open("a") as stream:
                        stream.write(json.dumps(record,allow_nan=False)+"\n")
                return verdict

    def __call__(self,q,fingers):
        return self.evaluate(q,fingers)
