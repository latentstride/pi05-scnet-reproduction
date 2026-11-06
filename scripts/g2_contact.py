"""Diagnostic contact prediction and measured grasp state. Never replace drive intent."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import ceil
import numpy as np

from g2_adapter import G2Error, SIDES, positive
from g2_kinematics import pose_error
from robotwin_control_geometry import Status, checked_pose, finite_array, inverse_pose


@dataclass(frozen=True)
class ContactLimits:
    penetration_m: float = .002
    closure_penetration_m: float = .0005
    scan_step_m: float = .001
    position_slip_m: float = .005
    angle_slip_deg: float = 5.
    closed_chain_position_m: float = .002
    closed_chain_angle_deg: float = 2.
    stable_steps: int = 3
    minimum_impulse: float = 1e-8
    contact_cap: int = 256

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            positive(getattr(self,field))
        if self.closure_penetration_m>=self.penetration_m:
            raise ValueError("closure projection must leave penetration reserve")
        if type(self.stable_steps) is not int or type(self.contact_cap) is not int:
            raise ValueError("contact sample counts must be integers")

    @property
    def profile_id(self):
        import hashlib,json
        from dataclasses import asdict
        return "g2_contact_v1:"+hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


class ContactTracker:
    """A holder requires both actual fingers, sustained physical contact and impulse."""
    def __init__(self,block_body_id,finger_body_ids,limits=None):
        self.limits=limits or ContactLimits()
        self.block_body_id=block_body_id
        self.finger_body_ids={side:frozenset(finger_body_ids[side]) for side in SIDES}
        if any(len(value)!=2 for value in self.finger_body_ids.values()):
            raise ValueError("two actual finger body IDs per arm are required")
        if self.finger_body_ids["left"] & self.finger_body_ids["right"]:
            raise ValueError("finger identities must distinguish the two articulations")
        self.contact_fingers={side:set() for side in SIDES}
        self.counts={side:0 for side in SIDES}
        self.holders={}
        self.measured_attachments={}
        self.candidates={}
        self.releasing=set()
        self.last_stamp=None
        self.block_pose=None
        self.events=[]
        self.ever_held=False
        self.saw_shared_hold=False

    @property
    def carrying(self):
        return {side:self.measured_attachments.get(side,pose)
                for side,pose in self.holders.items() if side not in self.releasing}

    @classmethod
    def from_scene(cls,robot,block,arms,limits=None):
        ids={}
        for side,arm in zip(SIDES,arms):
            entity=getattr(robot,side+"_entity")
            ids[side]=[entity.find_joint_by_name(name).child_link.entity.per_scene_id
                       for name in arm.grip.names]
        return cls(block.per_scene_id,ids,limits)

    @property
    def phase(self):
        sides=set(self.holders)
        if len(sides)==2:return "both_holding"
        if sides:return next(iter(sides))+"_holding"
        return "free_after_hold" if self.ever_held else "approach"

    def observe(self,stamp,contacts,tools,block_pose):
        tools=finite_array(tools,(2,4,4))
        block_pose=checked_pose(block_pose)
        if self.last_stamp is not None:
            if stamp==self.last_stamp:
                return self.snapshot()
            if (stamp.episode!=self.last_stamp.episode or stamp.tick<=self.last_stamp.tick
                    or stamp.sim_s<=self.last_stamp.sim_s):
                raise G2Error(Status.UNSYNCED,"non-increasing contact clock")
        consecutive=self.last_stamp is None or stamp.tick==self.last_stamp.tick+1
        self.block_pose=block_pose.copy()
        seen={side:set() for side in SIDES}
        for contact in contacts:
            ids=[body.entity.per_scene_id for body in contact.bodies]
            if self.block_body_id not in ids:continue
            if not any(np.linalg.norm(finite_array(point.impulse,(3,)))>=self.limits.minimum_impulse
                       for point in contact.points):
                continue
            other=ids[1] if ids[0]==self.block_body_id else ids[0]
            for side in SIDES:
                if other in self.finger_body_ids[side]:seen[side].add(other)
        before=set(self.holders)
        failure=None
        for side,tool in zip(SIDES,tools):
            checked_pose(tool)
            complete=seen[side]==set(self.finger_body_ids[side])
            if not complete:
                if side in self.holders and side not in self.releasing:
                    failure=("confirmed holder lost bilateral contact",side)
                self.counts[side]=0
                self.candidates.pop(side,None)
                self.holders.pop(side,None)
                self.measured_attachments.pop(side,None)
                self.releasing.discard(side)
            elif side in self.holders:
                error=pose_error(block_pose,tool@self.holders[side])
                if side not in self.releasing and (np.linalg.norm(error[:3])>self.limits.position_slip_m or
                        np.rad2deg(np.linalg.norm(error[3:]))>self.limits.angle_slip_deg):
                    failure=("object slipped relative to confirmed holder",side)
            else:
                relative=inverse_pose(tool)@block_pose
                anchor=self.candidates.get(side,relative)
                error=pose_error(anchor,relative)
                stable=(consecutive and np.linalg.norm(error[:3])<=self.limits.position_slip_m
                        and np.rad2deg(np.linalg.norm(error[3:]))<=self.limits.angle_slip_deg)
                self.counts[side]=self.counts[side]+1 if stable else 1
                self.candidates[side]=anchor if stable else relative
                if self.counts[side]>=self.limits.stable_steps:
                    self.holders[side]=self.candidates[side].copy()
            if side in self.holders:
                # Prediction starts from the actual current block/tool relation;
                # cumulative slip is still checked against holders[side], the
                # unchanged acquisition reference, above.
                self.measured_attachments[side]=inverse_pose(tool)@block_pose
        self.contact_fingers=seen
        self.last_stamp=stamp
        after=set(self.holders)
        if after:self.ever_held=True
        if len(after)==2:self.saw_shared_hold=True
        if before!=after:
            self.events.append(dict(tick=stamp.tick,sim_s=stamp.sim_s,before=sorted(before),
                                    after=sorted(after),phase=self.phase))
        if failure:
            raise G2Error(Status.BACKEND_ERROR,failure[0],side=failure[1],contact_state=self.snapshot())
        return self.snapshot()

    def snapshot(self):
        return dict(phase=self.phase,holders=sorted(self.holders),
                    contact_fingers={side:sorted(ids) for side,ids in self.contact_fingers.items()},
                    stable_counts=dict(self.counts),saw_shared_hold=self.saw_shared_hold,
                    releasing=sorted(self.releasing),
                    block_pose_task=self.block_pose.tolist() if self.block_pose is not None else None,
                    tool_block={side:pose.tolist() for side,pose in self.holders.items()},
                    tool_block_measured={side:pose.tolist() for side,pose in self.measured_attachments.items()})

    def guard_opening(self,previous,target,arms,giver="left"):
        receiver="right" if giver=="left" else "left"
        if giver not in self.holders or receiver in self.holders:return
        i=SIDES.index(giver)
        direction=np.sign(np.asarray(arms[i].grip.opened)-np.asarray(arms[i].grip.closed))
        if np.any((target[i]-previous[i])*direction>1e-9):
            raise G2Error(Status.BACKEND_ERROR,"donor opening before receiver contact confirmation",
                          contact_state=self.snapshot())

    def authorize_opening(self,previous,target,arms,giver="left"):
        self.guard_opening(previous,target,arms,giver)
        for i,side in enumerate(SIDES):
            direction=np.sign(np.asarray(arms[i].grip.opened)-np.asarray(arms[i].grip.closed))
            if side in self.holders and np.any((target[i]-previous[i])*direction>1e-9):
                self.releasing.add(side)


class ContactWorld:
    """Adds task-specific rules around the unmodified articulation self-ignore policy."""
    def __init__(self,base,tracker,arms,frames,block_name,environment_names,
                 finger_link_names,support_names=(),project_actions=True,world_task=None):
        from mplib.collision_detection.fcl import CollisionRequest
        self.base,self.tracker,self.arms=base,tracker,tuple(arms)
        self.frames=[checked_pose(x) for x in frames]
        self.world_task=checked_pose(np.eye(4) if world_task is None else world_task)
        self.block_name=block_name
        self.environment_names=tuple(environment_names)
        self.finger_links=tuple(tuple(names) for names in finger_link_names)
        self.support_names=set(support_names)
        if any(len(x)!=2 for x in self.finger_links):raise ValueError("missing actual finger link names")
        self.model_names=[model.get_name() for model in base.models]
        self.allowed_pairs={(model,link) for model,links in zip(self.model_names,self.finger_links) for link in links}
        self.limits=tracker.limits
        self.profile_id=self.limits.profile_id
        self.request=CollisionRequest(num_max_contacts=self.limits.contact_cap,enable_contact=True)
        self.project_actions=project_actions
        self.failure=None
        self.projection_log=[]
        self.initial_fingers=None
        self.initial_drive=None
        self.giver="left"

    @property
    def last_collisions(self):
        return [self.failure] if self.failure else self.base.last_collisions

    @property
    def self_profile(self):return self.base.self_profile

    def refresh(self):self.base.refresh()

    def set_start(self,fingers,drive=None):
        self.initial_fingers=finite_array(fingers,(2,2)).copy()
        self.initial_drive=finite_array(drive if drive is not None else fingers,(2,2)).copy()

    def drive_start(self,fingers):
        if self.initial_fingers is None or not np.allclose(fingers,self.initial_fingers,atol=1e-6,rtol=0):
            raise G2Error(Status.UNSYNCED,"contact planner has no matching measured start")
        return self.initial_drive.copy()

    def depth(self,result):
        contacts=result.get_contacts()
        if not contacts or len(contacts)>=self.limits.contact_cap:return None
        depths=np.asarray([c.penetration_depth for c in contacts])
        if not np.isfinite(depths).all():return None
        return float(max(0.,depths.max()))

    def allows(self,row):
        if row["max_penetration_m"] is None or row["max_penetration_m"]>self.limits.penetration_m:
            return False
        a=(row["object_name1"],row["link_name1"])
        b=(row["object_name2"],row["link_name2"])
        if a[0]==self.block_name and b in self.allowed_pairs:return True
        if b[0]==self.block_name and a in self.allowed_pairs:return True
        # Only the object's shallow support contact is permitted, never robot/table contact.
        return ((a[0]==self.block_name and b[0] in self.support_names) or
                (b[0]==self.block_name and a[0] in self.support_names))

    def tools(self,q):
        return [frame@arm.chain.fk(values)@arm.offset()
                for frame,arm,values in zip(self.frames,self.arms,q)]

    def held_pose_world(self,q):
        tools=self.tools(q)
        candidates=[tools[SIDES.index(side)]@relative for side,relative in self.tracker.carrying.items()]
        if not candidates:return None
        for other in candidates[1:]:
            error=pose_error(candidates[0],other)
            if (np.linalg.norm(error[:3])>self.limits.closed_chain_position_m or
                    np.rad2deg(np.linalg.norm(error[3:]))>self.limits.closed_chain_angle_deg):
                raise G2Error(Status.COLLISION,"two holders imply inconsistent object poses")
        return self.world_task@candidates[0]

    @contextmanager
    def object_candidate(self,q):
        from mplib.collision_detection.fcl import FCLObject
        from sapien import Pose
        from g2_robotwin_backend import pose7
        original=self.base.world.get_object(self.block_name)
        if original is None:raise ValueError("task block missing from collision world")
        # FCLObject shapes are shared mutable collision objects: reconstruct at the
        # saved pose on exit, instead of trusting the old wrapper's shape transforms.
        original_pose=original.pose
        try:
            if self.tracker.carrying and not getattr(self,"_measured",False):
                value=pose7(self.held_pose_world(q))
                self.base.world.add_object(FCLObject(self.block_name,Pose(value[:3],value[3:]),
                                                     original.shapes,original.shape_poses))
            yield
        finally:
            self.base.world.add_object(FCLObject(self.block_name,original_pose,original.shapes,original.shape_poses))

    def held_environment_collisions(self):
        if not self.tracker.carrying:return []
        from mplib.collision_detection.fcl import collide
        from mplib.collision_detection import WorldCollisionResult
        block=self.base.world.get_object(self.block_name)
        results=[]
        for name in self.environment_names:
            if name==self.block_name:continue
            other=self.base.world.get_object(name)
            if other is None:raise ValueError("held-object obstacle missing: "+name)
            result=collide(block,other,self.request)
            if result.is_collision():
                results.append(WorldCollisionResult(result,"held_object_environment",
                                                      self.block_name,name,self.block_name,name))
        return results

    def __call__(self,q,fingers):
        self.failure=None
        try:
            return self.base.evaluate(q,fingers,contact=self)
        except G2Error as error:
            if error.status!=Status.COLLISION:raise
            self.failure=dict(reason=str(error),**error.diagnostic)
            return False

    def measured_clear(self,q,fingers):
        previous=getattr(self,"_measured",False)
        self._measured=True
        try:
            return self(q,fingers)
        finally:
            self._measured=previous

    def finger_depth(self,q,fingers,side,finger):
        from mplib.collision_detection.fcl import collide
        cached=getattr(self,"_projection_q",None)
        context=(nullcontext() if cached is not None and np.array_equal(q,cached) else self.object_candidate(q))
        with self.base.candidate(q,fingers),context:
            objects=self.base.models[side].get_fcl_model().get_collision_objects()
            matches=[obj for obj in objects if obj.name==self.finger_links[side][finger]]
            if len(matches)!=1:raise ValueError("finger FCL geometry missing or duplicated")
            result=collide(matches[0],self.base.world.get_object(self.block_name),self.request)
            return self.depth(result) if result.is_collision() else 0.

    def project_fingers(self,q,requested,check,previous=None):
        q=finite_array(q,(2,6));requested=finite_array(requested,(2,2))
        start=finite_array(self.initial_fingers if previous is None else previous,(2,2))
        if not self.project_actions:return requested.copy(),[]
        with self.projection_context(q):
            return self._project_fingers_at_object(q,requested,check,start)

    @contextmanager
    def projection_context(self,q):
        # Finger-only scans keep the arm q and carried-object pose fixed.
        # Restore both pose and context on success, rejection, or deadline.
        previous=getattr(self,"_projection_q",None)
        with self.object_candidate(q):
            self._projection_q=np.asarray(q).copy()
            try:yield
            finally:self._projection_q=previous

    def _project_fingers_at_object(self,q,requested,check,previous=None):
        """Predict bounded contact geometry; these values are not drive targets.

        A separate force-limited controller must preserve the original closure
        intent. The initial configuration is measured, then propagated through
        the predicted path; a fully-open configuration is not invented mid-grasp.
        """
        q=finite_array(q,(2,6))
        result=finite_array(requested,(2,2)).copy()
        start_fingers=finite_array(self.initial_fingers if previous is None else previous,(2,2))
        changes=[]
        if not self.project_actions:return result,changes
        for side,arm in enumerate(self.arms):
            for finger in range(2):
                opened=float(start_fingers[side,finger])
                desired=float(result[side,finger])
                if not arm.grip.limits[finger,0]<=desired<=arm.grip.limits[finger,1]:
                    raise G2Error(Status.LIMIT,"requested finger outside calibrated limit")
                count=max(1,int(ceil(abs(desired-opened)/self.limits.scan_step_m)))
                previous_alpha=0.
                threshold=self.limits.closure_penetration_m
                for index in range(count+1):
                    check()
                    alpha=index/count
                    value=opened+(desired-opened)*alpha
                    probe=result.copy()
                    probe[side,finger]=value
                    depth=self.finger_depth(q,probe,side,finger)
                    if depth is None:
                        raise G2Error(Status.COLLISION,"contact depth unknown or truncated")
                    if index==0:
                        if depth>self.limits.penetration_m:
                            raise G2Error(Status.COLLISION,"predicted finger start intersects task block",
                                          side=SIDES[side],finger=finger)
                        threshold=max(threshold,depth)
                    if depth>threshold+1e-12:
                        if index==0:
                            raise G2Error(Status.COLLISION,"predicted finger start intersects task block",
                                          side=SIDES[side],finger=finger)
                        lo,hi=previous_alpha,alpha
                        for _ in range(14):
                            check()
                            mid=(lo+hi)/2
                            probe[side,finger]=opened+(desired-opened)*mid
                            mid_depth=self.finger_depth(q,probe,side,finger)
                            if mid_depth is not None and mid_depth<=threshold:
                                lo=mid
                            else:hi=mid
                        result[side,finger]=opened+(desired-opened)*lo
                        changes.append(dict(side=SIDES[side],finger=finger,requested_native_m=desired,
                                            predicted_contact_native_m=float(result[side,finger]),
                                            rule="first_bounded_finger_block_contact",profile=self.profile_id))
                        break
                    previous_alpha=alpha
        self.projection_log=changes
        return result,changes

    def planning_geometry(self,q,requested,previous,check):
        return self.project_fingers(q,requested,check,previous)[0]

    def action_barrier(self,q,requested,check):
        self.tracker.guard_opening(self.initial_drive,requested,self.arms,self.giver)
        geometry,changes=self.project_fingers(q,requested,check)
        if any(change["side"] not in self.tracker.holders for change in changes):
            return True
        for i,side in enumerate(SIDES):
            direction=np.sign(np.asarray(self.arms[i].grip.opened)-np.asarray(self.arms[i].grip.closed))
            if side in self.tracker.holders and np.any((requested[i]-self.initial_drive[i])*direction>1e-9):
                return True
        return False

    def prepare_tools(self,tools,requested,current_tools):
        self.tracker.guard_opening(self.initial_drive,requested,self.arms,self.giver)
        i=SIDES.index(self.giver)
        direction=np.sign(np.asarray(self.arms[i].grip.opened)-np.asarray(self.arms[i].grip.closed))
        if len(self.tracker.holders)==2 and np.any((requested[i]-self.initial_drive[i])*direction>1e-9):
            # Separate release from withdrawal: physical contact must end before
            # the donor moves away. The executor replans on the measured event.
            return np.asarray(current_tools).copy(),"donor_release_before_withdrawal"
        return np.asarray(tools),None

    def ik_clear(self,q,fingers,check):
        try:
            projected,_=self.project_fingers(q,fingers,check)
            return self(q,projected)
        except G2Error as error:
            if error.status!=Status.COLLISION:raise
            self.failure=dict(reason=str(error),**error.diagnostic)
            return False


CONTACT_MODE="bounded_force_contact_diagnostic_v1"

def finger_force_limits(value):
    """Expand an explicit scalar or ordered [left, right] pair of force caps."""
    array=np.asarray(value,dtype=float)
    result=np.full(2,float(array)) if array.ndim==0 else finite_array(array,(2,)).copy()
    if not np.isfinite(result).all() or np.any(result<=0):
        raise ValueError("invalid per-side finger force limits")
    return result


def contact_profile(config,actual_block_mass_kg=None):
    """Explicit development parameters only; no defaults enter real execution."""
    profile=config["contact_profile"]
    if profile["status"]!="cpu_diagnostic_only" or config["embodiment"]!="ARX-X5":
        raise ValueError("contact profile is restricted to diagnostic ARX execution")
    values=profile["limits"]
    if set(values)!=set(ContactLimits.__dataclass_fields__):
        raise ValueError("all contact limits must be explicit")
    limits=ContactLimits(**values)
    if profile["giver"] not in SIDES or not profile["source"]:
        raise ValueError("contact direction and provenance are required")
    drive=profile["drive"]
    if not drive["source"] or set(drive)-{"mode"}!={"source","stiffness","damping","force_limit_n"}:
        raise ValueError("explicit force drive properties are required")
    if 'mode' in drive:
        from g2_finger_force import MODE
        if drive['mode']!=MODE:raise ValueError('unknown finger force controller')
    for name in ("stiffness","damping"):positive(drive[name])
    finger_force_limits(drive["force_limit_n"])
    # A force calibrated on the 10 g task is not interchangeable with the
    # historical 720 g stress fixture. Compare the actual PhysX mass at setup.
    if "validated_block_mass_kg" in profile:
        expected=positive(profile["validated_block_mass_kg"])
        if actual_block_mass_kg is not None:
            actual=positive(actual_block_mass_kg)
            if abs(actual-expected)>max(1e-8,expected*1e-6):
                raise ValueError("contact drive calibrated for different block mass")
    return limits,drive
