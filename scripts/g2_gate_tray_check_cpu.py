from pathlib import Path
import argparse,json
import numpy as np
import sapien
from g2_gate_scene_cpu import build
from g2_adapter import deadline
from robotwin_control_geometry import inverse_pose
from g2_episode import save_json

parser=argparse.ArgumentParser()
for name in ('assets','audit','config','input','reference','measured','output'):
    parser.add_argument('--'+name,required=True,type=Path)
args=parser.parse_args()
s=build(args.assets,args.audit,args.config,args.input,args.reference,args.output)
trace=np.load(args.measured/'physics-trace.npz',allow_pickle=False)
ticks=trace['sample_ticks']
contacts=trace['contacts_4ms']
first=int(np.flatnonzero(contacts[:,2:4].sum(axis=1)>1e-8)[0])
records=[]
for tick in (first-160,first-80,first-20,first,first+20,first+80,first+160):
    i=np.searchsorted(ticks,tick)
    arm_rows=trace['actual_native'][i].reshape(2,8)
    q=arm_rows[:,:6]
    measured=arm_rows[:,6:]
    intended=trace['drive_native'][i].reshape(2,8)
    for entity,arm,row in zip(s.entities,s.arms,arm_rows):
        names=[j.name for j in entity.get_active_joints()]
        values=entity.get_qpos()
        for name,value in zip(arm.chain.names+arm.grip.names,row):
            values[names.index(name)]=value
        entity.set_qpos(values)
    block=trace['block_pose'][i]
    s.block.set_pose(sapien.Pose(block[:3],block[3:]))
    s.contact.refresh()
    right_tool=s.contact.tools(q)[1]
    s.tracker.holders={'right':inverse_pose(right_tool)@s.block.pose.to_transformation_matrix()}
    s.tracker.measured_attachments={k:v.copy() for k,v in s.tracker.holders.items()}
    s.contact.set_start(measured,intended[:,6:])
    row=dict(tick=int(ticks[i]),actual_fingers=measured,drive_fingers=intended[:,6:],
             requested_right_gripper=float(trace['normalized_gripper_target'][i,1]))
    row['measured_clear']=s.contact.measured_clear(q,measured)
    row['measured_blocked']=s.contact.last_collisions
    try:
        predicted=s.contact.planning_geometry(intended[:,:6],intended[:,6:],measured,deadline(30))
        row['projected_fingers']=predicted
        row['command_clear']=s.contact(intended[:,:6],predicted)
        row['command_blocked']=s.contact.last_collisions
    except Exception as error:
        row['projection_error']=str(error)
        row['diagnostic']=getattr(error,'diagnostic',{})
    records.append(row)
save_json(args.output/'tray-check.json',dict(first_impulse_tick=first,scope='archived poses and drive targets checked with real FCL; no dynamic replay',records=records))
print('TRAY_CHECK',[(r['tick'],r['measured_clear'],r.get('command_clear'),r.get('projection_error')) for r in records])
