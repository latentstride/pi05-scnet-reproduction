"""Create a mass-scoped diagnostic profile only from three verified CPU replays."""
import argparse,hashlib,json,ast
from dataclasses import asdict
from pathlib import Path
import numpy as np
from g2_contact import ContactLimits,finger_force_limits
from g2_finger_force import MODE

def monitor_contract(path):
    tree=ast.parse(path.read_text())
    names={'ContactLimits','ContactTracker','finger_force_limits','contact_profile'}
    nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    if len(nodes)!=len(names):raise ValueError('missing contact monitoring contract')
    return hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()

def make_profile(acceptance,output):
    report=json.loads(acceptance.read_text());rows=report.get('episodes',[])
    if (report.get('data_kind')!='diagnostic' or not report.get('all_verified') or
            sorted(r['episode'] for r in rows)!=[0,1,2] or
            not all(r['task_success'] and r['hold_success'] and r['checks'].get('all_control_forces_bounded') and
                    r['checks'].get('force_controller_identity') for r in rows)):
        raise ValueError('three complete bounded-force/contact/task verifications required')
    caps=finger_force_limits(rows[0]['gripper_force_limit_n'])
    if not all(np.array_equal(finger_force_limits(r['gripper_force_limit_n']),caps) for r in rows):
        raise ValueError('per-episode force tuning cannot form one profile')
    captured=acceptance.parent/'runtime-sources'
    contract=monitor_contract(captured/'g2_contact.py')
    if (contract!=monitor_contract(Path(__file__).parent/'g2_contact.py') or
            (captured/'g2_finger_force.py').read_bytes()!=(Path(__file__).parent/'g2_finger_force.py').read_bytes()):
        raise ValueError('captured force/monitoring contract differs from selected profile')
    profile=dict(status='cpu_diagnostic_only',giver='left',validated_block_mass_kg=.01,
        source='original10g ARX expert CPU replays0/1/2; task/hold/continuous contact guard passed',
        limits=asdict(ContactLimits()),drive=dict(stiffness=1000.,damping=200.,force_limit_n=caps.tolist(),mode=MODE,
        source='bounded aperture/center effort allocation; control effort excludes separate passive compensation'),
        acceptance_sha256=hashlib.sha256(acceptance.read_bytes()).hexdigest(),
        physical_trace_sha256=[r['physical_trace_sha256'] for r in rows],
        monitor_contract_sha256=contract,calibration_scope='finger force allocation and contact monitoring; planning geometry is verified separately',
        formal_release_ready=False,production_collision_gate_verified=all(r['production_collision_gate_exercised'] for r in rows))
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix('.tmp')
    temporary.write_text(json.dumps(profile,indent=2)+'\n');temporary.replace(output)
    return profile

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();make_profile(a.acceptance,a.output)
