"""Exact-pair PhysX filter parity, without importing SAPIEN or allocating resources."""
from types import SimpleNamespace as NS
import unittest
from g2_fixtures import ASSETS
from g2_collision import apply_audited_arx_physics, physics_pair_enabled

class Shape:
    def __init__(self,groups=None):self.groups=groups or [1,1,0,0]
    def get_collision_groups(self):return self.groups.copy()
    def set_collision_groups(self,value):self.groups=list(value)

def fixture():
    arts=[];shapes={}
    for side,start in [('L',1),('R',101)]:
        links=[]
        for index,name in enumerate(['base_link']+['link'+str(i) for i in range(1,9)]):
            shape=Shape();shapes[side,name]=shape
            links.append(NS(name=name,entity=NS(per_scene_id=start+index),collision_shapes=[shape]))
        arts.append(NS(get_links=lambda links=links:links))
    obstacle=Shape([1,1,0xffffffff,1])
    scene=NS(get_all_articulations=lambda:arts,get_all_actors=lambda:[NS(name='obstacle',components=[NS(collision_shapes=[obstacle])])])
    return scene,arts,shapes,obstacle

class PhysicsFilterTests(unittest.TestCase):
    def test_native_pairs_only_same_arm_and_external_groups_preserved(self):
        scene,arts,s,obstacle=fixture();old=obstacle.get_collision_groups()
        result=apply_audited_arx_physics(scene,arts,ASSETS)
        enabled=lambda a,b:physics_pair_enabled(a.get_collision_groups(),b.get_collision_groups())
        self.assertFalse(enabled(s['L','base_link'],s['L','link2']))
        self.assertTrue(enabled(s['L','base_link'],s['R','link2']))
        self.assertTrue(enabled(s['L','link2'],s['L','link5']))
        self.assertTrue(enabled(s['L','base_link'],obstacle))
        self.assertTrue(enabled(s['L','link7'],s['R','link8']))
        self.assertEqual(obstacle.get_collision_groups(),old)
        self.assertGreater(result['verified_scene_shape_pairs'],100)
        self.assertNotEqual(result['articulations'][0]['group_id'],result['articulations'][1]['group_id'])

    def test_existing_physical_profile_rejected_before_any_mutation(self):
        scene,arts,s,_=fixture();s['R','link8'].groups[2]=4
        original={key:shape.get_collision_groups() for key,shape in s.items()}
        with self.assertRaisesRegex(ValueError,'existing physical ignore'):
            apply_audited_arx_physics(scene,arts,ASSETS)
        self.assertEqual(original,{key:shape.get_collision_groups() for key,shape in s.items()})

    def test_unrequested_robot_filters_unchanged(self):
        scene,arts,s,_=fixture()
        old={name:shape.get_collision_groups() for (side,name),shape in s.items() if side=='R'}
        apply_audited_arx_physics(scene,[arts[0]],ASSETS)
        self.assertEqual(old,{name:shape.get_collision_groups() for (side,name),shape in s.items() if side=='R'})
        self.assertTrue(physics_pair_enabled(s['L','base_link'].groups,s['R','link2'].groups))

    def test_partial_write_failure_restores_all_groups(self):
        scene,arts,s,_=fixture()
        old={key:shape.get_collision_groups() for key,shape in s.items()}
        shape=s['L','link3'];original=shape.set_collision_groups
        failed=[False]
        def fail_once(value):
            if not failed[0]:failed[0]=True;raise RuntimeError('SDK rejected filter')
            original(value)
        shape.set_collision_groups=fail_once
        with self.assertRaises(RuntimeError):apply_audited_arx_physics(scene,arts,ASSETS)
        self.assertEqual(old,{key:shape.get_collision_groups() for key,shape in s.items()})

    def test_missing_articulation_rejected(self):
        scene,arts,s,_=fixture();other=NS(get_links=lambda:[NS(entity=NS(per_scene_id=999))])
        with self.assertRaises(ValueError):apply_audited_arx_physics(scene,[other],ASSETS)
