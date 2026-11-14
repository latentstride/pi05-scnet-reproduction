from pathlib import Path
import unittest,tempfile,importlib.util,json,io
import numpy as np,h5py
from PIL import Image
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pipeline',R/'code/analyze100.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Cases(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.old=m.R;m.R=Path(self.temp.name)
  for name in ['raw/aloha-agilex','data/episodes','dashboard/previews']:(m.R/name).mkdir(parents=True)
 def tearDown(self):m.R=self.old;self.temp.cleanup()
 def fixture(self):
  p=m.R/'raw/aloha-agilex/episode0.hdf5';im=Image.new('RGB',(320,240),(10,20,30));b=io.BytesIO();im.save(b,format='JPEG');bs=b.getvalue()
  with h5py.File(p,'w') as h:
   h['joint_action/vector']=np.zeros((2,14))
   for arm in ['left','right']:
    h[f'joint_action/{arm}_arm']=np.zeros((2,6));h[f'joint_action/{arm}_gripper']=np.zeros(2);h[f'endpose/{arm}_gripper']=np.zeros(2);h[f'endpose/{arm}_endpose']=np.tile([0,0,0,1,0,0,0],(2,1))
   for cam in ['head_camera','left_camera','right_camera']:
    h[f'observation/{cam}/rgb']=np.array([bs,bs],dtype=f'S{len(bs)}')
    for field,size in [('intrinsic_cv',(3,3)),('extrinsic_cv',(3,4)),('cam2world_gl',(4,4))]:h[f'observation/{cam}/{field}']=np.zeros((2,*size))
  return p
 def test_retains_normal_static_frames(self):
  r=m.analyze(self.fixture());self.assertEqual(r['status'],'read_ok');self.assertEqual(r['state_duplicate_rows'],1);self.assertEqual(r['rows_removed'],0)
 def test_corrupt_file_retained_as_failure(self):
  p=m.R/'raw/aloha-agilex/episode0.hdf5';p.write_bytes(b'invalid file');r=m.analyze(p);self.assertEqual(r['status'],'read_failed');self.assertTrue(p.exists())
 def test_missing_camera_is_not_silently_accepted(self):
  p=self.fixture()
  with h5py.File(p,'a') as h:del h['observation/head_camera/rgb']
  r=m.analyze(p);self.assertEqual(r['status'],'read_failed');self.assertIn('observation/head_camera/rgb',r['missing_fields'])
 def test_nonfinite_keeps_exact_anomaly_index(self):
  p=self.fixture()
  with h5py.File(p,'a') as h:h['joint_action/vector'][1,0]=float('nan')
  r=m.analyze(p);self.assertEqual(r['status'],'read_failed');self.assertEqual(r['numeric_nonfinite'],1);self.assertEqual(r['anomalies'][0]['indices'],[[1,0]])
 def test_gripper_outlier_flagged_not_deleted(self):
  p=self.fixture()
  with h5py.File(p,'a') as h:h['joint_action/vector'][1,6]=1.1;h['joint_action/left_gripper'][1]=1.1
  r=m.analyze(p);self.assertEqual(r['status'],'read_ok');self.assertEqual(r['gripper_out_of_range'],1);self.assertEqual(r['rows_removed'],0)
 def test_decode_error_index_retained(self):
  p=self.fixture()
  with h5py.File(p,'a') as h:h['observation/left_camera/rgb'][1]=b'bad'
  r=m.analyze(p);self.assertEqual(r['image_decode_errors'],1);self.assertTrue(any(x['type']=='image_decode' and x['frame']==1 for x in r['anomalies']))
if __name__=='__main__':unittest.main(verbosity=2)
