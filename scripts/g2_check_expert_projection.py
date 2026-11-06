import sys,json,itertools,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'scripts')
from g2_audit_expert_metadata import spawn
parser=argparse.ArgumentParser();parser.add_argument('--settled',type=Path);args=parser.parse_args()
root=Path('research/g2/expert-metadata-20261106'); results=[]
output=args.settled.parent if args.settled else root
settled=json.loads(args.settled.read_text()) if args.settled else None
for i in range(3):
 raw=Image.open(root/f'episode{i}_head_initial.jpg');rgb=np.array(raw.convert('RGB'));legacy=raw.info.get('comment')!=b'XPL-RGB1';rgb=rgb[:,:,::-1] if legacy else rgb;img=Image.fromarray(rgb);a=rgb.astype(int);draw=ImageDraw.Draw(img)
 cam=json.loads((root/f'episode{i}_cameras.json').read_text())['head_camera'];K=np.array(cam['intrinsic_cv']);E=np.array(cam['extrinsic_cv'])
 row={'episode':i,'legacy_jpeg_channel_reversal':bool(legacy),'scope':'2D_bbox_consistency_only_not_pose_measurement','objects':{}}
 for name,half,col,mask in [('box',[.03,.03,.1],'yellow',(a[:,:,0]>100)&(a[:,:,0]>a[:,:,1]*1.7)&(a[:,:,0]>a[:,:,2]*1.7)),('target_box',[.05,.05,.005],'cyan',(a[:,:,2]>70)&(a[:,:,2]>a[:,:,0]*1.7)&(a[:,:,2]>a[:,:,1]*1.7))]:
  p=spawn(i)[name]
  if settled:
   v=settled['records'][i]['snapshots']['aloha-agilex']['initial']['objects_task']['block' if name=='box' else 'target'];p=dict(position_m=v[:3],quaternion_wxyz=v[3:])
  w,x,y,z=p['quaternion_wxyz'];R=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
  pts=np.array(list(itertools.product(*[[-h,h] for h in half])))@R.T+np.array(p['position_m']);p2=(pts@E[:,:3].T+E[:,3])@K.T;p2=p2[:,:2]/p2[:,2,None]
  pred=np.r_[p2.min(axis=0),p2.max(axis=0)];pred=np.clip(pred,[0,0,0,0],[319,239,319,239]);yy,xx=np.where(mask);obs=np.array([xx.min(),yy.min(),xx.max(),yy.max()]);err=float(abs(pred-obs).max())
  row['objects'][name]={'predicted_bbox_px':pred.tolist(),'observed_color_bbox_px':obs.tolist(),'max_bbox_edge_difference_px':err,'color_pixel_count':len(xx)}
  draw.rectangle(tuple(pred),outline=col,width=2)
 img.resize((640,480)).save(output/f'episode{i}_projection.png');results.append(row)
(output/'image_consistency.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
