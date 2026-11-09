from pathlib import Path
import os,json,csv
R=Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(R/'build/mpl')
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
f='/System/Library/Fonts/STHeiti Light.ttc';fontManager.addfont(f)
plt.rcParams.update({'font.family':FontProperties(fname=f).get_name(),'font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
B='#2166AC';O='#D58B38';T='#127C80';G='#627486';S=json.loads((R/'data/summary.json').read_text());Q=list(csv.DictReader((R/'data/trajectory_quality.csv').open()));M=json.loads((R/'data/mapping_summary_full.json').read_text());I=list(csv.DictReader((R/'data/ik_results_full.csv').open()));F=R/'figures'
def save(fig,name):
 fig.tight_layout(pad=1.5)
 for ext in ['png','pdf','svg']:fig.savefig(F/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.5))
for j,(r,c) in enumerate([('aloha-agilex',B),('arx-x5',O)]):
 v=np.array([int(x['frames']) for x in Q if x['robot']==r]);ax.boxplot(v,positions=[j],widths=.35,showfliers=False,patch_artist=True,boxprops={'facecolor':c,'alpha':.25},medianprops={'color':c,'linewidth':2});ax.scatter(j+np.linspace(-.12,.12,len(v)),np.sort(v),color=c,s=13,alpha=.8)
ax.set(xticks=[0,1],xticklabels=['Aloha（n=50）','ARX（n=50）'],ylabel='每条轨迹记录帧数 / 帧');save(fig,'100_lengths')
fig,axs=plt.subplots(1,2,figsize=(11,3.5));r=S['robots'];axs[0].bar(['Aloha','ARX'],[x['frames'] for x in r],color=[B,O]);axs[0].set(ylabel='记录帧总数 / 帧');
for j,x in enumerate(r):axs[0].text(j,x['frames']+200,f"{x['frames']:,}",ha='center')
axs[0].set_ylim(0,16000);x=np.arange(2);axs[1].bar(x-.18,[v['state_duplicate_rows'] for v in r],.35,label='重复状态',color=B);axs[1].bar(x+.18,[v['full_frame_duplicates'] for v in r],.35,label='完整重复帧',color=O);axs[1].set(xticks=x,xticklabels=['Aloha','ARX'],ylabel='保留记录数 / 条');axs[1].legend(frameon=False);save(fig,'100_counts_duplicates')
fig,axs=plt.subplots(1,2,figsize=(11,3.8))
for robot,c,lab in [('aloha-agilex',B,'Aloha→ARX'),('ARX-X5',O,'ARX→Aloha')]:
 ok=[x for x in I if x['source_robot']==robot and x['status']=='kinematic_solution'];v=np.sort([float(x['position_mm']) for x in ok]);axs[0].plot(v,np.arange(1,len(v)+1)/len(v),color=c,label=f'{lab}，n={len(v):,}');a=np.sort([float(x['angle_deg']) for x in ok]);axs[1].plot(a,np.arange(1,len(a)+1)/len(a),color=c)
axs[0].set(xlabel='有解样本位置残差 / mm',ylabel='累计比例',xlim=(0,.5));axs[1].set(xlabel='有解样本姿态残差 / °',ylabel='累计比例',xlim=(0,.05));axs[0].legend(frameon=False,fontsize=11);save(fig,'100_ik_errors')
fig,ax=plt.subplots(figsize=(8.5,3.2));gs=M['groups'];ax.barh(['Aloha→ARX','ARX→Aloha'],[g['solutions'] for g in gs],color=[B,T]);ax.barh(['Aloha→ARX','ARX→Aloha'],[g['attempted_arm_targets']-g['solutions'] for g in gs],left=[g['solutions'] for g in gs],color=O)
for j,g in enumerate(gs):ax.text(g['solutions']*.5,j,f"有解 {g['solutions']:,}",va='center',ha='center',color='white',fontsize=14);ax.text(g['attempted_arm_targets']+300,j,f"未解 {g['attempted_arm_targets']-g['solutions']}",va='center')
ax.set(xlabel='全部尝试的单臂目标数',xlim=(0,33000));save(fig,'100_ik_counts')

# Replot the archived CPU series, with identical physical tick semantics.
hist=json.loads((R/'data/research_results.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(11,3.8))
for branch,color,label in [('original',O,'原持续闭爪'),('production_lift100',T,'4 mm预载，10 cm抬升')]:
 rows=[r for r in hist['cpu_rows'] if r['branch']==branch and r['tick']>=5088]
 for ax,metric in zip(axs,['slip_mm','drive_left_mm']):
  ax.plot([r['time_since_hold_s'] for r in rows],[r[metric] for r in rows],'-o',ms=2.8,color=color,label=label)
axs[0].axhline(5,color=G,ls=':')
for ax,label in zip(axs,['工具–物块相对平移 / mm','单指位置驱动 / mm']):ax.set(xlabel='首次左持后的物理时间 / s',ylabel=label)
axs[0].legend(frameon=False,fontsize=10)
save(fig,'preload')
print('Exported five figure groups to PDF/SVG and 300 dpi PNG.')
