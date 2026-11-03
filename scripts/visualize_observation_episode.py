"""Traceable 295-frame episode; no resampling or fabricated timestamps.
LineTrend production asset was inspected: cumulative publications/event annotations
are semantically incompatible. Inherit clean spines and line styling only.
"""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/"cache/matplotlib"))
# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)
import csv
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from observation_adapter import DATA, EpisodeAdapter, sha256


def main():
    out=DATA/'stage3_visualization'
    out.mkdir(exist_ok=True)
    with EpisodeAdapter('aloha-agilex','episode0') as adapter:
        q=adapter.file['joint_action/vector'][:]
        pose=np.stack([adapter.file[f'endpose/{a}_endpose'][:] for a in ['left','right']],axis=1)
        assert np.isfinite(q).all() and np.isfinite(pose).all()
        frames=[0,adapter.frames//2,adapter.frames-1]
        fig,axes=plt.subplots(3,3,figsize=(183/25.4,180/25.4))
        for row,frame in enumerate(frames):
            obs=adapter.observation(frame)
            for col,(camera,array) in enumerate(obs['images'].items()):
                rgb=array.transpose(1,2,0)
                Image.fromarray(rgb).save(out/f'frame_{frame:04d}_{camera}.png')
                axes[row,col].imshow(rgb, interpolation="none")
                axes[row,col].set_title(f'{camera}\nframe {frame}',fontsize=8)
                axes[row,col].axis('off')
        fig.suptitle('Recorded demonstration | aloha-agilex / episode0',fontsize=9)
        fig.subplots_adjust(top=.90,bottom=.02,left=.02,right=.98,hspace=.36,wspace=.08)
        fig.canvas.draw()
        save_cns_figure(fig,str(out/'three_view_frames'))
        fig.savefig(out/'three_view_frames.svg',bbox_inches='tight')
        plt.close(fig)
        # Small multiples preserve every dimension while using only 3 semantic colors.
        fig,axes=plt.subplots(7,2,figsize=(183/25.4,210/25.4),sharex=True)
        for arm in range(2):
            for j in range(7):
                ax=axes[j,arm]
                values=q[:,arm*7+j]
                ax.plot(np.arange(adapter.frames),values,color=CATEGORICAL[arm],lw=.8)
                ax.set_ylabel(f'joint {j+1} (rad)' if j<6 else 'gripper (0-1)')
                ax.set_xlim(0,adapter.frames-1)
                if j==0: ax.set_title(['Left: fl_joint1...6','Right: fr_joint1...6'][arm])
                if j==6: ax.set_xlabel('Frame index')
        fig.suptitle('recorded_joint_target | actual_qpos unavailable',fontsize=9)
        fig.tight_layout(rect=(0,0,1,.97))
        fig.canvas.draw()
        save_cns_figure(fig,str(out/'recorded_joint_target'))
        fig.savefig(out/'recorded_joint_target.svg',bbox_inches='tight')
        plt.close(fig)
        fig,axes=plt.subplots(7,2,figsize=(183/25.4,210/25.4),sharex=True)
        for arm in range(2):
            for j,label in enumerate(['x (m)','y (m)','z (m)','qw','qx','qy','qz']):
                ax=axes[j,arm]
                ax.plot(np.arange(adapter.frames),pose[:,arm,j],color=CATEGORICAL[arm],lw=.8)
                ax.set_ylabel(label)
                ax.set_xlim(0,adapter.frames-1)
                if j==0: ax.set_title(['Left recorded endpose','Right recorded endpose'][arm])
                if j==6: ax.set_xlabel('Frame index')
        fig.suptitle('recorded_endpose | world/tool convention retained',fontsize=9)
        fig.tight_layout(rect=(0,0,1,.97))
        fig.canvas.draw()
        save_cns_figure(fig,str(out/'recorded_endpose'))
        fig.savefig(out/'recorded_endpose.svg',bbox_inches='tight')
        plt.close(fig)
        # Raw plotted values for independent reconstruction, all 295 frames and 28 channels.
        with (out/'plotted_values.csv').open('w') as stream:
            writer=csv.writer(stream)
            writer.writerow(['frame']+[f'target_{i}' for i in range(14)]+[f'{a}_{v}' for a in ['left','right'] for v in ['x','y','z','qw','qx','qy','qz']])
            for i in range(adapter.frames): writer.writerow([i,*q[i],*pose[i].reshape(-1)])
        provenance={**adapter.provenance,'image_frame_indices':frames,'curve_frame_indices':[0,adapter.frames-1],
                    'all_curve_frames':adapter.frames,'downsampling':False,'time_axis':'frame index',
                    'image_operations':'official RGB decode, no contrast changes; thumbnails only in plate',
                    'source_fields':['joint_action/vector','endpose/left_endpose','endpose/right_endpose'],
                    'files':{p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='provenance.json'}}
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__': main()
