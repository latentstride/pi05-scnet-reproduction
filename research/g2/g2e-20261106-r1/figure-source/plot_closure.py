# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a,b) discrete closure PMF -> BarDistribution/plot_selfcorrection_math.py -> param inherit
# (c) empirical coverage -> LineTrend/plot_sweep.py -> param inherit
# Cross-type inheritance. Assets contain unrelated categorical/time-series toy data.
# Inherited two-spine, probability-axis, outside-legend and line semantics.
# RULE: native run uses a pre-rendered image. No panels here use native run.
# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
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

import argparse
from collections import Counter
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np


def render(selection, output, font_path, embodiment):
    fontManager.addfont(font_path)
    font=FontProperties(fname=font_path)
    mpl.rcParams['axes.unicode_minus']=False
    mpl.rcParams['font.sans-serif']=[font.get_name(),'Arial','DejaVu Sans']
    cases={name:rows for name,rows in selection['cases'].items() if ('A-native' in name)==(embodiment=='aloha')}
    fig=plt.figure(figsize=(183/25.4,147/25.4),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.15])
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
    handles=[];names=[]
    raw=json.loads(Path(selection['source']).read_text())
    for group,(name,rows) in enumerate(cases.items()):
        color=CATEGORICAL[[0,2,4,3][group]]
        title=f"种子{name[4]}，原窗口{name.split('w')[-1]}，n={rows['left']['blocks']}"
        for side,ax in zip(('left','right'),axes[:2]):
            index=0 if side=='left' else 1
            counts=Counter(x['first_close_index'][index] for x in raw[name])
            n=rows[side]['blocks']
            y=np.array([counts[i]/n for i in range(50)])
            line,=ax.step(range(50),y,where='mid',lw=1.05,color=color)
            ax.bar(52.5+group*.85,counts[None]/n,width=.65,color=color,alpha=.85)
            if side=='left':handles.append(line);names.append(title)
        left=rows['left']
        if left['blocks_with_close']:
            axes[2].step(range(1,51),[left['retained'][str(k)]/left['blocks_with_close'] for k in range(1,51)],
                         where='post',lw=1.2,color=color)
    for side,ax,label in zip(('左手','右手'),axes[:2],('a','b')):
        ax.set_title(f'{label}  {side}首次预测闭爪位置',loc='left',fontproperties=font)
        ax.set_xlabel('块内动作索引（从0开始）',fontproperties=font)
        ax.set_ylabel('占全部预测块比例',fontproperties=font)
        ax.set_xlim(-1,57);ax.set_xticks([0,10,20,30,40,49,54]);ax.set_xticklabels(['0','10','20','30','40','49','无'],fontproperties=font)
        ax.set_ylim(0,1.04 if side=='右手' or embodiment=='arx' else .32)
        ax.axvline(50.5,color=GREY,lw=.6,ls=':')
    ax=axes[2];ax.set_title('c  左手闭爪保留率（以块内有闭爪的预测块为分母）',loc='left',fontproperties=font)
    ax.set_xlim(1,50);ax.set_ylim(0,1.04)
    for k in (1,4):ax.axvline(k,color=GREY,lw=.65,ls=':')
    for k,color in zip(selection['selected_windows'],(CATEGORICAL[2],CATEGORICAL[3])):
        ax.axvline(k,color=color,lw=.8,ls='--');ax.text(k+.4,.06,f'K={k}',color=color,fontsize=7)
    for y in (.5,.9):ax.axhline(y,color=GREY,lw=.45,ls=':')
    ax.set_xlabel('执行窗口 K（保留索引 0 至 K-1）',fontproperties=font)
    ax.set_ylabel('保留率',fontproperties=font)
    fig.legend(handles,names,loc='outside lower center',ncol=2,prop=font,frameon=False)
    title='Aloha原生' if embodiment=='aloha' else 'ARX补充'
    fig.suptitle(f'{title}预测闭爪位置与执行窗口覆盖',fontproperties=font,fontsize=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');fig.canvas.draw();save_cns_figure(fig,output/f'{embodiment}-closure-distribution')
        errors=[str(w.message) for w in caught if 'Glyph' in str(w.message)]
    plt.close(fig)
    if errors:raise RuntimeError(errors)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('selection',type=Path);p.add_argument('output',type=Path);p.add_argument('--font',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    selection=json.loads(args.selection.read_text())
    for embodiment in ('aloha','arx'):render(selection,args.output,args.font,embodiment)
