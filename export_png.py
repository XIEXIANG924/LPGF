"""Export Fig. 5 as PNG for DOCX embedding."""
import numpy as np; import pandas as pd; from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial','Helvetica','sans-serif'],
    'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,'axes.labelsize':7.5,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.6,'legend.frameon':False})

BASE=Path("/sessions/charming-hopeful-galileo/mnt/uav_research/02data_transf")
R=800;HH=120;RW=45
XR=np.linspace(-750,750,160);YR=-0.2*XR
ND=np.array([0.2,1.0])/np.sqrt(1.04)
TC={'industrial':'#7B3294','residential':'#008837','mid_rise':'#E08214'}
UC='#00A6D6';RF='#4D5663';LY='#F6C85F'

df=pd.read_csv(BASE/'output'/'buildings_shadow_v5.csv',encoding='utf-8-sig')
df['E']=df['UTM_E']-511525.7;df['N']=df['UTM_N']-5026509.1;df['H']=df['height_m']

fig=plt.figure(figsize=(14,6),facecolor='white')
a3=fig.add_subplot(1,2,1,projection='3d')
a2=fig.add_subplot(1,2,2)
a3.set_facecolor('white')
for ax in[a3.xaxis,a3.yaxis,a3.zaxis]:ax.pane.fill=False;ax.pane.set_edgecolor('#D9D9D9')
a3.grid(True,color='#DADADA',lw=0.35,alpha=0.5)
nx,ny=ND
ys=np.vstack([YR-RW/2*nx,YR+RW/2*nx]);xs=np.vstack([XR-RW/2*ny,XR+RW/2*ny])
a3.plot_surface(xs,ys,np.zeros_like(xs),color=RF,alpha=0.30,shade=False)
a3.plot(XR,YR,zs=0.05,color=LY,lw=0.9,ls='--',alpha=0.8)
for t in['industrial','residential','mid_rise']:
    s=df[df['sem_type']==t];c=TC[t]
    for _,r in s.iterrows():a3.plot([r['E'],r['E']],[r['N'],r['N']],[0,r['H']],color=c,lw=0.9,alpha=0.62)
    a3.scatter(s['E'],s['N'],0,color=c,s=12,alpha=0.35,edgecolors='none',depthshade=False)
    a3.scatter(s['E'],s['N'],s['H'],color=c,s=26,marker='^',alpha=0.92,edgecolors='white',linewidths=0.25,depthshade=False)
a3.plot([0,0],[0,0],[0,HH],color=UC,lw=1.35,alpha=0.86)
a3.scatter([0],[0],[HH],marker='*',s=105,color=UC,edgecolors='white',linewidths=0.45,depthshade=False)
th=np.linspace(0,2*np.pi,240)
a3.plot(R*np.cos(th),R*np.sin(th),zs=0,color='#8A8A8A',lw=0.65,ls='--',alpha=0.42)
a3.set_xlim(-R,R);a3.set_ylim(-R,R);a3.set_zlim(0,140)
a3.set_xlabel('East (m)');a3.set_ylabel('N (m)');a3.set_zlabel('')
a3.view_init(elev=28,azim=-55);a3.dist=8.8
a3.set_title('(a) 3D location prior',fontweight='bold',loc='left')
# 2D
a2.add_patch(Circle((0,0),R,fill=False,color='#8A8A8A',lw=0.75,ls='--',alpha=0.55,zorder=0))
for t in['industrial','residential','mid_rise']:
    s=df[df['sem_type']==t];sz=np.clip(np.sqrt(s['H'])*23,22,92)
    a2.scatter(s['E'],s['N'],s=sz,c=TC[t],alpha=0.82,edgecolors='white',linewidths=0.55,zorder=3)
xu=XR+RW/2*ny;yu=YR+RW/2*nx;xl=XR-RW/2*ny;yl=YR-RW/2*nx
rp=np.column_stack([np.hstack([xu,xl[::-1]]),np.hstack([yu,yl[::-1]])])
a2.add_patch(Polygon(rp,closed=True,facecolor=RF,alpha=0.35,edgecolor='#242A31',lw=0.55,zorder=5))
a2.plot(XR,YR,color=LY,lw=0.95,ls='--',alpha=0.82,zorder=6)
a2.scatter([0],[0],marker='*',s=105,c=UC,edgecolors='white',linewidths=0.55,zorder=10)
a2.set_xlim(-850,850);a2.set_ylim(-850,850);a2.set_aspect('equal')
a2.set_xlabel('East (m)');a2.set_ylabel('North (m)')
a2.grid(True,color='#DADADA',lw=0.35,alpha=0.62)
a2.set_title('(b) ENU topology map',fontweight='bold',loc='left')
fig.suptitle(f'MiTra A50 Milan — Location Prior (n={len(df)} buildings)',fontsize=10,fontweight='bold')
fig.tight_layout(pad=1.5)
out=str(BASE/'output'/'natureskill'/'fig05_a50_combined.png')
fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white')
print(f'PNG saved: {out}'); plt.close()
