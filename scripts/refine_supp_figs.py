import os, warnings, zipfile, math
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Ellipse, PathPatch
from matplotlib.path import Path
from scipy.stats import mannwhitneyu, kruskal, spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
import networkx as nx

OUT=os.environ.get('FIG_OUTDIR', '/mnt/data')
os.makedirs(OUT, exist_ok=True)
CSV=os.environ.get('FIG_DATA', os.path.join(OUT,'Merged_by_Sample_ID_and_Group(4).csv'))
df=pd.read_csv(CSV)
labelmap={'01-Ctrl':'Control','02-Family':'Family','03-VHL':'VHL','03-EGLN2':'EGLN2','04-EPAS1':'EPAS1','05-EPOR':'EPOR','06-HAH':'HAH','07-NDD':'NDD','08-NDD/PDE4':'NDD/PDE4'}
df['Group_clean']=df['Group'].map(labelmap).fillna(df['Group'])
order=['Control','Family','VHL','EGLN2','EPAS1','EPOR','HAH','NDD','NDD/PDE4']
colors={'Control':'#6b6b6b','Family':'#e39b27','VHL':'#7646a2','EGLN2':'#3e82c4','EPAS1':'#18aaa7','EPOR':'#40b86a','HAH':'#1e9c7c','NDD':'#9b6a54','NDD/PDE4':'#cbd600'}
clinical_cols=[c for c in list(df.columns[2:26]) if pd.api.types.is_numeric_dtype(df[c])]
protein_cols=[c for c in list(df.columns[26:1488]) if pd.api.types.is_numeric_dtype(df[c]) and (pd.to_numeric(df[c], errors='coerce')>0).any()]
metab_cols=[c for c in list(df.columns[1488:-1]) if pd.api.types.is_numeric_dtype(df[c]) and (pd.to_numeric(df[c], errors='coerce')>0).any()]

# Treat zeros as missing for omics in all preprocessing/QC
def missing_df(cols, zero_missing=False):
    X=df[cols].apply(pd.to_numeric, errors='coerce').copy()
    if zero_missing: X=X.mask(X==0)
    return X

def prep_matrix(cols, log=True, zero_missing=True):
    X=df[cols].apply(pd.to_numeric, errors='coerce').copy()
    if zero_missing: X=X.mask(X==0)
    if log:
        X=np.log10(X.clip(lower=1e-12))
    X=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=cols, index=df.index)
    X=pd.DataFrame(StandardScaler().fit_transform(X), columns=cols, index=df.index)
    return X
Xclin=prep_matrix(clinical_cols, log=False, zero_missing=False)
Xprot=prep_matrix(protein_cols, log=True, zero_missing=True)
Xmet=prep_matrix(metab_cols, log=True, zero_missing=True)
Xall=pd.concat([Xclin,Xprot,Xmet],axis=1)
# robust outliers: sample-wise missingness + multivariate distance, top two
omics_raw=pd.concat([missing_df(clinical_cols,False),missing_df(protein_cols,True),missing_df(metab_cols,True)],axis=1)
miss_sample=omics_raw.isna().mean(axis=1)
score=np.sqrt((Xall.clip(-5,5)**2).mean(axis=1)) + 2*StandardScaler().fit_transform(miss_sample.values.reshape(-1,1)).ravel()/10
outlier_idx=list(pd.Series(score,index=df.index).sort_values(ascending=False).head(2).index)
keep=~df.index.isin(outlier_idx)

def bh(p):
    p=np.asarray(p,float); n=len(p); idx=np.argsort(p); q=np.empty(n); prev=1.0
    for k,i in enumerate(idx[::-1], start=1):
        rank=n-k+1; prev=min(prev, p[i]*n/rank); q[i]=prev
    return np.minimum(q,1)

def clean(s, n=28):
    return str(s).replace('_',' ').replace(' (','\n(')[:n]

def panel_letter(ax, l): ax.text(-.12,1.05,l,transform=ax.transAxes,fontsize=18,weight='bold',va='top')

# simple label placement for volcano avoiding vertical overlap
def label_points(ax, pts, side, xlim, ylim, col):
    # pts has diff, mlogp, feature; choose top 8 by p
    pts=pts.sort_values('p').head(8).copy().sort_values('mlogp')
    if len(pts)==0: return
    minsep=(ylim[1]-ylim[0])*0.055
    ys=[]
    for y in pts['mlogp']:
        yy=float(y)
        for _ in range(50):
            if all(abs(yy-y0)>=minsep for y0 in ys): break
            yy+=minsep*0.6
        yy=min(ylim[1]-0.15,max(ylim[0]+0.1,yy)); ys.append(yy)
    ha='left' if side=='right' else 'right'
    for (_,r), yy in zip(pts.iterrows(), ys):
        x=float(r['diff']); xt=x+(.13 if side=='right' else -.13)
        xt=max(xlim[0]+.2,min(xlim[1]-.2,xt))
        ax.annotate(clean(r['feature'],24), xy=(x,r['mlogp']), xytext=(xt,yy), fontsize=5.3, color=col,
                    ha=ha, va='center', arrowprops=dict(arrowstyle='-', lw=.35, color=col, alpha=.6))

def volcano_stats(cols, group):
    X=prep_matrix(cols, log=True, zero_missing=True)
    g=(df.Group_clean==group).values
    rows=[]
    for c in cols:
        a=X.loc[g,c].dropna(); b=X.loc[~g,c].dropna()
        try: p=mannwhitneyu(a,b,alternative='two-sided').pvalue if len(a)>1 and len(b)>1 else 1
        except Exception: p=1
        rows.append((c,float(np.nanmedian(a)-np.nanmedian(b)),p))
    res=pd.DataFrame(rows,columns=['feature','diff','p'])
    res['q']=bh(res.p.values); res['mlogp']=-np.log10(res.p.clip(1e-300)); res['sig']=(res.q<.05)&(res['diff'].abs()>=.8)
    return res

def volcano_ax(ax,res,title,typ):
    inc=res.sig&(res['diff']>0); dec=res.sig&(res['diff']<0); ns=~res.sig
    ax.scatter(res.loc[ns,'diff'],res.loc[ns,'mlogp'],s=7,c='#b8b8b8',alpha=.7,lw=0)
    ax.scatter(res.loc[dec,'diff'],res.loc[dec,'mlogp'],s=10,c='#0b559f',alpha=.95,lw=0)
    ax.scatter(res.loc[inc,'diff'],res.loc[inc,'mlogp'],s=10,c='#b20d1c',alpha=.95,lw=0)
    ax.axvline(0,color='0.35',ls='--',lw=.75); ax.axhline(-np.log10(.05),color='0.35',ls='--',lw=.75)
    xlim=(-4.2,4.2) if typ=='proteome' else (-5.2,10.2)
    ylim=(0,max(3.2,min(8,res.mlogp.quantile(.996)+.8)))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    label_points(ax,res[res['diff']>0], 'right', xlim, ylim, '#7a0010')
    label_points(ax,res[res['diff']<0], 'left', xlim, ylim, '#064a8a')
    ax.set_title(title,fontsize=9,weight='bold',pad=4); ax.set_xlabel('Median difference (z-score)',fontsize=7); ax.set_ylabel('-log10 p',fontsize=7); ax.tick_params(labelsize=6)
    up=int(inc.sum()); down=int(dec.sum()); nsn=int(ns.sum())
    ax.text(.975,.105,f'Up: {up}\nDown: {down}\nNS: {nsn}',transform=ax.transAxes,ha='right',va='bottom',fontsize=6.8, color='#b20d1c')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    return up,down,nsn

def add_ellipse(ax, x, y, color, alpha=.18):
    if len(x)<3: return
    cov=np.cov(np.vstack([x,y])); vals, vecs=np.linalg.eigh(cov)
    vals=np.maximum(vals,1e-9); orderi=vals.argsort()[::-1]; vals=vals[orderi]; vecs=vecs[:,orderi]
    theta=np.degrees(np.arctan2(*vecs[:,0][::-1])); chi2=5.991 # 95% 2d
    width,height=2*np.sqrt(vals*chi2)
    ell=Ellipse((np.mean(x),np.mean(y)),width,height,angle=theta,fc=color,ec=color,lw=1.1,alpha=alpha)
    ax.add_patch(ell)

def pls_scores(X, mask):
    y=LabelEncoder().fit_transform(df.loc[mask,'Group_clean']); Y=pd.get_dummies(y).values
    pls=PLSRegression(n_components=2); return pls.fit_transform(X.loc[mask].values,Y)[0]

# Supplementary Figure 1 refined
fig=plt.figure(figsize=(17,11)); gs=GridSpec(2,4,figure=fig,wspace=.34,hspace=.36)
fig.suptitle('Supplementary Figure 1. Quality control, feature filtering, and cohort structure.',fontsize=16,weight='bold',y=.985)
# A vector workflow
ax=fig.add_subplot(gs[0,0]); ax.axis('off'); panel_letter(ax,'A')
ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.text(5,9.6,'Cohort and analysis workflow',ha='center',fontsize=9,weight='bold')
box=FancyBboxPatch((.25,5.9),9.5,3.1,boxstyle='round,pad=0.02,rounding_size=.18',fc='#f7f7f7',ec='#c7c7c7',lw=1); ax.add_patch(box)
ax.text(5,8.65,f'Initial merged cohort (n = {len(df)})',ha='center',fontsize=8.3,weight='bold')
for j,g in enumerate(order):
    x=.65+(j%5)*1.82; y=7.35-(j//5)*.92
    ax.add_patch(FancyBboxPatch((x,y),1.55,.58,boxstyle='round,pad=.04,rounding_size=.10',fc=colors[g],ec='0.35',lw=.55))
    ax.text(x+.775,y+.36,g,ha='center',va='center',fontsize=6.4,color='white',weight='bold')
    ax.text(x+.775,y+.13,f'n={sum(df.Group_clean==g)}',ha='center',va='center',fontsize=6.2,color='white')
# Exclusion vector inset
ax.add_patch(FancyArrowPatch((5,5.9),(5,5.2),arrowstyle='simple',mutation_scale=25,fc='0.55',ec='0.55',alpha=.75))
ax.add_patch(FancyBboxPatch((1.2,4.25),7.6,.68,boxstyle='round,pad=.05,rounding_size=.14',fc='#fff5f5',ec='#d02b2b',lw=.9,linestyle='--'))
for x in [1.7,2.05]:
    ax.add_patch(Circle((x,4.6),.13,fc='#c02b35',ec='none')); ax.add_patch(FancyBboxPatch((x-.17,4.32),.34,.22,boxstyle='round,pad=.02,rounding_size=.06',fc='#c02b35',ec='none'))
ax.text(5.15,4.60,'Excluded: 2 multivariate outliers',ha='center',va='center',fontsize=7.2,weight='bold')
ax.text(5.15,4.35,'removed from downstream analyses',ha='center',va='center',fontsize=6.4)
ax.add_patch(FancyArrowPatch((5,4.25),(5,3.65),arrowstyle='simple',mutation_scale=25,fc='0.55',ec='0.55',alpha=.75))
ax.add_patch(FancyBboxPatch((.25,.6),9.5,2.85,boxstyle='round,pad=0.02,rounding_size=.18',fc='#fbfbfb',ec='#c7c7c7',lw=1))
ax.text(5,3.1,f'Final analytic cohort (n = {len(df)-2})',ha='center',fontsize=8.3,weight='bold')
# Use corrected Family n=4 after outlier removal if outliers in Family? display counts from keep
for j,g in enumerate(order):
    n=int(((df.Group_clean==g)&keep).sum())
    x=.65+(j%5)*1.82; y=1.9-(j//5)*.76
    ax.add_patch(FancyBboxPatch((x,y),1.55,.48,boxstyle='round,pad=.035,rounding_size=.09',fc=colors[g],ec='0.35',lw=.5))
    ax.text(x+.775,y+.29,g,ha='center',va='center',fontsize=6.2,color='white',weight='bold')
    ax.text(x+.775,y+.10,f'n={n}',ha='center',va='center',fontsize=6.1,color='white')
for x,lab in [(2.2,'Clinical'),(5,'Proteome'),(7.8,'Metabolome')]:
    ax.add_patch(FancyBboxPatch((x-1.0,.82),2.0,.42,boxstyle='round,pad=.03,rounding_size=.08',fc='white',ec='#bbb',lw=.7))
    ax.text(x,.98,lab,ha='center',va='center',fontsize=6.5)
# B improved missingness ranked with nonzero scale
ax=fig.add_subplot(gs[0,1]); panel_letter(ax,'B')
miss=omics_raw.isna().mean(axis=1); ordidx=miss.sort_values().index; ranks=np.arange(1,len(df)+1)
cols=['#d71920' if i in outlier_idx else colors[df.loc[i,'Group_clean']] for i in ordidx]
ax.vlines(ranks,0,miss.loc[ordidx],color='0.75',lw=.9,zorder=1)
ax.scatter(ranks,miss.loc[ordidx],s=23,c=cols,edgecolor='white',lw=.25,zorder=3)
ax.set_ylim(0,max(.02,miss.max()*1.12)); ax.set_title('Sample missingness across the cohort',fontsize=9,weight='bold')
ax.set_xlabel('Samples (ranked by fraction missing)',fontsize=7); ax.set_ylabel('Fraction missing',fontsize=7); ax.tick_params(labelsize=7)
summ=f"Median retained {miss[keep].median():.3f}\nIQR retained {miss[keep].quantile(.25):.3f}–{miss[keep].quantile(.75):.3f}\nRemoved outliers {', '.join([f'{miss[i]:.3f}' for i in outlier_idx])}"
ax.text(.06,.78,summ,transform=ax.transAxes,fontsize=6.5,bbox=dict(boxstyle='round,pad=.4',fc='white',ec='0.8',alpha=.9))
for s in ['top','right']: ax.spines[s].set_visible(False)
# C zero missing for omics
ax=fig.add_subplot(gs[0,2]); panel_letter(ax,'C')
layers=[('Clinical',clinical_cols,'#595959',False),('Proteome',protein_cols,'#b13a45',True),('Metabolome',metab_cols,'#2b9a8a',True)]
vals=[]
for name,cols_l,col,zm in layers:
    raw=missing_df(cols_l,zm)
    vals.append(1-raw.isna().mean().mean())
ax.bar(range(3),vals,color=[x[2] for x in layers],edgecolor='0.25',lw=.6)
ax.set_ylim(0,1.05); ax.set_ylabel('Proportion non-missing',fontsize=8); ax.set_title('Layer completeness / feature retention',fontsize=9,weight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels([f'{n}\n(n features = {len(c)})' for n,c,_,_ in layers],fontsize=7)
for i,v in enumerate(vals): ax.text(i,v+.025,f'{v:.2f}',ha='center',fontsize=8)
ax.tick_params(labelsize=7); [ax.spines[s].set_visible(False) for s in ['top','right']]
# D feature missingness by layer zero missing
ax=fig.add_subplot(gs[0,3]); panel_letter(ax,'D')
data=[]
for name,cols_l,col,zm in layers:
    raw=missing_df(cols_l,zm); data.append(raw.isna().mean().values+1e-5)
parts=ax.violinplot(data,showmedians=False,showextrema=False,widths=.8)
for pc,(_,_,col,_) in zip(parts['bodies'],layers): pc.set_facecolor(col); pc.set_edgecolor('black'); pc.set_alpha(.8); pc.set_linewidth(.6)
ax.boxplot(data,widths=.18,showfliers=False,medianprops={'color':'black','lw':1},boxprops={'color':'white','lw':.8},whiskerprops={'lw':.8},capprops={'lw':.8})
ax.set_yscale('log'); ax.set_xticks(range(1,4)); ax.set_xticklabels([x[0] for x in layers],fontsize=7); ax.set_ylabel('Fraction missing (per feature)',fontsize=8); ax.set_title('Feature missingness by layer',fontsize=9,weight='bold')
for i,d in enumerate(data,1): ax.text(i,np.nanmax(d)*1.15,f'Median\n{np.nanmedian(d):.1e}',ha='center',va='bottom',fontsize=6.2)
ax.tick_params(labelsize=7); [ax.spines[s].set_visible(False) for s in ['top','right']]
# E sample correlation heatmap with clustering and group color bands; select informative features to avoid noise
ax=fig.add_subplot(gs[1,0]); panel_letter(ax,'E')
Xcorr=Xall.loc[keep].copy()
# select high variance features across final cohort, cap to 300 to emphasize structure
vars=Xcorr.var(axis=0).sort_values(ascending=False).head(350).index
Xc=Xcorr[vars]
corr=np.corrcoef(Xc.values)
# cluster on 1-corr
D=1-corr; D=np.maximum(D,0); np.fill_diagonal(D,0)
try:
    leaves=leaves_list(linkage(squareform(D,checks=False),method='average'))
except Exception:
    leaves=np.arange(corr.shape[0])
corr_o=corr[np.ix_(leaves,leaves)]
im=ax.imshow(corr_o,cmap='RdBu_r',vmin=.35,vmax=1,interpolation='nearest')
ax.set_title('Sample-to-sample correlation\n(final cohort)',fontsize=9,weight='bold'); ax.set_xticks([]); ax.set_yticks([])
# color bars on top/left
final_groups=df.loc[keep,'Group_clean'].reset_index(drop=True).iloc[leaves].values
for i,g in enumerate(final_groups):
    ax.add_patch(Rectangle((i-.5,-2.2),1,1.1,fc=colors[g],ec='none',clip_on=False))
    ax.add_patch(Rectangle((-2.2,i-.5),1.1,1,fc=colors[g],ec='none',clip_on=False))
cbar=plt.colorbar(im,ax=ax,fraction=.046,pad=.03); cbar.set_label('Pearson r',fontsize=6); cbar.ax.tick_params(labelsize=6)
# F/G PLS with labels and ellipses
for grid,mask,title,letter in [(gs[1,1],np.ones(len(df),dtype=bool),'Multivariate projection\n(before outlier removal)','F'),(gs[1,2],keep,'Multivariate projection\n(after outlier removal)','G')]:
    ax=fig.add_subplot(grid); panel_letter(ax,letter)
    sc=pls_scores(Xall,mask); dsub=df.loc[mask].reset_index(drop=True)
    for g in order:
        m=(dsub.Group_clean==g).values
        if m.sum()==0: continue
        add_ellipse(ax,sc[m,0],sc[m,1],colors[g],alpha=.16)
        ax.scatter(sc[m,0],sc[m,1],s=32,c=colors[g],edgecolor='white',lw=.45,label=f'{g} (n={m.sum()})',zorder=3)
        ax.text(np.median(sc[m,0]),np.median(sc[m,1]),g,fontsize=6.4,weight='bold',ha='center',va='center',bbox=dict(boxstyle='round,pad=.17',fc='white',ec=colors[g],alpha=.75,lw=.6),zorder=4)
    # mark outliers in F
    if letter=='F':
        for k,oi in enumerate(outlier_idx,1):
            loc=np.where(np.where(mask)[0]==oi)[0][0] if oi in np.where(mask)[0] else oi
            ax.scatter(sc[loc,0],sc[loc,1],marker='D',s=45,c='#d71920',edgecolor='black',lw=.4,zorder=5)
            ax.text(sc[loc,0]+.2,sc[loc,1],f'Outlier {k}',fontsize=5.8,color='#8c0000')
    ax.set_xlabel('PLS-DA 1',fontsize=7); ax.set_ylabel('PLS-DA 2',fontsize=7); ax.set_title(title,fontsize=9,weight='bold'); ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.2,loc='center left',bbox_to_anchor=(1.0,.5),frameon=False)
    for s in ['top','right']: ax.spines[s].set_visible(False)
# H centroid distances
ax=fig.add_subplot(gs[1,3]); panel_letter(ax,'H')
sc=pls_scores(Xall,keep); dsub=df.loc[keep].reset_index(drop=True); labs=[]; cents=[]
for g in order:
    m=(dsub.Group_clean==g).values
    if m.sum(): labs.append(g); cents.append(sc[m].mean(axis=0))
cents=np.vstack(cents); dist=np.sqrt(((cents[:,None,:]-cents[None,:,:])**2).sum(axis=2))
im=ax.imshow(dist,cmap='viridis',interpolation='nearest')
ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs))); ax.set_xticklabels(labs,rotation=45,ha='right',fontsize=6); ax.set_yticklabels(labs,fontsize=6); ax.set_title('Group centroid distances\n(final cohort)',fontsize=9,weight='bold')
plt.colorbar(im,ax=ax,fraction=.046,pad=.03).ax.tick_params(labelsize=6)
fig.savefig(f'{OUT}/Supplementary_Figure_1_refined.svg',format='svg',bbox_inches='tight')
plt.close(fig)

# Supplementary Figure 2 refined
fig=plt.figure(figsize=(16,10.9)); gs=GridSpec(3,3,figure=fig,wspace=.24,hspace=.30)
fig.suptitle('Supplementary Figure 2. Extended subgroup-specific metabolomic and proteomic signatures.',fontsize=16,weight='bold',y=.985)
counts=[]; panels=[('VHL',protein_cols,'proteome'),('VHL',metab_cols,'metabolome'),('EPOR',protein_cols,'proteome'),('EPOR',metab_cols,'metabolome'),('HAH',protein_cols,'proteome'),('HAH',metab_cols,'metabolome'),('NDD',protein_cols,'proteome'),('NDD',metab_cols,'metabolome')]
legend_handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#b20d1c',markersize=5,label='Increased'),plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#0b559f',markersize=5,label='Decreased'),plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#b8b8b8',markersize=5,label='Not significant')]
for i,(grp,cols_l,typ) in enumerate(panels):
    ax=fig.add_subplot(gs[i//3,i%3]); panel_letter(ax,chr(65+i)); res=volcano_stats(cols_l,grp); u,d,n=volcano_ax(ax,res,f'{grp} {typ}',typ); counts.append((grp,typ,u,d,n))
    if i in [0,2,4,6]: ax.legend(handles=legend_handles,fontsize=5.8,loc='upper right',frameon=False,handletextpad=.3,borderaxespad=.2)
ax=fig.add_subplot(gs[2,2]); ax.axis('off'); panel_letter(ax,'I')
ax.set_title('Summary of subgroup-specific signatures',fontsize=10,weight='bold',pad=4)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(.22,.91,'Decreased (downregulated)',ha='center',fontsize=8,color='#0b559f',weight='bold'); ax.text(.78,.91,'Increased (upregulated)',ha='center',fontsize=8,color='#b20d1c',weight='bold')
ax.plot([.04,.40],[.875,.875],color='#0b559f',lw=.8); ax.plot([.60,.96],[.875,.875],color='#b20d1c',lw=.8)
ax.text(.12,.82,'Proteins',ha='center',fontsize=7,color='#0b559f'); ax.text(.30,.82,'Metabolites',ha='center',fontsize=7,color='#0b559f'); ax.text(.50,.82,'Subgroup',ha='center',fontsize=7,weight='bold'); ax.text(.70,.82,'Proteins',ha='center',fontsize=7,color='#b20d1c'); ax.text(.88,.82,'Metabolites',ha='center',fontsize=7,color='#b20d1c')
countdict={(g,t):(u,d) for g,t,u,d,n in counts}; maxv=max([v for pair in countdict.values() for v in pair]+[1])
for y,g in zip([.72,.58,.44,.30],['VHL','EPOR','HAH','NDD']):
    pu,pdown=countdict[(g,'proteome')]; mu,md=countdict[(g,'metabolome')]
    ax.plot([.04,.96],[y-.07,y-.07],color='.86',lw=.7)
    for x,val,col,side in [(.14,pdown,'#0b559f','left'),(.30,md,'#9fc9df','left'),(.64,pu,'#b20d1c','right'),(.82,mu,'#e6a3a8','right')]:
        w=.13*val/maxv
        if side=='left':
            ax.text(x-.025,y,str(val),ha='right',va='center',fontsize=7); ax.add_patch(Rectangle((x,y-.027),w,.054,fc=col,ec='none',alpha=.95))
        else:
            ax.add_patch(Rectangle((x,y-.027),w,.054,fc=col,ec='none',alpha=.95)); ax.text(x+w+.02,y,str(val),ha='left',va='center',fontsize=7)
    ax.add_patch(FancyBboxPatch((.445,y-.038),.11,.076,boxstyle='round,pad=.01,rounding_size=.018',fc=colors[g],ec='none',alpha=.95)); ax.text(.50,y,g,ha='center',va='center',fontsize=8,color='white',weight='bold')
# legend bottom, no overlap
ax.text(.07,.12,'Significance thresholds:\n|Median difference| ≥ 0.8 (z-score)\nand FDR-adjusted p < 0.05',fontsize=6.7,ha='left',va='top')
for yy,lab,col in [(.12,'Significantly increased','#b20d1c'),(.07,'Significantly decreased','#0b559f'),(.02,'Not significant','#b8b8b8')]:
    ax.scatter(.66,yy,s=45,c=col,edgecolor='0.5' if col=='#b8b8b8' else 'none'); ax.text(.70,yy,lab,fontsize=6.7,va='center')
fig.savefig(f'{OUT}/Supplementary_Figure_2_refined.svg',format='svg',bbox_inches='tight')
plt.close(fig)

# Supplementary Figure 3: copy previous if exists (liked overall), regenerate clean from current original script not necessary
import shutil
src=f'{OUT}/Supplementary_Figure_3_real_all_data.svg'
if os.path.exists(src): shutil.copy(src, f'{OUT}/Supplementary_Figure_3_refined.svg')
else:
    open(f'{OUT}/Supplementary_Figure_3_refined.svg','w').write('')

# Supplementary Figure 4 refined
known=~df.Group_clean.isin(['Control','Family','NDD','NDD/PDE4'])

def loocv_cm(X, ax, title):
    Xk=X.loc[known]; y=df.loc[known,'Group_clean']; le=LabelEncoder(); yy=le.fit_transform(y)
    k=min(25,Xk.shape[1])
    pipe=make_pipeline(SimpleImputer(strategy='median'),SelectKBest(f_classif,k=k),LogisticRegression(max_iter=3000,class_weight='balanced',solver='liblinear'))
    try: pred=cross_val_predict(pipe,Xk,yy,cv=LeaveOneOut())
    except Exception: pred=np.zeros_like(yy)
    cm=confusion_matrix(yy,pred,labels=range(len(le.classes_))); acc=accuracy_score(yy,pred); row=cm/cm.sum(axis=1,keepdims=True)*100
    im=ax.imshow(row,cmap='Blues',vmin=0,vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,str(cm[i,j]) if cm[i,j] else '·',ha='center',va='center',fontsize=8,color='white' if row[i,j]>55 else '0.25')
    ax.set_xticks(range(len(le.classes_))); ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels([f'{x}\n(n={sum(y==x)})' for x in le.classes_],fontsize=5.8); ax.set_yticklabels([f'{x}\n(n={sum(y==x)})' for x in le.classes_],fontsize=5.8)
    ax.set_xlabel(f'Predicted label\nAccuracy: {acc:.2f}',fontsize=6.6); ax.set_ylabel('True label',fontsize=6.6); ax.set_title(title,fontsize=8,weight='bold')
    return im

def curved_edge(ax, p1, p2, color, lw, alpha=.45, rad=.18):
    ax.add_patch(FancyArrowPatch(p1,p2,connectionstyle=f'arc3,rad={rad}',arrowstyle='-',lw=lw,color=color,alpha=alpha,mutation_scale=1,zorder=1))

# helper find cols for scatters
def find_col(name):
    if name in df.columns: return name
    for c in df.columns:
        if c.lower()==name.lower(): return c
    for c in df.columns:
        if name.lower() in c.lower(): return c
    return None

fig=plt.figure(figsize=(17,10.8)); gs=GridSpec(2,4,figure=fig,wspace=.36,hspace=.42, height_ratios=[.92,1.15])
fig.suptitle('Supplementary Figure 4. Classifier robustness and expanded clinical-omics validation.',fontsize=16,weight='bold',y=.985)
for i,(X,title) in enumerate([(Xclin,'LOOCV confusion matrix\nClinical-only classifier'),(Xmet,'LOOCV confusion matrix\nMetabolome-only classifier'),(Xprot,'LOOCV confusion matrix\nProteome-only classifier'),(Xall,'LOOCV confusion matrix\nCombined-data classifier')]):
    ax=fig.add_subplot(gs[0,i]); panel_letter(ax,chr(65+i)); loocv_cm(X,ax,title)
# E top predictors
ax=fig.add_subplot(gs[1,0]); panel_letter(ax,'E')
y=df.loc[known,'Group_clean']; Xsel=Xall.loc[known]
rf=RandomForestClassifier(n_estimators=500,random_state=4,class_weight='balanced',max_features='sqrt')
rf.fit(Xsel,y); imp=pd.Series(rf.feature_importances_,index=Xsel.columns).sort_values(ascending=False).head(15)[::-1]
barcols=['#b13a45' if f in clinical_cols else '#2b9a8a' if f in metab_cols else '#245da3' for f in imp.index]
ax.barh(range(len(imp)),imp.values,color=barcols,edgecolor='white',lw=.3)
ax.set_yticks(range(len(imp))); ax.set_yticklabels([clean(x,20) for x in imp.index],fontsize=6)
ax.set_xlabel('Random forest importance',fontsize=7); ax.set_title('Top 15 predictors (combined model)',fontsize=8,weight='bold'); ax.tick_params(axis='x',labelsize=6); [ax.spines[s].set_visible(False) for s in ['top','right']]
# F network polished, no orphan nodes
ax=fig.add_subplot(gs[1,1:3]); panel_letter(ax,'F'); ax.axis('off')
# pick top clinical, metabolites, proteins with correlation edges among categories
clin_sel=clinical_cols[:]
met_sel=[]; prot_sel=[]
# Use RF top + named candidate labels where present
for n in ['Hypoxanthine','L-alanine','N-Acetylneuraminate','2-Oxoglutarate','Ornithine','Putrescine','Adenosine','Glucose','Lactate','Orotate','Nicotinamide','Kynurenine']:
    c=find_col(n)
    if c and c in metab_cols and c not in met_sel: met_sel.append(c)
for n in ['NDUFS1','PSMD12','EIF4A1','HINT1','OLA1','GYPC','RAB35','S100A8','NME1','EEF1A1','HPRT1']:
    c=find_col(n)
    if c and c in protein_cols and c not in prot_sel: prot_sel.append(c)
# supplement with important features
for f in imp.index:
    if f in metab_cols and f not in met_sel and len(met_sel)<11: met_sel.append(f)
    if f in protein_cols and f not in prot_sel and len(prot_sel)<11: prot_sel.append(f)
clin_sel=[c for c in ['Ferritina (mg/dL)','IST (%)','Transferrin (mg/dL)','Uric A (mg/dL)','Glucose (mg/dL)','vLac (mmol/L)','MPV (fL)','PLT (/microL)','Age'] if c in df.columns]
# build edges only strongest inter-layer correlations
raw=pd.concat([missing_df(clin_sel,False),missing_df(met_sel,True),missing_df(prot_sel,True)],axis=1)
# log omics columns
for c in met_sel+prot_sel: raw[c]=np.log10(raw[c].clip(lower=1e-12))
raw=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(raw),columns=raw.columns)
edges=[]
sets=[('Clinical',clin_sel),('Metabolites',met_sel),('Proteins',prot_sel)]
for A,colsA in [('Clinical',clin_sel)]:
    for B,colsB in [('Metabolites',met_sel),('Proteins',prot_sel)]:
        for a in colsA:
            for b in colsB:
                try: rho,p=spearmanr(raw[a],raw[b])
                except Exception: continue
                if np.isfinite(rho) and abs(rho)>=.42: edges.append((a,b,rho))
for a in met_sel:
    for b in prot_sel:
        try: rho,p=spearmanr(raw[a],raw[b])
        except Exception: continue
        if np.isfinite(rho) and abs(rho)>=.50: edges.append((a,b,rho))
# Keep top edges and nodes with edges
edges=sorted(edges,key=lambda x:abs(x[2]),reverse=True)[:70]
nodes=set([x for e in edges for x in e[:2]])
clin_n=[c for c in clin_sel if c in nodes]; met_n=[c for c in met_sel if c in nodes]; prot_n=[c for c in prot_sel if c in nodes]
# positions as arcs/layers
pos={}
for i,c in enumerate(clin_n): pos[c]=(.07, .88-i*(.76/max(1,len(clin_n)-1)))
for i,c in enumerate(met_n): pos[c]=(.93, .88-i*(.76/max(1,len(met_n)-1)))
for i,c in enumerate(prot_n): pos[c]=(.50, .08+i*(.34/max(1,len(prot_n)-1)))
# draw edges curvy
for a,b,rho in edges:
    if a not in pos or b not in pos: continue
    col=plt.cm.RdBu_r((rho+1)/2)
    lw=.5+2.5*abs(rho)
    rad=.18 if pos[a][0]<pos[b][0] else -.18
    curved_edge(ax,pos[a],pos[b],col,lw,alpha=.45,rad=rad)
# draw nodes and labels
for group_nodes,fc,label,ytitle in [(clin_n,'#b13a45','Clinical',.97),(met_n,'#2b9a8a','Metabolites',.97),(prot_n,'#245da3','Proteins',.02)]:
    for c in group_nodes:
        x,y0=pos[c]; ax.scatter(x,y0,s=95,c=fc,edgecolor='white',lw=.7,zorder=3)
        ha='right' if x<.2 else 'left' if x>.8 else 'center'
        dx=-.025 if x<.2 else .025 if x>.8 else 0
        dy=0 if x!=.5 else -.035
        ax.text(x+dx,y0+dy,clean(c,23),ha=ha,va='center' if x!=.5 else 'top',fontsize=6.3,zorder=4)
ax.text(.07,.98,'Clinical',color='#b13a45',ha='center',fontsize=9,weight='bold')
ax.text(.93,.98,'Metabolites',color='#2b9a8a',ha='center',fontsize=9,weight='bold')
ax.text(.50,.02,'Proteins',color='#245da3',ha='center',fontsize=9,weight='bold')
ax.set_title('Expanded clinical–metabolite–protein correlation network (Spearman ρ)',fontsize=8.5,weight='bold',pad=2)
# legend for edge
ax.text(.43,.92,'Edge color (ρ)',fontsize=6.5,ha='center')
for k,val in enumerate(np.linspace(-1,1,5)):
    ax.plot([.35+k*.04,.375+k*.04],[.885,.885],lw=4,color=plt.cm.RdBu_r((val+1)/2),solid_capstyle='butt')
ax.text(.34,.86,'−1',fontsize=5.8,ha='center'); ax.text(.53,.86,'+1',fontsize=5.8,ha='center')
# G-J scatter in nested subgrid to avoid overlap
sub=gs[1,3].subgridspec(2,2,wspace=.55,hspace=.70)
scatter_pairs=[('Emogas pHv','vLac (mmol/L)','Blood pH vs. Lactate','blood pH','Lactate (mmol/L)'),('Glucose (mg/dL)',find_col('D-Glucose') or 'D-Glucose','Serum glucose vs. RBC glucose','Serum glucose (mg/dL)','D-Glucose'),(find_col('Hypoxanthine') or 'Hypoxanthine','Uric A (mg/dL)','Hypoxanthine vs. Uric acid','Hypoxanthine','Uric acid (mg/dL)'),('Ferritina (mg/dL)','IST (%)','Ferritin vs. Transferrin saturation','Ferritin (ng/mL)','Transferrin saturation (%)')]
for j,(xcol,ycol,title,xlab,ylab) in enumerate(scatter_pairs):
    ax=fig.add_subplot(sub[j//2,j%2]); panel_letter(ax,chr(71+j))
    x=pd.to_numeric(df[xcol],errors='coerce').copy(); yv=pd.to_numeric(df[ycol],errors='coerce').copy()
    if xcol in metab_cols+protein_cols: x=np.log10(x.mask(x==0).clip(lower=1e-12)); xlab=xlab+' (log10)'
    if ycol in metab_cols+protein_cols: yv=np.log10(yv.mask(yv==0).clip(lower=1e-12)); ylab=ylab+' (log10)'
    mask=np.isfinite(x)&np.isfinite(yv)
    for g in order:
        m=mask&(df.Group_clean==g)
        if m.sum(): ax.scatter(x[m],yv[m],s=15,c=colors[g],edgecolor='white',lw=.3,alpha=.85)
    if mask.sum()>4:
        rho,p=spearmanr(x[mask],yv[mask]); coef=np.polyfit(x[mask],yv[mask],1); xs=np.linspace(x[mask].min(),x[mask].max(),100); ax.plot(xs,coef[0]*xs+coef[1],color='0.25',lw=1.1,alpha=.8)
        ptxt=f'{p:.1e}' if p<.001 else f'{p:.3f}'
        ax.text(.05,.90,f'Spearman ρ = {rho:.2f}\nP = {ptxt}',transform=ax.transAxes,fontsize=5.8,va='top')
    ax.set_title(title,fontsize=7.2,weight='bold',pad=8); ax.set_xlabel(xlab,fontsize=6.2,labelpad=2); ax.set_ylabel(ylab,fontsize=6.2,labelpad=2); ax.tick_params(labelsize=5.6)
    for s in ['top','right']: ax.spines[s].set_visible(False)
fig.savefig(f'{OUT}/Supplementary_Figure_4_refined.svg',format='svg',bbox_inches='tight')
plt.close(fig)

# Zip refined outputs
zip_path=f'{OUT}/refined_supplementary_figures_svg.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for fn in ['Supplementary_Figure_1_refined.svg','Supplementary_Figure_2_refined.svg','Supplementary_Figure_3_refined.svg','Supplementary_Figure_4_refined.svg','refine_supp_figs.py']:
        p=f'{OUT}/{fn}'
        if os.path.exists(p): z.write(p,arcname=fn)
print('DONE',zip_path)
