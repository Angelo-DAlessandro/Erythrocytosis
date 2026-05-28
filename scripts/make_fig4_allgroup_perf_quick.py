import os, warnings, zipfile, math
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
OUT=os.environ.get('FIG_OUTDIR', '/mnt/data'); os.makedirs(OUT, exist_ok=True); CSV=os.environ.get('FIG_DATA', os.path.join(OUT,'Merged_by_Sample_ID_and_Group(4).csv'))
df=pd.read_csv(CSV)
labelmap={'01-Ctrl':'Control','02-Family':'Family','03-VHL':'VHL','03-EGLN2':'EGLN2','04-EPAS1':'EPAS1','05-EPOR':'EPOR','06-HAH':'HAH','07-NDD':'NDD','08-NDD/PDE4':'NDD/PDE4'}
df['Group_clean']=df['Group'].map(labelmap).fillna(df['Group'])
order=['Control','Family','VHL','EGLN2','EPAS1','EPOR','HAH','NDD','NDD/PDE4']
colors={'Control':'#6b6b6b','Family':'#e39b27','VHL':'#7646a2','EGLN2':'#3e82c4','EPAS1':'#18aaa7','EPOR':'#40b86a','HAH':'#1e9c7c','NDD':'#9b6a54','NDD/PDE4':'#cbd600','Unknown-like':'#b5b5b5'}
clinical_cols=[c for c in list(df.columns[2:26]) if pd.api.types.is_numeric_dtype(df[c])]
protein_cols=[c for c in list(df.columns[26:1488]) if pd.api.types.is_numeric_dtype(df[c]) and (pd.to_numeric(df[c], errors='coerce')>0).any()]
metab_cols=[c for c in list(df.columns[1488:-1]) if pd.api.types.is_numeric_dtype(df[c]) and (pd.to_numeric(df[c], errors='coerce')>0).any()]
def prep_matrix(cols, log=True, zero_missing=True):
    X=df[cols].apply(pd.to_numeric, errors='coerce').copy()
    if zero_missing: X=X.mask(X==0)
    if log: X=np.log10(X.clip(lower=1e-12))
    X=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=cols, index=df.index)
    X=pd.DataFrame(StandardScaler().fit_transform(X), columns=cols, index=df.index)
    return X
Xclin=prep_matrix(clinical_cols, log=False, zero_missing=False)
Xprot=prep_matrix(protein_cols, log=True, zero_missing=True)
Xmet=prep_matrix(metab_cols, log=True, zero_missing=True)
Xall=pd.concat([Xclin,Xprot,Xmet],axis=1)
def panel_letter(ax, l, x=-.12, y=1.07): ax.text(x,y,l,transform=ax.transAxes,fontsize=17,weight='bold',va='top',clip_on=False)
def clean(s,n=28):
    ss=str(s).replace('_',' ').replace(' (','\n(')
    return ss if len(ss)<=n else ss[:n-1]+'…'
def find_col(name):
    if name in df.columns: return name
    for c in df.columns:
        if c.lower()==str(name).lower(): return c
    for c in df.columns:
        if str(name).lower() in c.lower(): return c
    return None
def missing_df(cols, zero_missing=False):
    X=df[cols].apply(pd.to_numeric, errors='coerce').copy()
    return X.mask(X==0) if zero_missing else X
def curved_edge(ax, p1, p2, color, lw, alpha=.45, rad=.18):
    ax.add_patch(FancyArrowPatch(p1,p2,connectionstyle=f'arc3,rad={rad}',arrowstyle='-',lw=lw,color=color,alpha=alpha,mutation_scale=1,zorder=1))
# classifier labels include all visual rows explicitly
df['Class_for_LOOCV']=df['Group_clean'].where(df['Group_clean'].isin(['VHL','EGLN2','EPAS1','EPOR','HAH']),'Unknown-like')
mask_all=df['Class_for_LOOCV'].isin(['VHL','EGLN2','EPAS1','EPOR','HAH','Unknown-like'])
class_order=['VHL','EGLN2','EPAS1','EPOR','HAH','Unknown-like']
def loocv_cm(ax,X,title,kmax):
    Xk=X.loc[mask_all]; y=df.loc[mask_all,'Class_for_LOOCV']
    yy=pd.Categorical(y,categories=class_order,ordered=True).codes
    k=min(kmax, Xk.shape[1])
    pipe=make_pipeline(SimpleImputer(strategy='median'), SelectKBest(f_classif,k=k), StandardScaler(), KNeighborsClassifier(n_neighbors=1,metric='euclidean'))
    pred=cross_val_predict(pipe,Xk,yy,cv=LeaveOneOut())
    cm=confusion_matrix(yy,pred,labels=range(len(class_order)))
    row=np.divide(cm, cm.sum(axis=1,keepdims=True), out=np.zeros_like(cm,dtype=float), where=cm.sum(axis=1,keepdims=True)!=0)*100
    acc=accuracy_score(yy,pred); bacc=balanced_accuracy_score(yy,pred)
    im=ax.imshow(row,cmap='Blues',vmin=0,vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt=f'{cm[i,j]}\n{row[i,j]:.0f}%' if cm[i,j] else '·'
            ax.text(j,i,txt,ha='center',va='center',fontsize=5.7,color='white' if row[i,j]>55 else '0.23')
    ax.set_xticks(range(len(class_order))); ax.set_yticks(range(len(class_order)))
    ax.set_xticklabels([f'{x}\n(n={sum(y==x)})' for x in class_order],fontsize=5.2,rotation=35,ha='right')
    ax.set_yticklabels([f'{x}\n(n={sum(y==x)})' for x in class_order],fontsize=5.5)
    ax.set_xlabel('Predicted label',fontsize=6.4,labelpad=2); ax.set_ylabel('True label',fontsize=6.4,labelpad=2)
    ax.set_title(f'{title}\nAccuracy {acc:.2f}; balanced accuracy {bacc:.2f}',fontsize=7.2,weight='bold',pad=6)
    perf='\n'.join([f'{cl}: {row[ii,ii]:.0f}%' for ii,cl in enumerate(class_order)])
    ax.text(1.08,.50,'Recall\n'+perf, transform=ax.transAxes, ha='left', va='center',fontsize=5.7, clip_on=False,
            bbox=dict(boxstyle='round,pad=.20',fc='white',ec='0.82',lw=.5,alpha=.96))
    return im
fig=plt.figure(figsize=(18.8,11.6)); gs=GridSpec(2,4,figure=fig,wspace=.72,hspace=.60,height_ratios=[1.05,1.15])
fig.suptitle('Supplementary Figure 4. Classifier robustness and expanded clinical-omics validation.',fontsize=16,weight='bold',y=.985)
for i,(X,title,k) in enumerate([(Xclin,'LOOCV confusion matrix\nClinical-only classifier',20),(Xmet,'LOOCV confusion matrix\nMetabolome-only classifier',40),(Xprot,'LOOCV confusion matrix\nProteome-only classifier',60),(Xall,'LOOCV confusion matrix\nCombined-data classifier',80)]):
    ax=fig.add_subplot(gs[0,i]); panel_letter(ax,chr(65+i),x=-.18,y=1.12); im=loocv_cm(ax,X,title,k)
cax=fig.add_axes([0.946,0.61,0.010,0.24]); cb=fig.colorbar(im,cax=cax); cb.set_label('% of row',fontsize=7); cb.ax.tick_params(labelsize=6)
# E: feature importance in defined groups only
ax=fig.add_subplot(gs[1,0]); panel_letter(ax,'E')
defined_mask=df.Group_clean.isin(['VHL','EGLN2','EPAS1','EPOR','HAH']); y=df.loc[defined_mask,'Group_clean']; Xsel=Xall.loc[defined_mask]
rf=RandomForestClassifier(n_estimators=350,random_state=4,class_weight='balanced',max_features='sqrt')
rf.fit(Xsel,y); imp=pd.Series(rf.feature_importances_,index=Xsel.columns).sort_values(ascending=False).head(15)[::-1]
barcols=['#b13a45' if f in clinical_cols else '#2b9a8a' if f in metab_cols else '#245da3' for f in imp.index]
ax.barh(range(len(imp)),imp.values,color=barcols,edgecolor='white',lw=.3)
ax.set_yticks(range(len(imp))); ax.set_yticklabels([clean(x,20) for x in imp.index],fontsize=6)
ax.set_xlabel('Random forest importance',fontsize=7); ax.set_title('Top 15 predictors (combined model)',fontsize=8,weight='bold'); ax.tick_params(axis='x',labelsize=6)
for s in ['top','right']: ax.spines[s].set_visible(False)
# F correlation network polished
ax=fig.add_subplot(gs[1,1:3]); panel_letter(ax,'F'); ax.axis('off')
clin_sel=[c for c in ['Ferritina (mg/dL)','IST (%)','Transferrin (mg/dL)','Uric A (mg/dL)','Glucose (mg/dL)','vLac (mmol/L)','MPV (fL)','PLT (/microL)','Age'] if c in df.columns]
met_sel=[]; prot_sel=[]
for n in ['Hypoxanthine','L-alanine','N-Acetylneuraminate','2-Oxoglutarate','Ornithine','Putrescine','Adenosine','Glucose','Lactate','Orotate','Nicotinamide','Kynurenine']:
    c=find_col(n)
    if c and c in metab_cols and c not in met_sel: met_sel.append(c)
for n in ['NDUFS1','PSMD12','EIF4A1','HINT1','OLA1','GYPC','RAB35','S100A8','NME1','EEF1A1','HPRT1']:
    c=find_col(n)
    if c and c in protein_cols and c not in prot_sel: prot_sel.append(c)
for f in imp.index:
    if f in metab_cols and f not in met_sel and len(met_sel)<11: met_sel.append(f)
    if f in protein_cols and f not in prot_sel and len(prot_sel)<11: prot_sel.append(f)
raw=pd.concat([missing_df(clin_sel,False),missing_df(met_sel,True),missing_df(prot_sel,True)],axis=1)
for c in met_sel+prot_sel: raw[c]=np.log10(raw[c].clip(lower=1e-12))
raw=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(raw),columns=raw.columns)
edges=[]
for a in clin_sel:
    for b in met_sel+prot_sel:
        rho,p=spearmanr(raw[a],raw[b])
        if np.isfinite(rho) and abs(rho)>=.42: edges.append((a,b,rho))
for a in met_sel:
    for b in prot_sel:
        rho,p=spearmanr(raw[a],raw[b])
        if np.isfinite(rho) and abs(rho)>=.50: edges.append((a,b,rho))
edges=sorted(edges,key=lambda x:abs(x[2]),reverse=True)[:70]
nodes=set([x for e in edges for x in e[:2]]); clin_n=[c for c in clin_sel if c in nodes]; met_n=[c for c in met_sel if c in nodes]; prot_n=[c for c in prot_sel if c in nodes]
pos={}
for i,c in enumerate(clin_n): pos[c]=(.07,.88-i*(.76/max(1,len(clin_n)-1)))
for i,c in enumerate(met_n): pos[c]=(.93,.88-i*(.76/max(1,len(met_n)-1)))
for i,c in enumerate(prot_n): pos[c]=(.50,.08+i*(.34/max(1,len(prot_n)-1)))
for a,b,rho in edges:
    if a in pos and b in pos:
        curved_edge(ax,pos[a],pos[b],plt.cm.RdBu_r((rho+1)/2),.5+2.5*abs(rho),alpha=.45,rad=.20 if pos[a][0]<pos[b][0] else -.20)
for ns,fc in [(clin_n,'#b13a45'),(met_n,'#2b9a8a'),(prot_n,'#245da3')]:
    for c in ns:
        x0,y0=pos[c]; ax.scatter(x0,y0,s=95,c=fc,edgecolor='white',lw=.7,zorder=3)
        ha='right' if x0<.2 else 'left' if x0>.8 else 'center'; dx=-.025 if x0<.2 else .025 if x0>.8 else 0; dy=0 if x0!=.5 else -.035
        ax.text(x0+dx,y0+dy,clean(c,23),ha=ha,va='center' if x0!=.5 else 'top',fontsize=6.3,zorder=4)
ax.text(.07,.98,'Clinical',color='#b13a45',ha='center',fontsize=9,weight='bold'); ax.text(.93,.98,'Metabolites',color='#2b9a8a',ha='center',fontsize=9,weight='bold'); ax.text(.50,.02,'Proteins',color='#245da3',ha='center',fontsize=9,weight='bold')
ax.set_title('Expanded clinical–metabolite–protein correlation network (Spearman ρ)',fontsize=8.5,weight='bold',pad=2)
# G-J with two-line titles
sub=gs[1,3].subgridspec(2,2,wspace=.70,hspace=.92)
scatter_pairs=[('Emogas pHv','vLac (mmol/L)','Blood pH vs.\nLactate','blood pH','Lactate (mmol/L)'),('Glucose (mg/dL)',find_col('D-Glucose') or 'D-Glucose','Serum glucose vs.\nRBC glucose','Serum glucose (mg/dL)','D-Glucose'),(find_col('Hypoxanthine') or 'Hypoxanthine','Uric A (mg/dL)','Hypoxanthine vs.\nUric acid','Hypoxanthine','Uric acid (mg/dL)'),('Ferritina (mg/dL)','IST (%)','Ferritin vs.\nTransferrin saturation','Ferritin (ng/mL)','Transferrin saturation (%)')]
for j,(xcol,ycol,title,xlab,ylab) in enumerate(scatter_pairs):
    ax=fig.add_subplot(sub[j//2,j%2]); panel_letter(ax,chr(71+j),x=-.24,y=1.23)
    x=pd.to_numeric(df[xcol],errors='coerce').copy(); yv=pd.to_numeric(df[ycol],errors='coerce').copy()
    if xcol in metab_cols+protein_cols: x=np.log10(x.mask(x==0).clip(lower=1e-12)); xlab=xlab+' (log10)'
    if ycol in metab_cols+protein_cols: yv=np.log10(yv.mask(yv==0).clip(lower=1e-12)); ylab=ylab+' (log10)'
    mask=np.isfinite(x)&np.isfinite(yv)
    for g in order:
        m=mask&(df.Group_clean==g)
        if m.sum(): ax.scatter(x[m],yv[m],s=15,c=colors[g],edgecolor='white',lw=.3,alpha=.85)
    if mask.sum()>4:
        rho,p=spearmanr(x[mask],yv[mask]); coef=np.polyfit(x[mask],yv[mask],1); xs=np.linspace(x[mask].min(),x[mask].max(),100); ax.plot(xs,coef[0]*xs+coef[1],color='0.25',lw=1.1,alpha=.8)
        ptxt=f'{p:.1e}' if p<.001 else f'{p:.3f}'; ax.text(.05,.90,f'Spearman ρ = {rho:.2f}\nP = {ptxt}',transform=ax.transAxes,fontsize=5.8,va='top')
    ax.set_title(title,fontsize=7.5,weight='bold',pad=12); ax.set_xlabel(xlab,fontsize=6.2,labelpad=2); ax.set_ylabel(ylab,fontsize=6.2,labelpad=2); ax.tick_params(labelsize=5.6)
    for s in ['top','right']: ax.spines[s].set_visible(False)
out=f'{OUT}/Supplementary_Figure_4_final_all_group_performance.svg'
fig.savefig(out,format='svg',bbox_inches='tight'); plt.close(fig)
zip_path=f'{OUT}/Supplementary_Figure_4_final_all_group_performance.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    z.write(out,arcname=os.path.basename(out)); z.write(__file__,arcname='make_fig4_allgroup_perf_quick.py')
print(out); print(zip_path)
