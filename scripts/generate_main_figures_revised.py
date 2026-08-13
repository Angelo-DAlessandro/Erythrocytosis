import os, re, zipfile, warnings, math
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['svg.fonttype']='none'
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['font.sans-serif']=['Arial','Nimbus Sans','Liberation Sans','DejaVu Sans']
mpl.rcParams['font.size']=6.4
mpl.rcParams['axes.linewidth']=0.65
mpl.rcParams['xtick.major.width']=0.55
mpl.rcParams['ytick.major.width']=0.55
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, PathPatch
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.manifold import MDS

OUT=os.environ.get('FIG_OUTDIR', '/mnt/data')
os.makedirs(OUT, exist_ok=True)
DATA=os.environ.get('FIG_DATA', os.path.join(OUT,'Merged_by_Sample_ID_and_Group(4).csv'))
META=os.environ.get('FIG_META', os.path.join(os.path.dirname(DATA),'analysis_metadata.csv'))
ANALYSIS=os.environ.get('FIG_ANALYSIS_RESULTS', os.path.join(os.path.dirname(DATA),'analysis_results'))
df=pd.read_csv(DATA)
meta=pd.read_csv(META).set_index('Sample_ID')
clinical_cols=list(df.columns[2:26])
clinical_cols=[c for c in clinical_cols if c not in {'Gender','Age'}]
protein_cols=list(df.columns[26:1488])
met_cols=list(df.columns[1488:])
order=['01-Ctrl','02-Family','03-VHL','03-EGLN1','04-EPAS1','05-EPOR','06-HAH','07-NDD','08-PDE4-associated']
labels_short={'01-Ctrl':'Ctrl','02-Family':'Family','03-VHL':'VHL','03-EGLN1':'EGLN1','04-EPAS1':'EPAS1','05-EPOR':'EPOR','06-HAH':'HAH','07-NDD':'NDD','08-PDE4-associated':'PDE4 family'}
palette={'01-Ctrl':'#6e7f80','02-Family':'#F2A93B','03-VHL':'#6A3D9A','03-EGLN1':'#4C78A8','04-EPAS1':'#1F9EB3','05-EPOR':'#2DBA61','06-HAH':'#29A889','07-NDD':'#9C755F','08-PDE4-associated':'#C8D800'}
def_groups=['03-VHL','03-EGLN1','04-EPAS1','05-EPOR','06-HAH']
def_labels=[labels_short[g] for g in def_groups]
pooled_palette={'Control':'#6e7f80','Family':'#F2A93B','Erythrocytosis':'#D62728'}
# remove 2 multivariate outliers as before
out_ids={'F2_1','F6_1'}
use_idx=[i for i in df.index if str(df.loc[i,'Sample_ID']) not in out_ids]
D=df.loc[use_idx].copy().reset_index(drop=False).rename(columns={'index':'orig_index'})
D['Participant_ID']=D['Sample_ID'].map(meta['Participant_ID'])
rows=D['orig_index'].values
groups=D['Group'].values

def prep(cols, omics=False, rows=None, min_det=0.35, log=True):
    x=df.loc[rows if rows is not None else df.index, cols].copy().apply(pd.to_numeric, errors='coerce')
    if omics:
        x=x.mask(x==0)  # zero is missing for metabolomics/proteomics
        x=x.loc[:, x.notna().mean(axis=0)>=min_det]
        if log:
            for c in x.columns:
                vals=x[c].dropna()
                if len(vals)>2 and vals.min()>0:
                    x[c]=np.log2(x[c])
    x=x.replace([np.inf,-np.inf], np.nan)
    x=x.fillna(x.median(numeric_only=True))
    x=x.loc[:, x.std(axis=0)>1e-9]
    return pd.DataFrame(StandardScaler().fit_transform(x), index=(rows if rows is not None else df.index), columns=x.columns)

Xclin=prep(clinical_cols, False, rows=rows)
Xprot=prep(protein_cols, True, rows=rows)
Xmet=prep(met_cols, True, rows=rows)
Xall=pd.concat([Xclin, Xprot, Xmet], axis=1)

# ---------- helpers ----------
def panel_label(ax, s, x=-0.10, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontweight='bold', fontsize=10.5, ha='left', va='top')

def sig_star(p):
    if p < 1e-4: return '****'
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 0.05: return '*'
    return ''

def add_ellipse(ax, x, y, color, alpha=0.14):
    if len(x)<3: return
    cov=np.cov(x,y)
    if not np.isfinite(cov).all(): return
    vals, vecs=np.linalg.eigh(cov); vals=np.maximum(vals,1e-9)
    idx=np.argsort(vals)[::-1]; vals=vals[idx]; vecs=vecs[:,idx]
    ang=np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    w,h=2*np.sqrt(vals*5.991)
    ax.add_patch(Ellipse((np.mean(x),np.mean(y)), w,h, angle=ang, facecolor=color, edgecolor=color, lw=1.0, alpha=alpha, zorder=1))

def pls_scores(X, y):
    Y=pd.get_dummies(pd.Series(y)).values
    pls=PLSRegression(n_components=2)
    return pls.fit_transform(X.values, Y)[0]

def nice_feature_name(s, is_met=False):
    x=str(s).strip()
    clinical_map={
        'Emogas pHv':'pH','vLac (mmol/L)':'Lactate','vpO2 (mmHg)':'pO2','vpC02 (mmHg)':'pCO2',
        'Ferritin (ng/mL)':'Ferritin (ng/mL)','Transferrin (mg/dL)':'Transferrin','Uric A (mg/dL)':'Uric acid',
        'TOT Bil (mg/dL)':'Total bilirubin','IND BIL (mg/dL)':'Indirect bilirubin','Glucose (mg/dL)':'Glucose',
        'PLT (/microL)':'Platelets','Neu (/microL)':'Neutrophils','WBC (/microL)':'WBC','MPV (fL)':'MPV',
        'COL TOT (mg/dL)':'Total cholesterol','HDL':'HDL','LDL':'LDL','TG':'Triglycerides','IST (%)':'IST',
        'ALT (U/L)':'ALT','AST (U/L)':'AST','eGFR (ml/min)':'eGFR'
    }
    if x in clinical_map: return clinical_map[x]
    if is_met:
        low=x.lower().replace('_',' ').replace('-labeled','')
        reps={
            'l-alanine':'Alanine','alanine':'Alanine','l-arginine':'Arginine','arginine':'Arginine','l-aspartate':'Aspartate','aspartate':'Aspartate','l-glutamate':'Glutamate','glutamate':'Glutamate',
            'l-glutamine':'Glutamine','glutamine':'Glutamine','glycine':'Glycine','l-histidine':'Histidine','histidine':'Histidine','l-leucine':'Leucine','leucine':'Leucine','l-lysine':'Lysine','lysine':'Lysine',
            'l-methionine':'Methionine','methionine':'Methionine','l-serine':'Serine','serine':'Serine','l-tryptophan':'Tryptophan','tryptophan':'Tryptophan','l-valine':'Valine','valine':'Valine',
            'alpha-ketoglutarate':'α-Ketoglutarate','2-hydroxyglutarate':'2-Hydroxyglutarate','beta-hydroxybutyrate':'β-Hydroxybutyrate','pyroglutamate':'Pyroglutamate',
            'gamma-glutamyl-secys':'γ-SeCys','gamma-glutamyl-se-cys':'γ-SeCys','gamma-glutamyl-se-methylselenocysteine':'γ-SeCys','gamma-glutamylselenocysteine':'γ-SeCys',
            'octadecanoic acid':'FA 18:0','hexadecanoic acid':'FA 16:0','octadecenoic acid':'FA 18:1','octadecadienoic acid':'FA 18:2','octadecatrienoic acid':'FA 18:3',
            'eicosatetraenoic acid':'FA 20:4','docosahexaenoic acid':'FA 22:6','icosatrienoic acid':'FA 20:3','(8z-11z-14z)-icosatrienoic acid':'FA 20:3',
            'linoleate':'Linoleate','lactate':'Lactate','succinate':'Succinate','xanthine':'Xanthine','hypoxanthine':'Hypoxanthine','inosine':'Inosine','adenosine':'Adenosine',
            'spermine':'Spermine','spermidine':'Spermidine','creatine':'Creatine','citrulline':'Citrulline','kynurenine':'Kynurenine','nicotinamide':'Nicotinamide','niacinamide':'Niacinamide',
            'glutathione':'Glutathione','glutathione disulfide':'GSSG','biliverdin':'Biliverdin','pyruvate':'Pyruvate','citrate':'Citrate','malate':'Malate','fumarate':'Fumarate',
            'folate':'Folate','orotate':'Orotate','ribose':'Ribose','fructose':'Fructose','glucose':'Glucose','urea':'Urea','myo-inositol':'Myo-inositol',
            'erythrose 4-phosphate':'Erythrose-4P','10-formyldihydrofolate':'10-Formyl-THF','5,6-dihydrouracil':'Dihydrouracil','cystathionine':'Cystathionine'
        }
        for k,v in reps.items():
            if low==k or low.startswith(k): return v
        m=re.search(r'acyl[- ]?c(\d+)(?::(\d+))?[- ]?(oh)?', low, flags=re.I)
        if m:
            return 'Acyl-C'+m.group(1)+((':'+m.group(2)) if m.group(2) else '')+(('-OH') if m.group(3) else '')
        x=re.sub(r'^[LD]-','',x)
        x=x[:1].upper()+x[1:]
        return x if len(x)<=18 else x[:17]+'…'
    return x if len(x)<=24 else x[:22]+'…'

def top_anova(X, y, n=20):
    F,p=f_classif(X.values, np.array(y))
    res=pd.DataFrame({'feature':X.columns,'p':p,'F':F}).replace([np.inf,-np.inf],np.nan).dropna()
    return res.sort_values(['p','F'],ascending=[True,False]).head(n)

def heatmap_square(ax, X, features, y, title, is_met=False, panel=''):
    feats=list(dict.fromkeys(features))
    data=X[feats].copy()
    # sort columns by group order and within-group PC1
    sort=[]
    for g in order:
        inds=np.where(np.array(y)==g)[0]
        if len(inds)==0: continue
        if len(inds)>1:
            pc1=PCA(n_components=1, random_state=1).fit_transform(data.iloc[inds].values).ravel()
        else: pc1=np.array([0])
        for i,v in zip(inds, pc1): sort.append((order.index(g), v, i))
    idx=[i for _,__,i in sorted(sort)]
    vals=data.iloc[idx].T.values
    feats_order=feats
    if vals.shape[0]>2:
        try:
            leaves=leaves_list(linkage(vals, method='average', metric='correlation'))
            vals=vals[leaves]; feats_order=[feats[i] for i in leaves]
        except Exception:
            pass
    im=ax.imshow(vals, aspect='equal', cmap='RdBu_r', vmin=-2.5, vmax=2.5, interpolation='nearest')
    nrow,ncol=vals.shape
    ax.set_xlim(-0.5, ncol+2.3); ax.set_ylim(nrow-0.5, -2.35)
    ax.set_yticks(np.arange(nrow)); ax.set_yticklabels([nice_feature_name(f,is_met) for f in feats_order], fontsize=2.65)
    ax.set_xticks([])
    ax.set_title(title, fontweight='bold', fontsize=7.4, pad=16)
    # group top bars
    ybar=-1.35; ytext=-1.62
    yarr=np.array(y)[idx]; start=0
    for g in order:
        n=(yarr==g).sum()
        if n:
            ax.add_patch(Rectangle((start-0.5,ybar),n,0.34,color=palette[g],clip_on=False,lw=0))
            ax.text(start+n/2-0.5,ytext,f'{labels_short[g]}\n(n={n})',ha='center',va='bottom',rotation=0,fontsize=3.15,clip_on=False)
            start+=n
    # right-side p-value stars
    for r,f in enumerate(feats_order):
        arrays=[X.loc[np.array(y)==g, f].values for g in order if (np.array(y)==g).sum()>1]
        try: p=stats.kruskal(*arrays).pvalue
        except Exception: p=np.nan
        st=sig_star(p) if np.isfinite(p) else ''
        if st: ax.text(ncol+0.20, r, st, va='center', ha='left', fontsize=3.7, fontweight='bold')
    if panel:
        ax.text(-0.13, 1.09, panel, transform=ax.transAxes, fontweight='bold', fontsize=10.5, ha='left', va='top')
    for sp in ax.spines.values(): sp.set_linewidth(0.6); sp.set_color('0.4')
    return im

# ---------------- FIGURE 1 ----------------
fig=plt.figure(figsize=(8,8),facecolor='white')
gs=GridSpec(2,3,figure=fig,height_ratios=[0.90,1.28],width_ratios=[1.05,1,1.05],
            left=0.055,right=0.965,top=0.965,bottom=0.285,hspace=0.29,wspace=0.34)
# A
axA=fig.add_subplot(gs[0,0]); axA.axis('off'); panel_label(axA,'A',x=-0.17,y=1.02)
axA.set_title('Cohort and data layers',fontweight='bold',fontsize=7.8,pad=4)
counts_final=D['Group'].value_counts()
for label,n,col,y0 in [('Control',7,'#6e7f80',0.72),('Family',4,'#F2A93B',0.50),('Familial erythrocytosis',52,'#D62728',0.28)]:
    axA.add_patch(FancyBboxPatch((0.03,y0),0.43,0.13,boxstyle='round,pad=0.015,rounding_size=0.02',fc=col,ec='white',lw=1))
    axA.text(0.245,y0+0.065,f'{label}\n(n={n})',color='white',ha='center',va='center',fontsize=6.5,fontweight='bold')
axA.add_patch(FancyArrowPatch((0.48,0.345),(0.58,0.49),arrowstyle='-|>',mutation_scale=16,lw=1.4,color='0.35'))
subgroups=['03-VHL','03-EGLN1','04-EPAS1','05-EPOR','06-HAH','07-NDD','08-PDE4-associated']
coords=[(0.60,0.76),(0.78,0.76),(0.60,0.58),(0.78,0.58),(0.60,0.40),(0.78,0.40),(0.69,0.22)]
for g,(x,y0) in zip(subgroups,coords):
    n=int(counts_final.get(g,0))
    axA.add_patch(FancyBboxPatch((x,y0),0.16,0.11,boxstyle='round,pad=0.012,rounding_size=0.018',fc=palette[g],ec='white',lw=0.8))
    participant_n=int(D.loc[D['Group']==g,'Participant_ID'].nunique())
    count_text=f'n={n}' if participant_n==n else f'n={n}; N={participant_n}'
    axA.text(x+0.08,y0+0.055,f'{labels_short[g]}\n({count_text})',color='white' if g!='08-PDE4-associated' else 'black',ha='center',va='center',fontsize=4.55 if participant_n!=n else 5.25,fontweight='bold')
for i,(lab,col) in enumerate([('Clinical','#9ecae1'),('Proteome','#fdae6b'),('Metabolome','#a1d99b')]):
    xx=0.03+i*0.22
    axA.add_patch(FancyBboxPatch((xx,0.04),0.17,0.075,boxstyle='round,pad=0.008,rounding_size=0.015',fc=col,ec='0.35',lw=0.6))
    axA.text(xx+0.085,0.078,lab,ha='center',va='center',fontsize=5.25,fontweight='bold')
for cx,cy,r in [(0.51,0.28,0.032),(0.54,0.32,0.022),(0.48,0.34,0.019)]:
    axA.add_patch(Circle((cx,cy),r,fc='#c0392b',ec='#7f1d1d',lw=0.6,alpha=0.9))
    axA.add_patch(Circle((cx,cy),r*0.45,fc='#e9967a',ec='none',alpha=0.65))
axA.text(0.03,-0.018,f'Final analysis cohort: {len(D)} samples / {D.Participant_ID.nunique()} participants\nafter excluding 2 multivariate Family-group outliers',fontsize=4.55,ha='left',va='top',style='italic')
axA.set_xlim(0,1); axA.set_ylim(-0.08,1.02)
# B
axB=fig.add_subplot(gs[0,1]); panel_label(axB,'B',x=-0.12,y=1.02)
pooled=np.where(groups=='01-Ctrl','Control',np.where(groups=='02-Family','Family','Erythrocytosis'))
coords=pls_scores(Xall,pooled)
for g in ['Control','Family','Erythrocytosis']:
    m=pooled==g; col=pooled_palette[g]
    axB.scatter(coords[m,0],coords[m,1],s=18,color=col,edgecolor='white',lw=0.35,alpha=0.95,zorder=2)
    add_ellipse(axB,coords[m,0],coords[m,1],col,alpha=0.14)
    axB.text(coords[m,0].mean(),coords[m,1].mean(),f'{g}\n(n={m.sum()})',ha='center',va='center',fontsize=4.8,fontweight='bold',bbox=dict(fc='white',ec=col,lw=0.65,alpha=0.9,boxstyle='round,pad=0.14'))
axB.axhline(0,color='0.85',lw=0.7); axB.axvline(0,color='0.85',lw=0.7)
axB.set_xlabel('Latent variable 1'); axB.set_ylabel('Latent variable 2')
axB.set_title('Control vs Family\nvs erythrocytosis',fontweight='bold',fontsize=7.2,pad=4)
axB.tick_params(labelsize=5.5)
# C
axC=fig.add_subplot(gs[0,2]); panel_label(axC,'C',x=-0.10,y=1.02)
def_mask=np.isin(groups,def_groups)
coords2=pls_scores(Xall.loc[def_mask],groups[def_mask])
for g in def_groups:
    m=groups[def_mask]==g; col=palette[g]
    axC.scatter(coords2[m,0],coords2[m,1],s=18,color=col,edgecolor='white',lw=0.35,alpha=0.95,zorder=2)
    add_ellipse(axC,coords2[m,0],coords2[m,1],col,alpha=0.14)
    axC.text(coords2[m,0].mean(),coords2[m,1].mean(),f'{labels_short[g]}\n(n={m.sum()})',ha='center',va='center',fontsize=4.8,fontweight='bold',bbox=dict(fc='white',ec=col,lw=0.65,alpha=0.9,boxstyle='round,pad=0.14'))
axC.axhline(0,color='0.85',lw=0.7); axC.axvline(0,color='0.85',lw=0.7)
axC.set_xlabel('Latent variable 1'); axC.set_ylabel('Latent variable 2')
axC.set_title('Defined erythrocytosis subgroups',fontweight='bold',fontsize=7.2,pad=4)
axC.tick_params(labelsize=5.5)
# Heatmaps top 40/50 (clinical only has fewer vars)
axD=fig.add_subplot(gs[1,0])
clin_feats=list(top_anova(Xclin,groups,min(30,Xclin.shape[1]))['feature'])
im=heatmap_square(axD,Xclin,clin_feats,groups,'Clinical discriminants',False,'D'); axD.set_anchor('N')
axE=fig.add_subplot(gs[1,1])
met_feats=list(top_anova(Xmet,groups,50)['feature'])
heatmap_square(axE,Xmet,met_feats,groups,'Metabolite discriminants',True,'E'); axE.set_anchor('N')
axF=fig.add_subplot(gs[1,2])
prot_feats=list(top_anova(Xprot,groups,50)['feature'])
heatmap_square(axF,Xprot,prot_feats,groups,'Protein discriminants',False,'F'); axF.set_anchor('N')
cax=fig.add_axes([0.238,0.686,0.006,0.072]); cb=fig.colorbar(im,cax=cax); cb.set_label('Autoscaled\nabundance',fontsize=4.2,labelpad=2); cb.ax.tick_params(labelsize=4,length=2)
legax=fig.add_axes([0.252,0.684,0.074,0.076]); legax.axis('off')
legax.add_patch(Rectangle((0,0),1,1,fc='white',ec='0.65',lw=0.7))
legax.text(0.5,0.84,'Significance\n(p-value)',ha='center',va='top',fontsize=4.1,fontweight='bold')
for i,(st,txt) in enumerate([('*','p < 0.05'),('**','p < 0.01'),('***','p < 0.001'),('****','p < 0.0001')]):
    legax.text(0.18,0.50-i*0.14,st,ha='right',fontsize=4.0,fontweight='bold'); legax.text(0.25,0.50-i*0.14,txt,ha='left',fontsize=3.9)
fig.savefig(os.path.join(OUT,'Main_Figure_1_final_realdata.svg'),bbox_inches=None,pad_inches=0,facecolor='white')
fig.savefig(os.path.join(OUT,'Main_Figure_1_final_realdata.png'),dpi=300,bbox_inches=None,pad_inches=0,facecolor='white')
plt.close(fig)

# ---------------- FIGURE 2 ----------------
met_modules={
    'Energy / glycolysis / TCA':['lactate','pyruvate','citrate','succinate','fumarate','malate','ketoglutarate','glucose','phosphoglycerate','phosphoenolpyruvate'],
    'Fatty acids / lipid':['acyl','carnitine','linoleate','octadecanoic','hexadecanoic','octadecenoic','eicosa','docosa','lyso','phosphatidyl'],
    'One-carbon / folate':['folate','serine','glycine','methionine','homocysteine','methyl','spermidine','spermine'],
    'Purine / urate':['ATP','ADP','AMP','IMP','inosine','hypoxanthine','xanthine','urate','uric','adenosine','GMP','GDP'],
    'Tryptophan / indole':['tryptophan','kynurenine','indole','serotonin'],
    'Redox / sulfur':['glutathione','cyst','methionine','taurine','biliverdin','nicotinamide','NAD','sulfur','selenocysteine']
}
prot_modules={
    'Translation / chaperone':['RPL','RPS','EEF','EIF','PFDN','HSPA','HSP90','CCT'],
    'Proteostasis / proteasome':['PSMA','PSMB','PSMC','PSMD','UBE','VCP','HSPA','CCT'],
    'Cytoskeleton / membrane':['SPTA','SPTB','ANK1','SLC4A1','GYPA','GYPB','GYPC','EPB','ACTB','ADD','RHAG','RAB','VAMP'],
    'Vesicle trafficking':['RAB','VAMP','STX','SNX','CHMP','AP2','ARF','COP','VPS','CLIC'],
    'Redox / antioxidant':['SOD','PRDX','TXN','GPX','GSR','CAT','GLRX','GST','CYB5R3','BLVRB'],
    'Heme / globin / oxygen':['HBA','HBB','HBD','HBG','AHSP','CA1','CA2','BLVRB','BPGM','ALAS','ALAD']
}
def modscore(X, modules):
    out={}; members={}
    for name,keys in modules.items():
        feats=[c for c in X.columns if any(k.lower() in c.lower() for k in keys)]
        if feats:
            out[name]=X[feats].mean(axis=1); members[name]=feats
    return pd.DataFrame(out,index=X.index),members
Mmet,met_members=modscore(Xmet,met_modules)
Mprot,prot_members=modscore(Xprot,prot_modules)

def module_heat(ax,M,members,title):
    means=[]
    for g in def_groups: means.append(M.loc[groups==g].mean())
    mat=pd.DataFrame(means,index=def_labels).T
    mat=(mat.sub(mat.mean(axis=1),axis=0)).div(mat.std(axis=1).replace(0,1),axis=0)
    im=ax.imshow(mat.values,aspect='equal',cmap='RdBu_r',vmin=-1.5,vmax=1.5)
    ax.set_xticks(range(len(def_groups))); ax.set_xticklabels(def_labels,rotation=35,ha='right',fontsize=4.2)
    yl=[]
    for pth in mat.index:
        try: pval=stats.kruskal(*[M.loc[groups==g,pth].values for g in def_groups]).pvalue
        except Exception: pval=np.nan
        yl.append(f'{pth} (n={len(members.get(pth,[]))}) {sig_star(pval)}')
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(yl,fontsize=2.75)
    ax.set_title(title,fontweight='bold',fontsize=6.6,pad=6)
    for i in range(mat.shape[0]+1): ax.axhline(i-0.5,color='white',lw=0.8)
    for j in range(mat.shape[1]+1): ax.axvline(j-0.5,color='white',lw=0.8)
    return im

fig=plt.figure(figsize=(8,8),facecolor='white')
gs=GridSpec(3,5,figure=fig,height_ratios=[1.05,1.15,0.90],width_ratios=[1.45,0.22,1.0,1.0,1.0],
            left=0.03,right=0.975,top=0.965,bottom=0.285,hspace=0.50,wspace=0.42)
# A/B stacked in col 0 via subgrid
leftgs=GridSpecFromSubplotSpec(2,1,subplot_spec=gs[0:2,0],hspace=0.72)
axA=fig.add_subplot(leftgs[0])
pos=axA.get_position(); axA.set_position([pos.x0+0.04,pos.y0,pos.width-0.04,pos.height])
panel_label(axA,'A',x=-0.32,y=1.12)
imA=module_heat(axA,Mmet,met_members,'Metabolomics\nPathway Analysis')
cb=fig.colorbar(imA,ax=axA,orientation='horizontal',fraction=0.055,pad=0.22); cb.set_label('Median pathway score (z)',fontsize=4.2); cb.ax.tick_params(labelsize=4)
axB=fig.add_subplot(leftgs[1])
pos=axB.get_position(); axB.set_position([pos.x0+0.04,pos.y0,pos.width-0.04,pos.height])
panel_label(axB,'B',x=-0.32,y=1.12)
imB=module_heat(axB,Mprot,prot_members,'Proteomics\nPathways Analysis')
cb=fig.colorbar(imB,ax=axB,orientation='horizontal',fraction=0.055,pad=0.22); cb.set_label('Median pathway score (z)',fontsize=4.2); cb.ax.tick_params(labelsize=4)
# volcanoes C-E occupying cols 2-4, with col1 blank-ish for space? Use cols 2-4.
all_omics=pd.concat([Xmet.add_prefix('Met: '),Xprot.add_prefix('Prot: ')],axis=1)
keep=[]
keep += [f'Met: {c}' for c in top_anova(Xmet.loc[def_mask],groups[def_mask],160)['feature']]
keep += [f'Prot: {c}' for c in top_anova(Xprot.loc[def_mask],groups[def_mask],160)['feature']]
all_omics=all_omics[[c for c in dict.fromkeys(keep) if c in all_omics.columns]]

def repel_label_positions(points, min_dy=0.20):
    # simple same-side vertical spacing
    pts=points.copy().sort_values('mlogp',ascending=False).head(9)
    return pts

def volcano_cov(ax,cov_col,title,letter):
    panel_label(ax,letter,x=-0.18,y=1.10)
    x=Xclin[cov_col].loc[def_mask]
    rowsv=[]
    for feat in all_omics.columns:
        y=all_omics[feat].loc[def_mask]
        r,p=stats.spearmanr(x,y)
        if np.isfinite(r) and np.isfinite(p): rowsv.append((feat,r,p))
    v=pd.DataFrame(rowsv,columns=['feat','rho','p'])
    v['mlogp']=-np.log10(v['p'].clip(lower=1e-300))
    sig=(v['p']<0.05)&(v['rho'].abs()>0.35)
    colors=np.where(~sig,'#bdbdbd',np.where(v['rho']>0,'#D62728','#1F77B4'))
    ax.scatter(v['rho'],v['mlogp'],s=15,c=colors,alpha=0.78,edgecolor='white',lw=0.15)
    ax.axhline(-np.log10(0.05),color='0.55',ls='--',lw=0.7); ax.axvline(0,color='0.75',lw=0.7)
    lab=v[sig].sort_values('p').head(10)
    # place labels with connector lines and alternating offsets
    for k,(_,r) in enumerate(lab.iterrows()):
        name=r['feat'].split(': ',1)[1]; ismet=r['feat'].startswith('Met:')
        labname=nice_feature_name(name,ismet)
        side=1 if r['rho']>0 else -1
        dx=side*(0.12+0.03*(k%3)); dy=0.14+0.10*(k%4)
        ax.annotate(labname,xy=(r['rho'],r['mlogp']),xytext=(r['rho']+dx,r['mlogp']+dy),fontsize=3.35,ha='left' if side>0 else 'right',va='bottom',color='#7f1d1d' if side>0 else '#234b83',arrowprops=dict(arrowstyle='-',lw=0.3,color='0.5',alpha=0.6),clip_on=False)
    ax.set_title(title,fontweight='bold',fontsize=6.8)
    ax.set_xlabel('Spearman rho',fontsize=5); ax.set_ylabel('-log10 p',fontsize=5)
    ax.set_xlim(-1.02,1.02); ax.tick_params(labelsize=5)
    ax.text(0.98,0.92,f'n={sig.sum()} covariates',ha='right',va='top',transform=ax.transAxes,fontsize=4,color='0.35')
for i,(cov,title,letter) in enumerate([('Emogas pHv','pH-linked\ncovariates','C'),('Glucose (mg/dL)','Glucose\ncovariates','D'),('vLac (mmol/L)','Lactate\ncovariates','E')]):
    ax=fig.add_subplot(gs[0,i+2]); volcano_cov(ax,cov,title,letter)
# Panel F network across full row
axF=fig.add_subplot(gs[1,1:5]); panel_label(axF,'F',x=-0.055,y=1.07); axF.axis('off')
axF.set_title('Top clinical-omics correlations',fontweight='bold',fontsize=7.2,pad=2)
clin_list=['IND BIL (mg/dL)','TOT Bil (mg/dL)','ALT (U/L)','Uric A (mg/dL)','Transferrin (mg/dL)','Ferritin (ng/mL)','vpC02 (mmHg)','vLac (mmol/L)','Emogas pHv','TG','LDL','HDL','MPV (fL)','PLT (/microL)','WBC (/microL)','Neu (/microL)','Glucose (mg/dL)']
clin_list=[c for c in clin_list if c in Xclin]
met_nodes=[]; prot_nodes=[]; node_module={}
for mod,feats in met_members.items():
    best=[]
    for f in feats:
        if f in Xmet:
            mx=max([abs(stats.spearmanr(Xclin[c].loc[def_mask],Xmet[f].loc[def_mask])[0]) for c in clin_list if np.isfinite(stats.spearmanr(Xclin[c].loc[def_mask],Xmet[f].loc[def_mask])[0])] or [0])
            best.append((mx,f))
    for _,f in sorted(best,reverse=True)[:2]: met_nodes.append(f); node_module[f]=mod
for mod,feats in prot_members.items():
    best=[]
    for f in feats:
        if f in Xprot:
            mx=max([abs(stats.spearmanr(Xclin[c].loc[def_mask],Xprot[f].loc[def_mask])[0]) for c in clin_list if np.isfinite(stats.spearmanr(Xclin[c].loc[def_mask],Xprot[f].loc[def_mask])[0])] or [0])
            best.append((mx,f))
    for _,f in sorted(best,reverse=True)[:3]: prot_nodes.append(f); node_module[f]=mod
met_nodes=list(dict.fromkeys(met_nodes))[:13]; prot_nodes=list(dict.fromkeys(prot_nodes))[:18]
pos={}
for i,n in enumerate(met_nodes): pos[('met',n)]=(-1.0,0.83-1.66*i/max(1,len(met_nodes)-1))
for i,n in enumerate(clin_list): pos[('clin',n)]=(0.0,0.90-1.80*i/max(1,len(clin_list)-1))
for i,n in enumerate(prot_nodes): pos[('prot',n)]=(1.0,0.83-1.66*i/max(1,len(prot_nodes)-1))
mod_colors={'Energy / glycolysis / TCA':'#E45756','Fatty acids / lipid':'#F28E2B','One-carbon / folate':'#B0892D','Purine / urate':'#8E63CE','Tryptophan / indole':'#4E9ED4','Redox / sulfur':'#3BAE75','Translation / chaperone':'#6A3D9A','Proteostasis / proteasome':'#C44E52','Cytoskeleton / membrane':'#1F77B4','Vesicle trafficking':'#7A943B','Redox / antioxidant':'#1B9E77','Heme / globin / oxygen':'#E07B39'}
edges=[]
for c in clin_list:
    for f in met_nodes:
        r,p=stats.spearmanr(Xclin[c].loc[def_mask],Xmet[f].loc[def_mask])
        if np.isfinite(r) and p<0.05 and abs(r)>0.36: edges.append((('met',f),('clin',c),r,p))
    for f in prot_nodes:
        r,p=stats.spearmanr(Xclin[c].loc[def_mask],Xprot[f].loc[def_mask])
        if np.isfinite(r) and p<0.05 and abs(r)>0.36: edges.append((('clin',c),('prot',f),r,p))
for u,v,r,p in sorted(edges,key=lambda z:abs(z[2]),reverse=True)[:75]:
    x1,y1=pos[u]; x2,y2=pos[v]
    col='#D62728' if r>0 else '#1F77B4'
    lw=0.18+0.85*abs(r)  # reduced edge widths
    mid=0.18 if x2>x1 else -0.18
    path=Path([(x1,y1),(x1+mid,y1),(x2-mid,y2),(x2,y2)],[Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4])
    axF.add_patch(PathPatch(path,fc='none',ec=col,lw=lw,alpha=0.28,zorder=1))
axF.text(-1,1.05,'Metabolites',ha='center',fontsize=6.4,fontweight='bold')
axF.text(0,1.05,'Clinical',ha='center',fontsize=6.4,fontweight='bold')
axF.text(1,1.05,'Proteins',ha='center',fontsize=6.4,fontweight='bold')
for key,(x,y) in pos.items():
    kind,name=key
    if kind=='clin': col='#d9d9d9'; label=nice_feature_name(name,False); size=72
    elif kind=='met': col=mod_colors.get(node_module.get(name,''),'#999999'); label=nice_feature_name(name,True); size=88
    else: col=mod_colors.get(node_module.get(name,''),'#999999'); label=nice_feature_name(name,False); size=88
    axF.scatter([x],[y],s=size,fc=col,ec='white',lw=0.8,zorder=3)
    if kind=='met': axF.text(x-0.045,y,label,ha='right',va='center',fontsize=3.75)
    elif kind=='prot': axF.text(x+0.045,y,label,ha='left',va='center',fontsize=3.75)
    else: axF.text(x,y,label,ha='center',va='center',fontsize=3.15,color='0.25',zorder=4)
for i,(m,c) in enumerate(list(mod_colors.items())[:6]): axF.text(-1.55,0.72-i*0.18,m,color=c,fontsize=4.0,fontweight='bold',ha='left')
for i,(m,c) in enumerate(list(mod_colors.items())[6:]): axF.text(1.30,0.72-i*0.16,m,color=c,fontsize=4.0,fontweight='bold',ha='left')
axF.text(0,-1.04,'Nodes represent features within pathway modules with strong clinical Spearman correlations; edge color reflects correlation direction.',ha='center',fontsize=3.9,color='0.4')
axF.set_xlim(-1.65,1.65); axF.set_ylim(-1.12,1.15)
# G-K scatter
pairs=[('Emogas pHv','MYH10','Prot'),('vLac (mmol/L)','Biliverdin','Met'),('Transferrin (mg/dL)','CCT3','Prot'),('TOT Bil (mg/dL)','CA2','Prot'),('Ferritin (ng/mL)','ATP','Met')]
bottom_gs=GridSpecFromSubplotSpec(1,5,subplot_spec=gs[2,:],wspace=0.52)
for k,(xv,yv,kind) in enumerate(pairs):
    ax=fig.add_subplot(bottom_gs[0,k]); panel_label(ax,chr(ord('G')+k),x=-0.18,y=1.10)
    x=Xclin[xv].loc[def_mask]
    source=Xmet if kind=='Met' else Xprot
    if yv in source.columns: y=source[yv].loc[def_mask]
    else:
        cand=[c for c in source.columns if yv.lower() in c.lower()]
        y=source[cand[0]].loc[def_mask] if cand else pd.Series(np.nan,index=x.index)
    gg=groups[def_mask]
    for g in def_groups:
        m=gg==g; ax.scatter(x[m],y[m],s=14,color=palette[g],alpha=0.9,edgecolor='white',lw=0.35)
    r,p=stats.spearmanr(x,y); xx=np.linspace(x.min(),x.max(),100); slope,intercept=np.polyfit(x,y,1); ax.plot(xx,slope*xx+intercept,color='0.25',lw=1.05)
    ax.set_title(f'{nice_feature_name(xv)} vs {nice_feature_name(yv,kind=="Met")}',fontweight='bold',fontsize=6.2)
    ax.set_xlabel(nice_feature_name(xv),fontsize=4.8); ax.set_ylabel(nice_feature_name(yv,kind=='Met'),fontsize=4.8)
    ax.text(0.03,0.96,f'{kind} • ρ={r:.2f}; p={p:.1e}',transform=ax.transAxes,va='top',ha='left',fontsize=3.6,bbox=dict(fc='white',ec='none',alpha=0.72))
    ax.tick_params(labelsize=4.8)
handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=palette[g],markersize=5,label=labels_short[g]) for g in def_groups]
fig.legend(handles=handles,loc='upper left',ncol=1,frameon=True,fontsize=4.8,
           bbox_to_anchor=(0.025,0.565),borderpad=0.35,handletextpad=0.25,labelspacing=0.25)
fig.savefig(os.path.join(OUT,'Main_Figure_2_final_realdata.svg'),bbox_inches=None,pad_inches=0,facecolor='white')
fig.savefig(os.path.join(OUT,'Main_Figure_2_final_realdata.png'),dpi=300,bbox_inches=None,pad_inches=0,facecolor='white')
plt.close(fig)

# ---------------- FIGURE 3 ----------------
# classifier panels and NDD mapping
mask_defined=np.isin(groups,def_groups)
mask_unknown=np.isin(groups,['07-NDD','08-PDE4-associated'])
classes=def_labels
# Participant-grouped validation outputs are generated by run_statistical_revision.py.
metrics_table=pd.read_csv(os.path.join(ANALYSIS,'classifier_metrics.csv')).set_index('Model')
model_files=[
    ('Clinical','Clinical primary','confusion_clinical_primary.csv'),
    ('Metabolome','Metabolome','confusion_metabolome.csv'),
    ('Proteome','Proteome','confusion_proteome.csv'),
    ('Combined','Combined multi-omics','confusion_combined_multi-omics.csv'),
]
clf_results=[]
for display,metric_name,filename in model_files:
    cm=pd.read_csv(os.path.join(ANALYSIS,filename),index_col=0).reindex(index=classes,columns=classes).values
    row=metrics_table.loc[metric_name]
    rec=np.array([row[f'Recall_{label}'] for label in classes])
    clf_results.append((display,cm,float(row['Accuracy']),rec))
prob_cons=pd.read_csv(os.path.join(ANALYSIS,'unknown_consensus_probabilities.csv')).set_index('Participant_ID')[classes]

# Participant-level exploratory random-forest proximity embedding.
map_mask=mask_defined|mask_unknown
Xmap_samples=Xall.loc[map_mask].copy()
Xmap_samples['Participant_ID']=D.loc[map_mask,'Participant_ID'].values
Xmap=Xmap_samples.groupby('Participant_ID',sort=False).mean()
participant_groups=(D.loc[map_mask,['Participant_ID','Group']]
                    .drop_duplicates('Participant_ID')
                    .set_index('Participant_ID')
                    .reindex(Xmap.index)['Group'])
y_map=np.array([labels_short[g] for g in participant_groups])
rf=RandomForestClassifier(n_estimators=600,random_state=3,class_weight='balanced_subsample',max_features='sqrt')
rf.fit(Xmap,y_map)
leaves=rf.apply(Xmap)
prox=(leaves[:,None,:]==leaves[None,:,:]).mean(axis=2)
dist=1-prox
emb=MDS(n_components=2,dissimilarity='precomputed',random_state=2,normalized_stress='auto').fit_transform(dist)
# binary unknown-vs-defined RF importance, one row per participant
Xbin=Xmap
ybin=np.where(np.isin(participant_groups.values,['07-NDD','08-PDE4-associated']),'Unresolved/PDE4-associated','Defined')
rfb=RandomForestClassifier(n_estimators=800,random_state=4,class_weight='balanced_subsample',max_features='sqrt')
rfb.fit(Xbin,ybin)
imp=pd.Series(rfb.feature_importances_,index=Xbin.columns).sort_values(ascending=False).head(18)
feat_top=list(imp.index)
# representative variables use top known if available
rep_vars=[f for f in ['RAB5C','HPRT1','GYPC','Nicotinamide','Uric A (mg/dL)'] if f in Xall.columns]
if len(rep_vars)<5: rep_vars += [f for f in feat_top if f not in rep_vars][:5-len(rep_vars)]

fig=plt.figure(figsize=(8,8),facecolor='white')
gs=GridSpec(3,4,figure=fig,height_ratios=[0.90,1.02,1.02],
            left=0.072,right=0.975,top=0.965,bottom=0.285,hspace=0.78,wspace=0.43)
# confusion matrices A-D
for i,(name,cm,acc,rec) in enumerate(clf_results):
    ax=fig.add_subplot(gs[0,i]); panel_label(ax,chr(ord('A')+i),x=-0.19,y=1.15)
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(1,cm.max()))
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes,rotation=45,ha='right',fontsize=4.9); ax.set_yticklabels(classes,fontsize=4.9)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]): ax.text(c,r,str(cm[r,c]),ha='center',va='center',fontsize=5.4,color='white' if cm[r,c]>cm.max()/2 else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{name}\naccuracy={acc:.2f}',fontweight='bold',fontsize=6.6,pad=2)
    ax.text(0.5,-0.42,'Recall: '+' | '.join([f'{cl} {rv:.2f}' for cl,rv in zip(classes,rec)]),transform=ax.transAxes,ha='center',va='top',fontsize=3.7)
    ax.set_xticks(np.arange(-.5,len(classes),1),minor=True); ax.set_yticks(np.arange(-.5,len(classes),1),minor=True)
    ax.grid(which='minor',color='white',lw=1); ax.tick_params(which='minor',bottom=False,left=False)
# E viridis heatmap
axE=fig.add_subplot(gs[1,0:2]); panel_label(axE,'E',x=-0.08,y=1.10)
imE=axE.imshow(prob_cons.values,cmap='viridis',vmin=0,vmax=1,aspect='auto')
axE.set_yticks(range(len(prob_cons.index))); axE.set_yticklabels(prob_cons.index,fontsize=4.5)
axE.set_xticks(range(len(classes))); axE.set_xticklabels(classes,fontsize=5.2)
for r in range(prob_cons.shape[0]):
    for c in range(prob_cons.shape[1]):
        val=prob_cons.values[r,c]
        axE.text(c,r,f'{val:.2f}',ha='center',va='center',fontsize=4.5,color='white' if val<0.25 or val>0.65 else 'black')
axE.set_title('Unresolved/PDE4 consensus neighborhood probabilities',fontweight='bold',fontsize=7.2)
cb=fig.colorbar(imE,ax=axE,fraction=0.025,pad=0.02); cb.ax.tick_params(labelsize=7)
# F proximity
axF=fig.add_subplot(gs[1,2:4]); panel_label(axF,'F',x=-0.08,y=1.10)
map_groups=list(dict.fromkeys(y_map))
map_pal={lab:palette[[g for g,v in labels_short.items() if v==lab][0]] for lab in map_groups if lab in labels_short.values()}
# ensure order
for lab in ['VHL','EGLN1','EPAS1','EPOR','HAH','NDD','PDE4-associated']:
    if lab in y_map:
        m=y_map==lab; col=map_pal.get(lab,'#999999')
        axF.scatter(emb[m,0],emb[m,1],s=18,color=col,edgecolor='white',lw=0.4,alpha=0.95,label=lab,zorder=2)
        add_ellipse(axF,emb[m,0],emb[m,1],col,alpha=0.12)
        axF.text(emb[m,0].mean(),emb[m,1].mean(),lab,ha='center',va='center',fontsize=4.7,fontweight='bold',bbox=dict(fc='white',ec=col,lw=0.65,alpha=0.88,boxstyle='round,pad=0.12'))
axF.set_title('Random-forest proximity:\nunresolved cases map to multiple neighborhoods',fontweight='bold',fontsize=7.2)
axF.set_xlabel('Proximity PC1'); axF.set_ylabel('Proximity PC2')
axF.legend(loc='upper right',fontsize=4.3,frameon=True,ncol=1,borderpad=0.25,handletextpad=0.25,labelspacing=0.2)
axF.tick_params(labelsize=4.8)
# G colored RF importance
axG=fig.add_subplot(gs[2,0:2]); panel_label(axG,'G',x=-0.08,y=1.10)
colors=plt.cm.viridis(np.linspace(0.05,0.95,len(imp)))
ypos=np.arange(len(imp))[::-1]
axG.barh(ypos,imp.values,color=colors,edgecolor='white',lw=0.4)
axG.set_yticks(ypos); axG.set_yticklabels([nice_feature_name(f, f in Xmet.columns) for f in imp.index],fontsize=3.9)
axG.set_xlabel('Random forest importance')
axG.set_title('Top features distinguishing\nunresolved from defined erythrocytosis',fontweight='bold',fontsize=7.2)
axG.tick_params(axis='x',labelsize=4.8)
# H heatmap top features, square cells, horizontal top group labels
axH=fig.add_subplot(gs[2,2]); panel_label(axH,'H',x=-0.48,y=1.48)
features=feat_top[:14]
# participants sorted by defined and unresolved groups
Xh=Xmap[features]
yh=participant_groups.values
sort=[]
for g in def_groups+['07-NDD','08-PDE4-associated']:
    inds=np.where(yh==g)[0]
    for i in inds: sort.append((order.index(g),i))
idx=[i for _,i in sorted(sort)]
vals=Xh.iloc[idx].T.values
imH=axH.imshow(vals,aspect='equal',cmap='RdBu_r',vmin=-2.5,vmax=2.5,interpolation='nearest')
axH.set_yticks(range(len(features))); axH.set_yticklabels([nice_feature_name(f, f in Xmet.columns) for f in features],fontsize=3.5)
axH.set_xticks([]); axH.set_xlim(-0.5,vals.shape[1]-0.5); axH.set_ylim(vals.shape[0]-0.5,-2.1)
yarr=yh[idx]; start=0
for g in def_groups+['07-NDD','08-PDE4-associated']:
    n=(yarr==g).sum()
    if n:
        axH.add_patch(Rectangle((start-0.5,-1.18),n,0.30,color=palette[g],clip_on=False,lw=0))
        axH.text(start+n/2-0.5,-1.38,labels_short[g],ha='center',va='bottom',fontsize=3.0,clip_on=False)
        start+=n
axH.set_title('NDD-like feature pattern',fontweight='bold',fontsize=7.2,pad=20)
cb=fig.colorbar(imH,ax=axH,fraction=0.045,pad=0.02); cb.ax.tick_params(labelsize=7)
# I boxplot reps
axI=fig.add_subplot(gs[2,3]); panel_label(axI,'I',x=-0.18,y=1.12)
positions=[]; data=[]; colors=[]; ticklabels=[]; p=1
map_unknown=np.isin(participant_groups.values,['07-NDD','08-PDE4-associated'])
for v in rep_vars[:5]:
    data.append(Xmap.loc[~map_unknown, v].values); positions.append(p); colors.append('#d9d9d9')
    data.append(Xmap.loc[map_unknown, v].values); positions.append(p+0.32); colors.append('#b98b7e')
    ticklabels.append(nice_feature_name(v, v in Xmet.columns)); p+=1.05
bp=axI.boxplot(data,positions=positions,widths=0.26,patch_artist=True,showfliers=False)
for patch,c in zip(bp['boxes'],colors): patch.set_facecolor(c); patch.set_edgecolor('0.35'); patch.set_linewidth(0.8)
for med in bp['medians']: med.set_color('0.2'); med.set_linewidth(1)
axI.set_xticks([np.mean(positions[i*2:i*2+2]) for i in range(len(ticklabels))]); axI.set_xticklabels(ticklabels,rotation=35,ha='right',fontsize=4.5)
axI.set_ylabel('Autoscaled value')
axI.set_title('Representative\nNDD-associated variables',fontweight='bold',fontsize=7.2,pad=8)
axI.axhline(0,color='0.85',lw=0.7)
# small legend
axI.scatter([],[],s=30,color='#d9d9d9',edgecolor='0.35',label='Defined')
axI.scatter([],[],s=30,color='#b98b7e',edgecolor='0.35',label='Unresolved/PDE4-associated')
axI.legend(loc='upper right',fontsize=4.0,frameon=False)
fig.savefig(os.path.join(OUT,'Main_Figure_3_final_realdata.svg'),bbox_inches=None,pad_inches=0,facecolor='white')
fig.savefig(os.path.join(OUT,'Main_Figure_3_final_realdata.png'),dpi=300,bbox_inches=None,pad_inches=0,facecolor='white')
plt.close(fig)

# zip outputs
zip_path=os.path.join(OUT,'Main_Figures_1_2_3_final_realdata_svg.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in ['Main_Figure_1_final_realdata.svg','Main_Figure_2_final_realdata.svg','Main_Figure_3_final_realdata.svg']:
        z.write(os.path.join(OUT,f),arcname=f)
    z.write(__file__, arcname='generate_main_figures_final_realdata.py')
print('Saved final SVGs:')
for f in ['Main_Figure_1_final_realdata.svg','Main_Figure_2_final_realdata.svg','Main_Figure_3_final_realdata.svg', 'Main_Figures_1_2_3_final_realdata_svg.zip']:
    print(os.path.join(OUT,f))
