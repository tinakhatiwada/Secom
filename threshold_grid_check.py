"""
Grid-floor sensitivity check for Balanced Bagging + SMOTE on SECOM.
For each of 10 seeds, run the paper's single-split pipeline and select the
threshold on the (disjoint) calibration-validation slice with TWO grids:
  A: 0.05..0.95 step 0.01   (original; floor 0.05)
  B: 0.005..0.95 step 0.005 (expanded floor 0.005)
Report whether the expanded grid changes the selected threshold / validation F1
for the seeds that hit the 0.05 floor. The test set is NOT used to choose the grid.
"""
import warnings, numpy as np, pandas as pd, json
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE

DATA=Path("data/secom")
X=pd.read_csv(DATA/"secom.data",sep=r"\s+",header=None).values
y=np.array([0 if int(float(l.split()[0]))==-1 else 1
            for l in (DATA/"secom_labels.data").read_text().strip().splitlines()])

def best_thr(yv,pv,grid):
    bs,bt=0.0,grid[0]
    for t in grid:
        f=f1_score(yv,(pv>=t).astype(int),zero_division=0)
        if f>bs: bs,bt=f,float(t)
    return round(bt,3),bs

gridA=np.arange(0.05,0.95,0.01); gridB=np.arange(0.005,0.95,0.005)
rows=[]
for seed in [42,1,2,3,4,5,6,7,8,9]:
    Xtp,Xte,ytp,yte=train_test_split(X,y,test_size=0.20,random_state=seed,stratify=y)
    Xtr,Xcal,ytr,ycal=train_test_split(Xtp,ytp,test_size=0.20,random_state=seed,stratify=ytp)
    imp=SimpleImputer(strategy="median"); Xtr=imp.fit_transform(Xtr)
    sc=RobustScaler(); Xtr=sc.fit_transform(Xtr)
    sel=RFE(RandomForestClassifier(n_estimators=50,random_state=seed,n_jobs=-1),
            n_features_to_select=min(200,Xtr.shape[1]),step=25).fit(Xtr,ytr)
    Xtr=sel.transform(Xtr)
    Xcal=sel.transform(sc.transform(imp.transform(Xcal)))
    Xte =sel.transform(sc.transform(imp.transform(Xte)))
    Xr,yr=SMOTE(random_state=seed,k_neighbors=min(5,int(ytr.sum())-1)).fit_resample(Xtr,ytr)
    base=BalancedBaggingClassifier(n_estimators=100,random_state=seed).fit(Xr,yr)
    Xcf,Xcv,ycf,ycv=train_test_split(Xcal,ycal,test_size=0.50,random_state=seed,stratify=ycal)
    cal=CalibratedClassifierCV(FrozenEstimator(base),method="isotonic").fit(Xcf,ycf)
    pcv=cal.predict_proba(Xcv)[:,1]; pte=cal.predict_proba(Xte)[:,1]
    tA,vA=best_thr(ycv,pcv,gridA); tB,vB=best_thr(ycv,pcv,gridB)
    fA=f1_score(yte,(pte>=tA).astype(int),zero_division=0)
    fB=f1_score(yte,(pte>=tB).astype(int),zero_division=0)
    rows.append(dict(seed=seed,thrA=tA,valF1_A=round(vA,3),testF1_A=round(fA,3),
                     thrB=tB,valF1_B=round(vB,3),testF1_B=round(fB,3),
                     hitfloorA=(tA<=0.05)))
    print(f"  seed {seed:>2}: A thr={tA:.3f} valF1={vA:.3f} testF1={fA:.3f} | "
          f"B thr={tB:.3f} valF1={vB:.3f} testF1={fB:.3f}"
          f"{'  <-A at floor' if tA<=0.05 else ''}",flush=True)

df=pd.DataFrame(rows)
nfloor=int(df.hitfloorA.sum())
moved=df[(df.hitfloorA)&(df.thrB<0.05)]
print(f"\nSeeds at A-floor (0.05): {nfloor}/10")
print(f"Of those, expanded grid B chose a threshold below 0.05 in {len(moved)} seed(s)")
print(f"Mean validation F1: A={df.valF1_A.mean():.3f}  B={df.valF1_B.mean():.3f}  (Δ={df.valF1_B.mean()-df.valF1_A.mean():+.3f})")
print(f"Mean test F1:       A={df.testF1_A.mean():.3f}  B={df.testF1_B.mean():.3f}  (Δ={df.testF1_B.mean()-df.testF1_A.mean():+.3f})")
print(f"Threshold A: {df.thrA.mean():.3f} ± {df.thrA.std():.3f}   Threshold B: {df.thrB.mean():.3f} ± {df.thrB.std():.3f}")
Path("threshold_grid_check.json").write_text(df.to_json(orient="records"))
print("saved threshold_grid_check.json")