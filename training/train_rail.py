import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.rail_features import extract_rail_features

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train-dir',required=True); p.add_argument('--labels',required=True); p.add_argument('--output',default='models/rail_model.joblib'); a=p.parse_args()
    labels=pd.read_csv(a.labels); rows=[]; y=[]
    for _,r in labels.iterrows():
        rows.append(extract_rail_features(pd.read_csv(Path(a.train_dir)/r['filename'])).iloc[0]); y.append(r['label'])
    X=pd.DataFrame(rows).fillna(0); y=pd.Series(y)
    model=RandomForestClassifier(n_estimators=400,class_weight='balanced',random_state=42,n_jobs=-1)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    s=cross_val_score(model,X,y,scoring='f1_macro',cv=cv,n_jobs=-1)
    print('CV Macro F1: %.4f +/- %.4f'%(s.mean(),s.std()))
    model.fit(X,y); Path(a.output).parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,a.output)
if __name__=='__main__': main()
