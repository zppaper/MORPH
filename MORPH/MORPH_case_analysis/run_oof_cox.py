import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import cox_data as morph
from cox_training import fit_morph, prepare_common_fold, set_seeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results'/'oof_cox')
    parser.add_argument('--device',default='auto')
    args = parser.parse_args()
    if args.data_dir is not None:
        morph.DATA_ROOT = args.data_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use an empty output directory')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    counts = {'LAML':(798,200),'SKCM':(997,300)}
    seeds = [1028,2024,3407]
    resources = morph.load_prior_resources()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows,metrics = [],[]
    for dataset,(ng,nm) in counts.items():
        xg,xm,genes,mirnas,labels = morph.load_dataset(dataset)
        for seed in seeds:
            cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
            for fold,(outer,test) in enumerate(cv.split(np.arange(len(labels)),labels.event.to_numpy()),1):
                tg,tm,vg,vm,gkeep,mkeep,_,_,_ = prepare_common_fold(xg,xm,outer,test,ng,nm)
                train_labels = labels.iloc[outer].reset_index(drop=True)
                local_train,local_val = train_test_split(np.arange(len(outer)),test_size=0.2,stratify=train_labels.event.to_numpy(),random_state=seed)
                set_seeds(seed*100+fold)
                risk,info = fit_morph(tg,tm,vg,vm,[genes[i] for i in gkeep],[mirnas[i] for i in mkeep],train_labels,local_train,local_val,resources,device,200,30,5e-4,1e-4,20)
                for j,i in enumerate(test):
                    label = labels.iloc[int(i)]
                    rows.append(dict(dataset=dataset,model='MORPH-Cox',seed=seed,fold=fold,sample_id=label.sample_id,time=float(label.time),event=int(label.event),risk=float(risk[j])))
                metrics.append(dict(dataset=dataset,seed=seed,fold=fold,c_index=morph.safe_cindex(labels.iloc[test],risk),**info))
                pd.DataFrame(rows).to_csv(args.output_dir/'oof_predictions.csv',index=False)
                pd.DataFrame(metrics).to_csv(args.output_dir/'fold_metrics.csv',index=False)
    (args.output_dir/'protocol.json').write_text(json.dumps(dict(seeds=seeds,folds=5,features=counts,epochs=200,patience=30,learning_rate=5e-4,weight_decay=1e-4,device=str(device),torch=torch.__version__),indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
