from _settings import data_dir, prior_dir, output_dir, external_root
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import re
warnings.filterwarnings('ignore')
PROJECT_ROOT = external_root()

def is_tumor_sample(sample_id):
    try:
        if not isinstance(sample_id, str):
            return False
        parts = sample_id.split('-')
        if len(parts) >= 4:
            sample_code = parts[3][:2]
            if sample_code.isdigit():
                code = int(sample_code)
                return code < 10
        return True
    except:
        return False

def get_patient_id(sample_id):
    sample_id = str(sample_id).upper()
    if sample_id.startswith('TCGA'):
        parts = sample_id.split('-')
        if len(parts) >= 3:
            return '-'.join(parts[:3])
    return sample_id

def standardize_mirna_id_fixed(mirna_id):
    mirna_id = str(mirna_id)
    mirna_id = re.sub('miR', 'mir', mirna_id, flags=re.IGNORECASE)
    parts = mirna_id.split('-')
    if len(parts) > 3:
        if parts[0] == 'hsa':
            mirna_id = '-'.join(parts[:3])
        else:
            mirna_id = '-'.join(parts[:-1])
    return mirna_id

def align_and_preprocess_dataset(dataset_name, raw_data_dir, output_dir):
    print(f"\n{'=' * 80}")
    print(f'处理数据集: {dataset_name}')
    print('=' * 80)
    dataset_dir = raw_data_dir / dataset_name
    label_file = output_dir / 'stage' / dataset_name / 'label.csv'
    if not label_file.exists():
        print(f' 标签文件不存在: {label_file}')
        return False
    try:
        print('读取并处理标签文件...')
        label_df = pd.read_csv(label_file)
        label_dict = {}
        for (_, row) in label_df.iterrows():
            pid = get_patient_id(row['sample_id'])
            label_dict[pid] = row['label']
        print(f' 有效标签患者数: {len(label_dict)}')
        mrna_file = dataset_dir / 'TPM.txt'
        if not mrna_file.exists():
            print(f' mRNA文件不存在: {mrna_file}')
            return False
        print('读取mRNA数据...')
        mrna_df = pd.read_csv(mrna_file, sep='\t', index_col=0, low_memory=False)
        mirna_file = dataset_dir / 'miRNA_RPM.txt'
        if not mirna_file.exists():
            print(f' miRNA文件不存在: {mirna_file}')
            return False
        print('读取miRNA数据...')
        mirna_df = pd.read_csv(mirna_file, sep='\t', index_col=0, low_memory=False)
        mirna_df.index = mirna_df.index.map(standardize_mirna_id_fixed)
        mirna_df = mirna_df.groupby(mirna_df.index).mean()
        print(f' miRNA ID已标准化并合并重复项')

        def filter_and_map_cols(df, type_name):
            valid_cols = []
            patient_to_col = {}
            for col in df.columns:
                pid = get_patient_id(col)
                if pid not in label_dict:
                    continue
                if col.startswith('TCGA') and (not is_tumor_sample(col)):
                    continue
                if pid not in patient_to_col:
                    patient_to_col[pid] = col
                    valid_cols.append(col)
            print(f'  {type_name} 筛选后保留: {len(valid_cols)} 样本 (已剔除正常/重复样本)')
            return (df[valid_cols], patient_to_col)
        (mrna_filtered, mrna_pid_map) = filter_and_map_cols(mrna_df, 'mRNA')
        (mirna_filtered, mirna_pid_map) = filter_and_map_cols(mirna_df, 'miRNA')
        common_pids = sorted(list(set(mrna_pid_map.keys()) & set(mirna_pid_map.keys())))
        if len(common_pids) == 0:
            print(' 没有共同样本，跳过该数据集')
            return False
        print(f' 共同有效患者数: {len(common_pids)}')
        mrna_cols = [mrna_pid_map[pid] for pid in common_pids]
        mirna_cols = [mirna_pid_map[pid] for pid in common_pids]
        mrna_aligned = mrna_filtered[mrna_cols]
        mirna_aligned = mirna_filtered[mirna_cols]
        mrna_aligned.columns = common_pids
        mirna_aligned.columns = common_pids
        label_aligned = pd.DataFrame({'patient_id': common_pids, 'sample_id': common_pids, 'label': [label_dict[pid] for pid in common_pids]})
        print(f'\nmRNA预处理:')
        mrna_aligned = mrna_aligned.loc[(mrna_aligned != 0).any(axis=1)]
        missing_rate = mrna_aligned.isna().mean(axis=1)
        mrna_aligned = mrna_aligned.loc[missing_rate <= 0.2]
        if mrna_aligned.shape[0] > 0:
            var_thresh = mrna_aligned.var(axis=1).quantile(0.75)
            mrna_aligned = mrna_aligned.loc[mrna_aligned.var(axis=1) > var_thresh]
        mrna_aligned = mrna_aligned.apply(lambda row: row.fillna(row.median()), axis=1)
        print(f' mRNA最终维度: {mrna_aligned.shape}')
        print(f'\nmiRNA预处理:')
        mirna_aligned = mirna_aligned.loc[(mirna_aligned != 0).any(axis=1)]
        missing_rate = mirna_aligned.isna().mean(axis=1)
        mirna_aligned = mirna_aligned.loc[missing_rate <= 0.2]
        if mirna_aligned.shape[0] > 0:
            var_thresh = mirna_aligned.var(axis=1).quantile(0.75)
            mirna_aligned = mirna_aligned.loc[mirna_aligned.var(axis=1) > var_thresh]
        mirna_aligned = mirna_aligned.apply(lambda row: row.fillna(row.median()), axis=1)
        print(f' miRNA最终维度: {mirna_aligned.shape}')
        final_genes_mrna = mrna_aligned.index.tolist()
        final_genes_mirna = mirna_aligned.index.tolist()
        if len(final_genes_mrna) == 0 or len(final_genes_mirna) == 0:
            print(' 预处理后特征数量为0，跳过保存')
            return False
        output_dataset_dir = output_dir / 'stage' / dataset_name
        output_dataset_dir.mkdir(parents=True, exist_ok=True)
        mrna_output = output_dataset_dir / 'mRNA.csv'
        mrna_aligned.to_csv(mrna_output)
        mirna_output = output_dataset_dir / 'miRNA.csv'
        mirna_aligned.to_csv(mirna_output)
        label_output = output_dataset_dir / 'label.csv'
        label_aligned.to_csv(label_output, index=False)
        print(f'\n 数据已保存到: {output_dataset_dir}')
        counts = label_aligned['label'].value_counts()
        (early, late) = (counts.get(0, 0), counts.get(1, 0))
        total = early + late
        return {'dataset': dataset_name, 'total_samples': total, 'early_count': early, 'late_count': late, 'mrna_genes': len(final_genes_mrna), 'mirna_count': len(final_genes_mirna)}
    except Exception as e:
        print(f' 处理出错: {e}')
        import traceback
        traceback.print_exc()
        return False

def process_all_datasets(dataset_names=None):
    if dataset_names is None:
        dataset_names = ['KIRC', 'MESO', 'STAD', 'THCA']
    raw_data_dir = external_root() / 'raw' / 'TCGA'
    output_dir = external_root()
    print('开始批量处理...')
    success_count = 0
    stats = []
    for name in dataset_names:
        res = align_and_preprocess_dataset(name, raw_data_dir, output_dir)
        if res:
            success_count += 1
            stats.append(res)
    if stats:
        df_stats = pd.DataFrame(stats)
        print('\n处理汇总:')
        print(df_stats.to_string())
        df_stats.to_csv(output_dir / 'stage' / 'dataset_summary.csv', index=False)
if __name__ == '__main__':
    process_all_datasets()
