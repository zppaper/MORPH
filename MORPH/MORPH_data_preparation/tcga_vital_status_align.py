from _settings import data_dir, prior_dir, output_dir, external_root
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
PROJECT_ROOT = external_root()

def extract_patient_id(sample_id):
    if isinstance(sample_id, str) and sample_id.startswith('TCGA-'):
        parts = sample_id.split('-')
        if len(parts) >= 3:
            return '-'.join(parts[:3])
    return sample_id

def align_and_preprocess_dataset(dataset_name, raw_data_dir, output_dir):
    print(f"\n{'=' * 80}")
    print(f'处理数据集: {dataset_name}')
    print('=' * 80)
    dataset_dir = raw_data_dir / dataset_name
    label_file = output_dir / 'vital_status' / dataset_name / 'label.csv'
    if not label_file.exists():
        print(f' 标签文件不存在: {label_file}')
        return False
    try:
        print('读取标签文件...')
        label_df = pd.read_csv(label_file)
        label_samples = set(label_df['sample_id'].str.upper())
        print(f' 标签样本数: {len(label_samples)}')
        mrna_file = dataset_dir / 'TPM.txt'
        if not mrna_file.exists():
            print(f' mRNA文件不存在: {mrna_file}')
            return False
        print('读取mRNA数据...')
        mrna_df = pd.read_csv(mrna_file, sep='\t', index_col=0, low_memory=False)
        print(f' mRNA原始样本数: {mrna_df.shape[1]}, 基因数: {mrna_df.shape[0]}')
        mirna_file = dataset_dir / 'miRNA_RPM.txt'
        if not mirna_file.exists():
            print(f' miRNA文件不存在: {mirna_file}')
            return False
        print('读取miRNA数据...')
        mirna_df = pd.read_csv(mirna_file, sep='\t', index_col=0, low_memory=False)
        print(f' miRNA原始样本数: {mirna_df.shape[1]}, miRNA数: {mirna_df.shape[0]}')

        def standardize_mirna_id(mirna_id):
            import re
            mirna_id = str(mirna_id)
            mirna_id = re.sub('miR', 'mir', mirna_id)
            if '-' in mirna_id:
                parts = mirna_id.rsplit('-', 1)
                if len(parts) == 2:
                    (base, suffix) = parts
                    if suffix and (not suffix[0].isdigit()):
                        mirna_id = base
            return mirna_id
        mirna_df.index = mirna_df.index.map(standardize_mirna_id)
        print(f' miRNA ID已标准化')

        def extract_patient_id(sample_id):
            sample_id = str(sample_id).upper()
            if sample_id.startswith('TCGA'):
                return sample_id[:12]
            return sample_id
        label_dict = {}
        for (_, row) in label_df.iterrows():
            patient_id = extract_patient_id(row['sample_id'])
            label_dict[patient_id] = row['label']
        mrna_cols_with_label = [col for col in mrna_df.columns if extract_patient_id(col) in label_dict]
        mirna_cols_with_label = [col for col in mirna_df.columns if extract_patient_id(col) in label_dict]
        mrna_patient_ids = {extract_patient_id(col) for col in mrna_cols_with_label}
        mirna_patient_ids = {extract_patient_id(col) for col in mirna_cols_with_label}
        common_patient_ids = mrna_patient_ids & mirna_patient_ids
        print(f'\n样本对齐:')
        print(f'  标签样本: {len(label_dict)}')
        print(f'  mRNA样本: {len(mrna_patient_ids)}')
        print(f'  miRNA样本: {len(mirna_patient_ids)}')
        print(f'  共同样本: {len(common_patient_ids)}')
        if len(common_patient_ids) == 0:
            print(' 没有共同样本，跳过该数据集')
            return False
        common_samples = common_patient_ids
        mrna_cols_to_keep = [col for col in mrna_df.columns if extract_patient_id(col) in common_patient_ids]
        mirna_cols_to_keep = [col for col in mirna_df.columns if extract_patient_id(col) in common_patient_ids]
        mrna_aligned = mrna_df[mrna_cols_to_keep].copy()
        mirna_aligned = mirna_df[mirna_cols_to_keep].copy()
        mrna_patient_ids = [extract_patient_id(col) for col in mrna_aligned.columns]
        mirna_patient_ids = [extract_patient_id(col) for col in mirna_aligned.columns]
        mrna_pid_to_col = {extract_patient_id(col): col for col in mrna_aligned.columns}
        mirna_pid_to_col = {extract_patient_id(col): col for col in mirna_aligned.columns}
        sorted_patient_ids = sorted(set(mrna_patient_ids) & set(mirna_patient_ids))
        mrna_aligned = mrna_aligned[[mrna_pid_to_col[pid] for pid in sorted_patient_ids]]
        mirna_aligned = mirna_aligned[[mirna_pid_to_col[pid] for pid in sorted_patient_ids]]
        label_data = []
        for col in mrna_aligned.columns:
            patient_id = extract_patient_id(col)
            label_data.append({'sample_id': col, 'label': label_dict[patient_id]})
        label_aligned = pd.DataFrame(label_data)
        print(f' 对齐后样本数: {len(sorted_patient_ids)}')
        print(f'\nmRNA预处理:')
        mrna_before = mrna_aligned.shape[0]
        mrna_aligned = mrna_aligned[(mrna_aligned != 0).any(axis=1)]
        mrna_after = mrna_aligned.shape[0]
        print(f'  移除全0基因: {mrna_before - mrna_after} 个')
        mrna_before = mrna_aligned.shape[0]
        missing_ratio = mrna_aligned.isna().sum(axis=1) / mrna_aligned.shape[1]
        mrna_aligned = mrna_aligned[missing_ratio <= 0.2]
        mrna_after = mrna_aligned.shape[0]
        print(f'  移除缺失值过多的基因(>20%): {mrna_before - mrna_after} 个')
        mrna_before = mrna_aligned.shape[0]
        variance = mrna_aligned.var(axis=1)
        variance_threshold = variance.quantile(0.75)
        mrna_aligned = mrna_aligned[variance > variance_threshold]
        mrna_after = mrna_aligned.shape[0]
        print(f'  移除低方差基因(下75%): {mrna_before - mrna_after} 个')
        missing_count = mrna_aligned.isna().sum().sum()
        if missing_count > 0:
            for idx in mrna_aligned.index:
                row_median = mrna_aligned.loc[idx].median()
                if pd.notna(row_median):
                    mrna_aligned.loc[idx] = mrna_aligned.loc[idx].fillna(row_median)
            print(f'  填充剩余缺失值: {missing_count} 个')
        print(f' mRNA最终: {mrna_aligned.shape[0]} 个基因, {mrna_aligned.shape[1]} 个样本')
        print(f'\nmiRNA预处理:')
        mirna_before = mirna_aligned.shape[0]
        mirna_aligned = mirna_aligned[(mirna_aligned != 0).any(axis=1)]
        mirna_after = mirna_aligned.shape[0]
        print(f'  移除全0 miRNA: {mirna_before - mirna_after} 个')
        mirna_before = mirna_aligned.shape[0]
        missing_ratio = mirna_aligned.isna().sum(axis=1) / mirna_aligned.shape[1]
        mirna_aligned = mirna_aligned[missing_ratio <= 0.2]
        mirna_after = mirna_aligned.shape[0]
        print(f'  移除缺失值过多的 miRNA(>20%): {mirna_before - mirna_after} 个')
        mirna_before = mirna_aligned.shape[0]
        variance = mirna_aligned.var(axis=1)
        variance_threshold = variance.quantile(0.75)
        mirna_aligned = mirna_aligned[variance > variance_threshold]
        mirna_after = mirna_aligned.shape[0]
        print(f'  移除低方差 miRNA(下75%): {mirna_before - mirna_after} 个')
        missing_count = mirna_aligned.isna().sum().sum()
        if missing_count > 0:
            for idx in mirna_aligned.index:
                row_median = mirna_aligned.loc[idx].median()
                if pd.notna(row_median):
                    mirna_aligned.loc[idx] = mirna_aligned.loc[idx].fillna(row_median)
            print(f'  填充剩余缺失值: {missing_count} 个')
        print(f' miRNA最终: {mirna_aligned.shape[0]} 个 miRNA, {mirna_aligned.shape[1]} 个样本')
        print(f'\n最终对齐:')
        print(f'  mRNA样本: {mrna_aligned.shape[1]}')
        print(f'  miRNA样本: {mirna_aligned.shape[1]}')
        print(f'  标签样本: {len(label_aligned)}')

        def extract_patient_id(sample_id):
            sample_id = str(sample_id).upper()
            if sample_id.startswith('TCGA'):
                return sample_id[:12]
            return sample_id
        mrna_pid_to_col = {extract_patient_id(col): col for col in mrna_aligned.columns}
        mirna_pid_to_col = {extract_patient_id(col): col for col in mirna_aligned.columns}
        label_pid_to_idx = {extract_patient_id(row['sample_id']): idx for (idx, row) in label_aligned.iterrows()}
        mrna_pids = set(mrna_pid_to_col.keys())
        mirna_pids = set(mirna_pid_to_col.keys())
        label_pids = set(label_pid_to_idx.keys())
        common_pids = mrna_pids & mirna_pids & label_pids
        common_pids = sorted(list(common_pids))
        if len(common_pids) == 0:
            print(' 没有共同样本，跳过保存')
            return False
        print(f'  共同样本: {len(common_pids)}')
        mrna_cols = [mrna_pid_to_col[pid] for pid in common_pids]
        mirna_cols = [mirna_pid_to_col[pid] for pid in common_pids]
        label_indices = [label_pid_to_idx[pid] for pid in common_pids]
        mrna_final = mrna_aligned[mrna_cols].copy()
        mirna_final = mirna_aligned[mirna_cols].copy()
        label_final = label_aligned.iloc[label_indices].copy()
        mrna_final.columns = common_pids
        mirna_final.columns = common_pids
        label_final = label_final.reset_index(drop=True)
        label_final['patient_id'] = label_final['sample_id'].apply(extract_patient_id)
        label_final = label_final.set_index('patient_id').loc[common_pids].reset_index(drop=True)
        label_final['sample_id'] = common_pids
        output_dataset_dir = output_dir / 'vital_status' / dataset_name
        output_dataset_dir.mkdir(parents=True, exist_ok=True)
        mrna_output = output_dataset_dir / 'mRNA.csv'
        mrna_final.to_csv(mrna_output)
        print(f'\n mRNA预处理数据已保存: {mrna_output} ({mrna_final.shape[0]} 基因, {mrna_final.shape[1]} 样本)')
        mirna_output = output_dataset_dir / 'miRNA.csv'
        mirna_final.to_csv(mirna_output)
        print(f' miRNA预处理数据已保存: {mirna_output} ({mirna_final.shape[0]} miRNA, {mirna_final.shape[1]} 样本)')
        label_output = output_dataset_dir / 'label.csv'
        label_final.to_csv(label_output, index=False)
        print(f' 对齐后的标签已保存: {label_output} ({len(label_final)} 样本)')
        label_counts = label_final['label'].value_counts().sort_index()
        alive_count = label_counts.get(0, 0)
        dead_count = label_counts.get(1, 0)
        total_count = len(label_final)
        balance_ratio = min(alive_count, dead_count) / max(alive_count, dead_count) if max(alive_count, dead_count) > 0 else 0
        stats_info = {'dataset': dataset_name, 'total_samples': total_count, 'alive_count': alive_count, 'dead_count': dead_count, 'alive_ratio': f'{alive_count / total_count * 100:.1f}%' if total_count > 0 else '0%', 'dead_ratio': f'{dead_count / total_count * 100:.1f}%' if total_count > 0 else '0%', 'balance_ratio': f'{balance_ratio:.3f}', 'mrna_genes': mrna_final.shape[0], 'mirna_count': mirna_final.shape[0]}
        print(f'\n标签分布统计:')
        print(f'  存活 (0): {alive_count} ({alive_count / total_count * 100:.1f}%)')
        print(f'  死亡 (1): {dead_count} ({dead_count / total_count * 100:.1f}%)')
        print(f'  平衡度: {balance_ratio:.3f}')
        return stats_info
    except Exception as e:
        print(f' 处理出错: {e}')
        import traceback
        traceback.print_exc()
        return False

def process_all_datasets(dataset_names=None):
    if dataset_names is None:
        dataset_names = ['LAML', 'SKCM', 'UVM']
    print('\n' + '╔' + '=' * 78 + '╗')
    print('║' + ' ' * 15 + '生存状态分类 - 样本对齐与数据预处理' + ' ' * 18 + '║')
    print('╚' + '=' * 78 + '╝')
    print()
    raw_data_dir = external_root() / 'raw' / 'TCGA'
    output_dir = external_root()
    print(f'原始数据目录: {raw_data_dir}')
    print(f'输出目录: {output_dir}\n')
    if not raw_data_dir.exists():
        print(f' 错误: 原始数据目录不存在: {raw_data_dir}')
        return
    print(f'待处理数据集 ({len(dataset_names)}):')
    for (i, name) in enumerate(dataset_names, 1):
        print(f'  {i}. {name}')
    print()
    success_count = 0
    failed_datasets = []
    stats_list = []
    for dataset_name in dataset_names:
        result = align_and_preprocess_dataset(dataset_name, raw_data_dir, output_dir)
        if result and isinstance(result, dict):
            success_count += 1
            stats_list.append(result)
        else:
            failed_datasets.append(dataset_name)
    print('\n' + '=' * 80)
    print('样本对齐与预处理完成')
    print('=' * 80)
    print(f'成功: {success_count}/{len(dataset_names)} 个数据集')
    if failed_datasets:
        print(f'\n失败的数据集 ({len(failed_datasets)}):')
        for dataset in failed_datasets:
            print(f'   {dataset}')
    if stats_list:
        print(f'\n生成汇总统计文件...')
        stats_df = pd.DataFrame(stats_list)
        summary_output = output_dir / 'vital_status' / 'dataset_summary.csv'
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(summary_output, index=False)
        print(f' 汇总统计已保存: {summary_output}')
        print(f"\n{'=' * 100}")
        print('数据集分布汇总统计')
        print(f"{'=' * 100}")
        print(stats_df.to_string(index=False))
        print(f"{'=' * 100}")
        total_samples = stats_df['total_samples'].sum()
        total_alive = stats_df['alive_count'].sum()
        total_dead = stats_df['dead_count'].sum()
        overall_balance = min(total_alive, total_dead) / max(total_alive, total_dead) if max(total_alive, total_dead) > 0 else 0
        print(f'\n总体统计:')
        print(f'  总样本数: {total_samples}')
        print(f'  存活样本: {total_alive} ({total_alive / total_samples * 100:.1f}%)')
        print(f'  死亡样本: {total_dead} ({total_dead / total_samples * 100:.1f}%)')
        print(f'  总体平衡度: {overall_balance:.3f}')
    if success_count > 0:
        print(f'\n 预处理数据保存位置:')
        print(f'  {output_dir}/vital_status/<数据集名>/')
        print(f'  - mRNA.csv: 预处理后的mRNA表达数据')
        print(f'  - miRNA.csv: 预处理后的miRNA表达数据')
        print(f'  - label.csv: 对齐后的生存标签')
        print(f'\n 汇总统计文件:')
        print(f'  {output_dir}/vital_status/dataset_summary.csv')
if __name__ == '__main__':
    process_all_datasets()
