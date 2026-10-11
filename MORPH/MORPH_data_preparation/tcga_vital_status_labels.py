from _settings import data_dir, prior_dir, output_dir, external_root
import pandas as pd
from pathlib import Path
PROJECT_ROOT = external_root()

def process_all_datasets(dataset_names=None):
    if dataset_names is None:
        dataset_names = ['LAML', 'SKCM', 'UVM']
    print('\n' + '╔' + '=' * 78 + '╗')
    print('║' + ' ' * 20 + '生存状态分类 - 标签生成' + ' ' * 28 + '║')
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
    for dataset_name in dataset_names:
        print(f"\n{'=' * 80}")
        print(f'处理数据集: {dataset_name}')
        print('=' * 80)
        clinical_file = raw_data_dir / dataset_name / 'clinical.txt'
        if not clinical_file.exists():
            print(f' 临床数据文件不存在: {clinical_file}')
            failed_datasets.append(dataset_name)
            continue
        try:
            print(f'读取临床数据文件...')
            clinical_df = pd.read_csv(clinical_file, sep='\t', low_memory=False)
            print(f' 成功读取 {len(clinical_df)} 条记录')
            required_columns = ['bcr_patient_barcode', 'survival_status']
            missing_columns = [col for col in required_columns if col not in clinical_df.columns]
            if missing_columns:
                print(f' 缺少必要的列: {missing_columns}')
                print(f'可用的列: {list(clinical_df.columns)}')
                failed_datasets.append(dataset_name)
                continue
            print(f' 找到必要的列: {required_columns}')
            label_df = clinical_df[['bcr_patient_barcode', 'survival_status']].copy()
            label_df.columns = ['sample_id', 'label']
            before_count = len(label_df)
            label_df = label_df.dropna(subset=['label'])
            after_count = len(label_df)
            if before_count > after_count:
                removed = before_count - after_count
                print(f'! 移除 {removed} 个标签缺失的样本')
            label_df['label'] = label_df['label'].astype(int)
            label_counts = label_df['label'].value_counts().sort_index()
            print(f'\n标签分布:')
            print('-' * 60)
            for (label_value, count) in label_counts.items():
                label_name = 'Alive (存活)' if label_value == 0 else 'Dead (死亡)'
                percentage = count / len(label_df) * 100
                print(f'  标签 {label_value} ({label_name}): {count:>4} 样本 ({percentage:>5.1f}%)')
            print('-' * 60)
            print(f'  总计: {len(label_df)} 样本')
            output_dataset_dir = output_dir / 'vital_status' / dataset_name
            output_dataset_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dataset_dir / 'label.csv'
            label_df.to_csv(output_file, index=False)
            print(f'\n 标签文件已保存')
            print(f'  位置: {output_file}')
            print(f'  格式: CSV (sample_id, label)')
            success_count += 1
        except Exception as e:
            print(f' 处理出错: {e}')
            import traceback
            traceback.print_exc()
            failed_datasets.append(dataset_name)
    print('\n' + '=' * 80)
    print('标签生成完成')
    print('=' * 80)
    print(f'成功: {success_count}/{len(dataset_names)} 个数据集')
    if failed_datasets:
        print(f'\n失败的数据集 ({len(failed_datasets)}):')
        for dataset in failed_datasets:
            print(f'   {dataset}')
    if success_count > 0:
        print(f'\n 标签文件保存位置:')
        print(f'  {output_dir}/vital_status/<数据集名>/label.csv')
if __name__ == '__main__':
    process_all_datasets()
