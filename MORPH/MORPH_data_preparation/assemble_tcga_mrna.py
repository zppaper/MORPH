from _settings import external_root
import os
import json
import glob
import pandas as pd
import numpy as np
datasets = os.environ.get('MORPH_COHORTS', 'KIRC,MESO,STAD,THCA,LAML,SKCM,UVM').split(',')
base_input_dir = str(external_root() / 'downloads' / 'TCGA')
base_output_dir = str(external_root() / 'raw' / 'TCGA')
print(f"{'=' * 60}")
print(f'开始批量处理 mRNA 数据 (保留所有样本 Tumor + Normal)')
print(f"{'=' * 60}\n")
success_count = 0
skip_count = 0
fail_count = 0
for dataset in datasets:
    print(f'正在处理数据集: [{dataset}] ...')
    current_work_dir = os.path.join(base_input_dir, dataset)
    metadata_path = os.path.join(current_work_dir, 'mRNA_metadata.json')
    current_output_dir = os.path.join(base_output_dir, dataset)
    if not os.path.exists(current_work_dir) or not os.path.exists(metadata_path):
        print(f'   跳过: 目录或 Metadata 不存在')
        skip_count += 1
        continue
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        file_to_barcode = {}
        for entry in metadata:
            file_name = entry['file_name']
            try:
                sample_id = entry['associated_entities'][0]['entity_submitter_id']
                file_to_barcode[file_name] = sample_id
            except:
                pass
        tsv_files = glob.glob(os.path.join(current_work_dir, 'mRNA', '**', '*.tsv'), recursive=True)
        if not tsv_files:
            print(f'   跳过: 未找到 .tsv 文件')
            skip_count += 1
            continue
        print(f'  -> 找到 {len(tsv_files)} 个数据文件，开始读取...')
        data_list = []
        gene_annotation = None
        valid_sample_count = 0
        for file_path in tsv_files:
            filename = os.path.basename(file_path)
            if filename not in file_to_barcode:
                continue
            full_barcode = file_to_barcode[filename]
            parts = full_barcode.split('-')
            if len(parts) < 4:
                continue
            patient_id = '-'.join(parts[:4])
            try:
                df = pd.read_csv(file_path, sep='\t', comment='#', header=0)
                df = df[~df['gene_id'].str.startswith('N_')]
                if gene_annotation is None:
                    gene_annotation = df[['gene_id', 'gene_name', 'gene_type']].set_index('gene_id')
                df = df.set_index('gene_id')
                if 'tpm_unstranded' in df.columns:
                    series = df['tpm_unstranded']
                else:
                    series = df.iloc[:, 5]
                series.name = patient_id
                if series.index.duplicated().any():
                    series = series[~series.index.duplicated(keep='first')]
                data_list.append(series)
                valid_sample_count += 1
            except Exception as e:
                pass
        if not data_list:
            print(f'   警告: 无有效数据提取')
            fail_count += 1
            continue
        tpm_matrix = pd.concat(data_list, axis=1)
        if tpm_matrix.columns.duplicated().any():
            tpm_matrix = tpm_matrix.groupby(tpm_matrix.columns, axis=1).mean()
        full_df = gene_annotation.join(tpm_matrix, how='inner')
        full_df = full_df[full_df['gene_type'] == 'protein_coding']
        full_df = full_df.drop(columns=['gene_type'])
        full_df = full_df.reset_index(drop=True).groupby('gene_name').mean()
        final_df = np.log2(full_df + 1)
        final_df.index.name = 'ID'
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        save_path = os.path.join(current_output_dir, 'TPM.txt')
        final_df.to_csv(save_path, sep='\t', float_format='%.4f')
        print(f'   成功: 保存至 {save_path}')
        print(f'    (维度: {final_df.shape}, 总样本数: {valid_sample_count})')
        success_count += 1
    except Exception as e:
        print(f'   失败: {e}')
        fail_count += 1
    print('-' * 40)
print(f"\n{'=' * 60}")
print('所有任务完成！')
print(f'成功: {success_count} | 跳过: {skip_count} | 失败: {fail_count}')
print(f"{'=' * 60}")
