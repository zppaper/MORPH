from _settings import external_root
import os
import pandas as pd
import numpy as np
datasets = os.environ.get('MORPH_COHORTS', 'KIRC,MESO,STAD,THCA,LAML,SKCM,UVM').split(',')
base_warehouse_dir = str(external_root() / 'raw' / 'TCGA')
print(f"{'=' * 60}")
print(f'开始批量处理 miRNA 数据 (RPM -> Log2)')
print(f"{'=' * 60}\n")
success_count = 0
skip_count = 0
fail_count = 0
for dataset in datasets:
    print(f'正在处理数据集: [{dataset}] ...')
    dataset_dir = os.path.join(base_warehouse_dir, dataset)
    input_file = os.path.join(dataset_dir, 'miRNA.RPM.txt')
    output_file = os.path.join(dataset_dir, 'miRNA_RPM.txt')
    if not os.path.exists(input_file):
        print(f'   跳过: 输入文件不存在 {input_file}')
        skip_count += 1
        continue
    try:
        df = pd.read_csv(input_file, sep='\t', index_col='ID')
        print(f'  -> 读取数据: {df.shape}')
        df_log2 = np.log2(df + 1)
        df_log2.to_csv(output_file, sep='\t', float_format='%.4f')
        print(f'   成功: 保存至 {output_file}')
        print(f'    (维度: {df_log2.shape})')
        success_count += 1
    except Exception as e:
        print(f'   失败: {e}')
        import traceback
        traceback.print_exc()
        fail_count += 1
    print('-' * 40)
print(f"\n{'=' * 60}")
print('所有任务完成！')
print(f'成功: {success_count} | 跳过: {skip_count} | 失败: {fail_count}')
print(f"{'=' * 60}")
