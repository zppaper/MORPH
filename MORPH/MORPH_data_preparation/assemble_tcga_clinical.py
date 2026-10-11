from _settings import external_root
import os
import glob
import re
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
datasets = os.environ.get('MORPH_COHORTS', 'KIRC,MESO,STAD,THCA,LAML,SKCM,UVM').split(',')
base_output_dir = str(external_root() / 'raw' / 'TCGA')
base_xml_dir = str(external_root() / 'downloads' / 'TCGA')

def parse_tcga_xml_enhanced(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        patient_node = None
        for child in root:
            if child.tag.endswith('patient'):
                patient_node = child
                break
        if patient_node is None and len(root) > 1:
            patient_node = root[1]
        if patient_node is None:
            return None
        data_dict = {}
        followup_days_list = []
        death_days_list = []
        for element in patient_node.iter():
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            text = element.text.strip() if element.text else ''
            if not text:
                continue
            if tag == 'days_to_last_followup':
                try:
                    val = float(text)
                    if val >= 0:
                        followup_days_list.append(val)
                except:
                    pass
            elif tag == 'days_to_death':
                try:
                    val = float(text)
                    if val >= 0:
                        death_days_list.append(val)
                except:
                    pass
            if tag not in data_dict:
                data_dict[tag] = text
        if followup_days_list:
            data_dict['days_to_last_followup'] = max(followup_days_list)
        if death_days_list:
            data_dict['days_to_death'] = max(death_days_list)
        return data_dict
    except Exception as e:
        print(f'解析错误 {file_path}: {e}')
        return None

def parse_stage_event(event_str):
    if pd.isna(event_str) or str(event_str).strip() == '':
        return pd.Series([None, None, None, None])
    text = str(event_str)
    stage_match = re.search('Stage\\s*([IVX]+)', text, re.IGNORECASE)
    stage_val = 'Stage ' + stage_match.group(1).upper() if stage_match else None
    t_match = re.search('(T[0-9]+[a-z]?)', text, re.IGNORECASE)
    t_val = t_match.group(1).upper() if t_match else None
    n_match = re.search('(N[0-9]+[a-z]?)', text, re.IGNORECASE)
    n_val = n_match.group(1).upper() if n_match else None
    m_match = re.search('(M[0-9]+[a-z]?)', text, re.IGNORECASE)
    m_val = m_match.group(1).upper() if m_match else None
    return pd.Series([stage_val, t_val, n_val, m_val])

def process_clinical_data(df):
    print('正在处理特征工程 (含分期清洗)...')
    t_cols = ['pathologic_T', 'clinical_T', 'pathologic_t', 'clinical_t']
    n_cols = ['pathologic_N', 'clinical_N', 'pathologic_n', 'clinical_n']
    m_cols = ['pathologic_M', 'clinical_M', 'pathologic_m', 'clinical_m']
    stage_cols = ['pathologic_stage', 'clinical_stage', 'stage_event', 'tumor_stage']

    def find_first_val(row, candidates):
        for col in candidates:
            if col in row.index and pd.notna(row[col]) and (str(row[col]).strip() != ''):
                return str(row[col]).strip()
        return None
    df['T'] = df.apply(lambda row: find_first_val(row, t_cols), axis=1)
    df['N'] = df.apply(lambda row: find_first_val(row, n_cols), axis=1)
    df['M'] = df.apply(lambda row: find_first_val(row, m_cols), axis=1)
    df['Stage'] = df.apply(lambda row: find_first_val(row, stage_cols), axis=1)

    def clean_stage_name(text):
        if pd.isna(text):
            return None
        text = str(text).strip()
        match = re.search('(Stage\\s*[IVX]+)', text, re.IGNORECASE)
        if match:
            cleaned = match.group(1).upper()
            return cleaned.replace('STAGE', 'Stage')
        return None
    df['Stage'] = df['Stage'].apply(clean_stage_name)
    raw_stage_col = None
    for c in stage_cols:
        if c in df.columns:
            raw_stage_col = c
            break
    if raw_stage_col:

        def extract_tnm(text, pattern):
            if pd.isna(text):
                return None
            match = re.search(pattern, str(text), re.IGNORECASE)
            return match.group(1).upper() if match else None
        mask_t = df['T'].isna()
        df.loc[mask_t, 'T'] = df.loc[mask_t, raw_stage_col].apply(lambda x: extract_tnm(x, '(T[0-9]+[a-z]?)'))
        mask_n = df['N'].isna()
        df.loc[mask_n, 'N'] = df.loc[mask_n, raw_stage_col].apply(lambda x: extract_tnm(x, '(N[0-9]+[a-z]?)'))
        mask_m = df['M'].isna()
        df.loc[mask_m, 'M'] = df.loc[mask_m, raw_stage_col].apply(lambda x: extract_tnm(x, '(M[0-9]+[a-z]?)'))
    cols_time = ['days_to_last_followup', 'days_to_death']
    for c in cols_time:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['survival_time'] = np.nan
    df['survival_status'] = np.nan
    if 'vital_status' in df.columns:
        status = df['vital_status'].astype(str).str.upper().str.strip()
        is_dead = status == 'DEAD'
        is_alive = status == 'ALIVE'
        df.loc[is_dead, 'survival_status'] = 1
        df.loc[is_alive, 'survival_status'] = 0
        df.loc[is_dead, 'survival_time'] = df.loc[is_dead, 'days_to_death']
        df.loc[is_alive, 'survival_time'] = df.loc[is_alive, 'days_to_last_followup']
    df_clean = df.dropna(subset=['survival_time', 'survival_status']).copy()
    df_clean = df_clean[df_clean['survival_time'] > 0]
    order = ['bcr_patient_barcode', 'survival_status', 'survival_time', 'Stage', 'T', 'N', 'M', 'vital_status']
    final_cols = [c for c in order if c in df_clean.columns] + [c for c in df_clean.columns if c not in order]
    return df_clean[final_cols]
for dataset_name in datasets:
    print(f"\n{'=' * 50}")
    print(f'开始处理数据集: {dataset_name}')
    print(f"{'=' * 50}")
    xml_dir = os.path.join(base_xml_dir, dataset_name)
    output_dir = os.path.join(base_output_dir, dataset_name)
    if not os.path.exists(xml_dir):
        print(f'警告: XML目录不存在 {xml_dir}，跳过此数据集')
        continue
    xml_files = glob.glob(os.path.join(xml_dir, '**', '*.xml'), recursive=True)
    print(f'找到 {len(xml_files)} 个文件')
    if len(xml_files) == 0:
        print(f'警告: 未找到XML文件，跳过此数据集')
        continue
    data_list = []
    for f in xml_files:
        d = parse_tcga_xml_enhanced(f)
        if d:
            data_list.append(d)
    if data_list:
        df_raw = pd.DataFrame(data_list)
        df_final = process_clinical_data(df_raw)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, 'clinical.txt')
        df_final.to_csv(save_path, sep='\t', index=False)
        print(f' {dataset_name} 处理完成，保存至: {save_path}')
    else:
        print(f' {dataset_name} 未提取到数据')
print(f"\n{'=' * 50}")
print('所有数据集处理完成！')
print(f"{'=' * 50}")
