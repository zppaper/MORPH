from __future__ import annotations
from _settings import external_root
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import requests
SCRIPT_VERSION = 'prepare-cptac-ccrcc-v1'
GDC_FILES_ENDPOINT = 'https://api.gdc.cancer.gov/files'
GDC_DATA_ENDPOINT = 'https://api.gdc.cancer.gov/data'
PROJECT_ID = 'CPTAC-3'
PRIMARY_SITE = 'Kidney'
PRIMARY_DIAGNOSIS = 'Renal cell carcinoma, NOS'
DATA_TYPES = ('Gene Expression Quantification', 'miRNA Expression Quantification')
MODALITY_BY_DATA_TYPE = {'Gene Expression Quantification': 'mrna', 'miRNA Expression Quantification': 'mirna'}

def package_root() -> Path:
    return Path(__file__).resolve().parent

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)

def atomic_csv(path: Path, frame: pd.DataFrame, *, sep: str=',', index: bool=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, sep=sep, index=index)
    os.replace(temporary, path)

def stage_to_label(value: object) -> int | None:
    text = re.sub('\\s+', ' ', str(value).strip().upper())
    text = text.replace('STAGE ', '')
    early = {'I', 'IA', 'IB', 'IC', 'II', 'IIA', 'IIB', 'IIC'}
    late = {'III', 'IIIA', 'IIIB', 'IIIC', 'IV', 'IVA', 'IVB', 'IVC'}
    if text in early:
        return 0
    if text in late:
        return 1
    return None

def normalize_mirna_id(value: object) -> str:
    text = re.sub('miR', 'mir', str(value), flags=re.IGNORECASE)
    parts = text.split('-')
    if len(parts) > 3:
        if parts[0] == 'hsa':
            text = '-'.join(parts[:3])
        else:
            text = '-'.join(parts[:-1])
    return text

def gdc_filters() -> dict[str, Any]:
    return {'op': 'and', 'content': [{'op': '=', 'content': {'field': 'cases.project.project_id', 'value': [PROJECT_ID]}}, {'op': 'in', 'content': {'field': 'data_type', 'value': list(DATA_TYPES)}}, {'op': '=', 'content': {'field': 'access', 'value': ['open']}}, {'op': '=', 'content': {'field': 'cases.primary_site', 'value': [PRIMARY_SITE]}}]}

def query_gdc(timeout_seconds: int) -> dict[str, Any]:
    fields = ','.join(['file_id', 'file_name', 'file_size', 'md5sum', 'data_type', 'data_format', 'access', 'analysis.workflow_type', 'cases.case_id', 'cases.submitter_id', 'cases.primary_site', 'cases.project.project_id', 'cases.diagnoses.primary_diagnosis', 'cases.diagnoses.ajcc_pathologic_stage', 'cases.samples.sample_id', 'cases.samples.submitter_id', 'cases.samples.sample_type', 'cases.samples.tissue_type', 'cases.samples.portions.analytes.aliquots.aliquot_id', 'cases.samples.portions.analytes.aliquots.submitter_id'])
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(GDC_FILES_ENDPOINT, data={'filters': json.dumps(gdc_filters(), separators=(',', ':')), 'format': 'JSON', 'size': '5000', 'fields': fields}, timeout=(30, timeout_seconds))
        response.raise_for_status()
        payload = response.json()
    total = int(payload['data']['pagination']['total'])
    hits = payload['data']['hits']
    if total != len(hits):
        raise RuntimeError(f'GDC response was truncated: total={total}, returned={len(hits)}')
    return payload

def iter_aliquots(sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for portion in sample.get('portions') or []:
        for analyte in portion.get('analytes') or []:
            for aliquot in analyte.get('aliquots') or []:
                yield aliquot

def diagnosis_stage(case: dict[str, Any]) -> tuple[str, int] | None:
    candidates: list[tuple[str, int]] = []
    for diagnosis in case.get('diagnoses') or []:
        primary = str(diagnosis.get('primary_diagnosis', '')).strip()
        stage = str(diagnosis.get('ajcc_pathologic_stage', '')).strip()
        label = stage_to_label(stage)
        if primary == PRIMARY_DIAGNOSIS and label is not None:
            candidates.append((stage, label))
    if not candidates:
        return None
    labels = {label for (_, label) in candidates}
    if len(labels) != 1:
        raise RuntimeError(f"Conflicting eligible stage labels for case {case.get('submitter_id')}: {candidates}")
    return sorted(candidates)[0]

def flatten_eligible_files(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hit in payload['data']['hits']:
        data_type = str(hit.get('data_type', ''))
        modality = MODALITY_BY_DATA_TYPE.get(data_type)
        if modality is None:
            continue
        for case in hit.get('cases') or []:
            resolved = diagnosis_stage(case)
            if resolved is None:
                continue
            (stage, label) = resolved
            for sample in case.get('samples') or []:
                if sample.get('sample_type') != 'Primary Tumor':
                    continue
                for aliquot in iter_aliquots(sample):
                    aliquot_id = str(aliquot.get('aliquot_id', '')).strip()
                    aliquot_submitter = str(aliquot.get('submitter_id', '')).strip()
                    if not aliquot_id or not aliquot_submitter:
                        continue
                    rows.append({'patient_id': str(case.get('submitter_id', '')), 'case_id': str(case.get('case_id', '')), 'sample_submitter_id': str(sample.get('submitter_id', '')), 'sample_id': str(sample.get('sample_id', '')), 'aliquot_submitter_id': aliquot_submitter, 'aliquot_id': aliquot_id, 'stage': stage, 'label': int(label), 'modality': modality, 'file_id': str(hit.get('file_id', '')), 'file_name': str(hit.get('file_name', '')), 'file_size': int(hit.get('file_size', 0)), 'md5sum': str(hit.get('md5sum', '')), 'workflow_type': str((hit.get('analysis') or {}).get('workflow_type', ''))})
    frame = pd.DataFrame(rows).drop_duplicates()
    if frame.empty:
        raise RuntimeError('The GDC query returned no eligible file records.')
    return frame

def select_exact_pairs(records: pd.DataFrame) -> pd.DataFrame:
    entity_columns = ['patient_id', 'case_id', 'sample_submitter_id', 'sample_id', 'aliquot_submitter_id', 'aliquot_id', 'stage', 'label']
    paired_rows: list[dict[str, Any]] = []
    for (entity, group) in records.groupby(entity_columns, sort=True):
        by_modality: dict[str, pd.Series] = {}
        for modality in ('mrna', 'mirna'):
            candidates = group[group['modality'].eq(modality)].sort_values(['file_id', 'file_name'])
            if candidates.empty:
                break
            by_modality[modality] = candidates.iloc[0]
        if len(by_modality) != 2:
            continue
        base = dict(zip(entity_columns, entity))
        for (modality, row) in by_modality.items():
            for column in ('file_id', 'file_name', 'file_size', 'md5sum', 'workflow_type'):
                base[f'{modality}_{column}'] = row[column]
        paired_rows.append(base)
    pairs = pd.DataFrame(paired_rows)
    if pairs.empty:
        raise RuntimeError('No exact mRNA/miRNA aliquot pairs were found.')
    selected = pairs.sort_values(['patient_id', 'sample_submitter_id', 'aliquot_submitter_id']).groupby('patient_id', sort=True, as_index=False).first().sort_values('patient_id').reset_index(drop=True)
    if selected['patient_id'].duplicated().any():
        raise RuntimeError('Patient duplication remains after sample selection.')
    if selected['aliquot_id'].duplicated().any():
        raise RuntimeError('Aliquot duplication remains after sample selection.')
    if set(selected['label'].unique()) != {0, 1}:
        raise RuntimeError('The selected external cohort does not contain both classes.')
    return selected

def create_or_load_manifest(args: argparse.Namespace) -> pd.DataFrame:
    audit_dir = args.work_dir / 'audit'
    selected_path = audit_dir / 'selected_samples.csv'
    if selected_path.exists() and (not args.refresh_manifest):
        selected = pd.read_csv(selected_path)
        print(f'Using frozen cohort manifest: {selected_path} ({len(selected)} patients)')
        return selected
    if selected_path.exists() and args.refresh_manifest:
        preserved = selected_path.with_name(f"selected_samples.pre_refresh_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        selected_path.replace(preserved)
        print(f'Preserved prior manifest: {preserved}')
    payload = query_gdc(args.timeout)
    audit_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(audit_dir / 'gdc_files_response.json', payload)
    records = flatten_eligible_files(payload)
    selected = select_exact_pairs(records)
    atomic_csv(audit_dir / 'eligible_file_records.csv', records)
    atomic_csv(selected_path, selected)
    manifest_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        for modality in ('mrna', 'mirna'):
            manifest_rows.append({'id': getattr(row, f'{modality}_file_id'), 'filename': getattr(row, f'{modality}_file_name'), 'md5': getattr(row, f'{modality}_md5sum'), 'size': int(getattr(row, f'{modality}_file_size')), 'state': 'released', 'modality': modality, 'patient_id': row.patient_id, 'sample_submitter_id': row.sample_submitter_id, 'aliquot_submitter_id': row.aliquot_submitter_id})
    manifest = pd.DataFrame(manifest_rows)
    atomic_csv(args.work_dir / 'manifests' / 'selected_gdc_manifest.tsv', manifest, sep='\t')
    flow = pd.DataFrame([{'stage': 'eligible GDC file-entity records', 'count': len(records)}, {'stage': 'patients with exact paired RNA aliquots', 'count': int(records.groupby(['patient_id', 'sample_id', 'aliquot_id'])['modality'].nunique().eq(2).groupby(level=0).any().sum())}, {'stage': 'selected one aliquot per patient', 'count': len(selected)}, {'stage': 'label 0: Stage I/II', 'count': int(selected['label'].eq(0).sum())}, {'stage': 'label 1: Stage III/IV', 'count': int(selected['label'].eq(1).sum())}])
    atomic_csv(audit_dir / 'cohort_flow.csv', flow)
    summary = {'script_version': SCRIPT_VERSION, 'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'project': PROJECT_ID, 'primary_site': PRIMARY_SITE, 'primary_diagnosis': PRIMARY_DIAGNOSIS, 'sample_type': 'Primary Tumor', 'pairing': 'exact RNA aliquot', 'one_aliquot_per_patient': 'lexicographically first sample/aliquot', 'stage_mapping': {'Stage I/II': 0, 'Stage III/IV': 1}, 'patients': int(len(selected)), 'label_counts': {str(key): int(value) for (key, value) in selected['label'].value_counts().sort_index().items()}, 'download_bytes': {modality: int(pd.to_numeric(manifest.loc[manifest['modality'].eq(modality), 'size']).sum()) for modality in ('mrna', 'mirna')}, 'external_result_used_for_selection': False}
    atomic_json(audit_dir / 'manifest_summary.json', summary)
    print('CPTAC-CCRCC cohort manifest frozen.')
    print(f"Patients: {summary['patients']}")
    print(f"Labels  : {summary['label_counts']}")
    for (modality, byte_count) in summary['download_bytes'].items():
        print(f'{modality:5s}  : {byte_count / 1000000000.0:.3f} GB')
    return selected

def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def download_path(raw_dir: Path, file_id: str, file_name: str) -> Path:
    return raw_dir / file_id / file_name

def preserve(path: Path, reason: str) -> Path:
    suffix = time.strftime('%Y%m%d_%H%M%S')
    preserved = path.with_name(f'{path.name}.{reason}_{suffix}')
    counter = 1
    while preserved.exists():
        preserved = path.with_name(f'{path.name}.{reason}_{suffix}_{counter}')
        counter += 1
    path.replace(preserved)
    return preserved

def file_is_complete(path: Path, expected_size: int, expected_md5: str, verify_md5: bool) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    return not verify_md5 or md5_file(path).lower() == expected_md5.lower()

def download_one(record: dict[str, Any], raw_dir: Path, retries: int, timeout_seconds: int, verify_md5: bool) -> tuple[str, str]:
    file_id = str(record['file_id'])
    file_name = str(record['file_name'])
    expected_size = int(record['file_size'])
    expected_md5 = str(record['md5sum'])
    target = download_path(raw_dir, file_id, file_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_is_complete(target, expected_size, expected_md5, verify_md5):
        return (file_id, 'already_complete')
    if target.exists():
        preserve(target, 'invalid')
    part = target.with_suffix(target.suffix + '.part')
    url = f'{GDC_DATA_ENDPOINT}/{file_id}'
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(1, retries + 1):
            try:
                existing = part.stat().st_size if part.exists() else 0
                headers = {'Range': f'bytes={existing}-'} if existing else {}
                with session.get(url, headers=headers, stream=True, timeout=(30, timeout_seconds)) as response:
                    if existing and response.status_code != 206:
                        preserve(part, 'range_unsupported')
                        existing = 0
                        response.close()
                        with session.get(url, stream=True, timeout=(30, timeout_seconds)) as fresh:
                            fresh.raise_for_status()
                            with part.open('wb') as stream:
                                for chunk in fresh.iter_content(1024 * 1024):
                                    if chunk:
                                        stream.write(chunk)
                    else:
                        response.raise_for_status()
                        mode = 'ab' if existing else 'wb'
                        with part.open(mode) as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                if chunk:
                                    stream.write(chunk)
                if part.stat().st_size != expected_size:
                    raise IOError(f'size mismatch {part.stat().st_size} != {expected_size}')
                if verify_md5 and md5_file(part).lower() != expected_md5.lower():
                    preserve(part, 'md5_mismatch')
                    raise IOError('MD5 mismatch')
                os.replace(part, target)
                return (file_id, 'downloaded')
            except Exception as error:
                if attempt == retries:
                    raise RuntimeError(f'{file_id}/{file_name} failed after {retries} attempts: {error}') from error
                time.sleep(min(30, 2 ** attempt))
    raise AssertionError('Unreachable download state.')

def selected_file_records(selected: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        for modality in ('mrna', 'mirna'):
            records.append({'patient_id': row.patient_id, 'modality': modality, 'file_id': getattr(row, f'{modality}_file_id'), 'file_name': getattr(row, f'{modality}_file_name'), 'file_size': int(getattr(row, f'{modality}_file_size')), 'md5sum': getattr(row, f'{modality}_md5sum')})
    unique = {str(record['file_id']): record for record in records}
    return [unique[key] for key in sorted(unique)]

def download_selected(args: argparse.Namespace, selected: pd.DataFrame) -> None:
    records = selected_file_records(selected)
    if args.gdc_client.is_file() and (not args.use_python_downloader):
        download_selected_with_gdc_client(args, records)
        return
    print(f'Downloading/resuming {len(records)} selected GDC files with {args.workers} workers...')
    status_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, record, args.raw_dir, args.retries, args.timeout, not args.skip_md5): record for record in records}
        completed = 0
        for future in as_completed(futures):
            record = futures[future]
            (file_id, status) = future.result()
            completed += 1
            status_rows.append({**record, 'status': status})
            if completed % 20 == 0 or completed == len(records):
                print(f'Downloaded/verified {completed}/{len(records)} files')
    atomic_csv(args.work_dir / 'audit' / 'download_status.csv', pd.DataFrame(status_rows).sort_values(['modality', 'patient_id']))

def download_selected_with_gdc_client(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    manifest_dir = args.work_dir / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, args.retries + 1):
        pending = []
        for record in records:
            path = download_path(args.raw_dir, str(record['file_id']), str(record['file_name']))
            if path.is_file() and path.stat().st_size == int(record['file_size']):
                continue
            pending.append({'id': record['file_id'], 'filename': record['file_name'], 'md5': record['md5sum'], 'size': int(record['file_size']), 'state': 'released'})
        print(f'GDC client: completed={len(records) - len(pending)}/{len(records)}, pending={len(pending)}, attempt={attempt}/{args.retries}', flush=True)
        if not pending:
            return
        pending_path = manifest_dir / 'pending_gdc_manifest.tsv'
        atomic_csv(pending_path, pd.DataFrame(pending), sep='\t')
        command = [str(args.gdc_client), 'download', '-m', str(pending_path), '-d', str(args.raw_dir), '-n', str(args.workers), '--retry-amount', str(args.retries), '--wait-time', '10', '--no-related-files', '--no-annotations']
        print('$', subprocess.list2cmdline(command), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            continue
        print(f'GDC client returned {result.returncode}; the next attempt will resume all completed and partial files.', flush=True)
        time.sleep(min(30, 2 ** attempt))
    missing = []
    for record in records:
        path = download_path(args.raw_dir, str(record['file_id']), str(record['file_name']))
        if not path.is_file() or path.stat().st_size != int(record['file_size']):
            missing.append(str(path))
    if missing:
        raise RuntimeError(f'GDC client download remains incomplete for {len(missing)} files. Examples:\n' + '\n'.join(missing[:10]))

def verify_selected_downloads(selected: pd.DataFrame, raw_dir: Path) -> None:
    missing: list[str] = []
    for record in selected_file_records(selected):
        path = download_path(raw_dir, str(record['file_id']), str(record['file_name']))
        if not path.is_file() or path.stat().st_size != int(record['file_size']):
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f'{len(missing)} selected files are missing/incomplete. Examples:\n' + '\n'.join(missing[:10]))

def load_reference_features(train_dir: Path) -> tuple[list[str], list[str], pd.Series, pd.Series]:
    mrna = pd.read_csv(train_dir / 'mRNA.csv', index_col=0)
    mirna = pd.read_csv(train_dir / 'miRNA.csv', index_col=0)
    mrna.index = mrna.index.astype(str)
    mirna.index = mirna.index.astype(str)
    if not mrna.index.is_unique or not mirna.index.is_unique:
        raise ValueError('TCGA-KIRC reference feature IDs must be unique.')
    return (mrna.index.tolist(), mirna.index.tolist(), mrna.median(axis=1), mirna.median(axis=1))

def read_one_mrna(path: Path, features: list[str], reference_medians: pd.Series) -> tuple[np.ndarray, int]:
    frame = pd.read_csv(path, sep='\t', comment='#', usecols=['gene_name', 'tpm_unstranded'], low_memory=False)
    frame['tpm_unstranded'] = pd.to_numeric(frame['tpm_unstranded'], errors='coerce')
    frame = frame.dropna(subset=['gene_name', 'tpm_unstranded'])
    expression = frame.groupby('gene_name', sort=False)['tpm_unstranded'].mean()
    expression = np.log2(expression.clip(lower=0.0) + 1.0)
    aligned = expression.reindex(features)
    missing = int(aligned.isna().sum())
    if missing:
        aligned = aligned.fillna(reference_medians.reindex(features))
    return (aligned.to_numpy(dtype=np.float32), missing)

def read_one_mirna(path: Path, features: list[str], reference_medians: pd.Series) -> tuple[np.ndarray, int]:
    frame = pd.read_csv(path, sep='\t', usecols=['miRNA_ID', 'reads_per_million_miRNA_mapped'], low_memory=False)
    frame['rpm'] = pd.to_numeric(frame['reads_per_million_miRNA_mapped'], errors='coerce')
    frame = frame.dropna(subset=['miRNA_ID', 'rpm']).copy()
    frame['miRNA_ID'] = frame['miRNA_ID'].map(normalize_mirna_id)
    frame['log_rpm'] = np.log2(frame['rpm'].clip(lower=0.0) + 1.0)
    expression = frame.groupby('miRNA_ID', sort=False)['log_rpm'].mean()
    aligned = expression.reindex(features)
    missing = int(aligned.isna().sum())
    if missing:
        aligned = aligned.fillna(reference_medians.reindex(features))
    return (aligned.to_numpy(dtype=np.float32), missing)

def preprocess(args: argparse.Namespace, selected: pd.DataFrame) -> None:
    verify_selected_downloads(selected, args.raw_dir)
    (mrna_features, mirna_features, mrna_reference_medians, mirna_reference_medians) = load_reference_features(args.train_dir)
    n_samples = len(selected)
    mrna_matrix = np.empty((len(mrna_features), n_samples), dtype=np.float32)
    mirna_matrix = np.empty((len(mirna_features), n_samples), dtype=np.float32)
    audit_rows: list[dict[str, Any]] = []
    for (index, row) in enumerate(selected.itertuples(index=False)):
        mrna_path = download_path(args.raw_dir, row.mrna_file_id, row.mrna_file_name)
        mirna_path = download_path(args.raw_dir, row.mirna_file_id, row.mirna_file_name)
        (mrna_values, mrna_missing) = read_one_mrna(mrna_path, mrna_features, mrna_reference_medians)
        (mirna_values, mirna_missing) = read_one_mirna(mirna_path, mirna_features, mirna_reference_medians)
        mrna_matrix[:, index] = mrna_values
        mirna_matrix[:, index] = mirna_values
        audit_rows.append({'patient_id': row.patient_id, 'sample_submitter_id': row.sample_submitter_id, 'aliquot_submitter_id': row.aliquot_submitter_id, 'stage': row.stage, 'label': int(row.label), 'mrna_missing_reference_features': mrna_missing, 'mirna_missing_reference_features': mirna_missing})
        if (index + 1) % 25 == 0 or index + 1 == n_samples:
            print(f'Preprocessed {index + 1}/{n_samples} patients')
    patient_order = selected['patient_id'].astype(str).tolist()
    mrna_output = pd.DataFrame(mrna_matrix, index=mrna_features, columns=patient_order)
    mirna_output = pd.DataFrame(mirna_matrix, index=mirna_features, columns=patient_order)
    labels = selected[['patient_id', 'label']].copy()
    labels.insert(1, 'sample_id', labels['patient_id'])
    if patient_order != labels['sample_id'].tolist():
        raise RuntimeError('Expression and label patient orders differ.')
    if mrna_output.isna().any().any() or mirna_output.isna().any().any():
        raise RuntimeError('NaN remains in final external matrices.')
    if not np.isfinite(mrna_output.to_numpy()).all():
        raise RuntimeError('Non-finite value in final mRNA matrix.')
    if not np.isfinite(mirna_output.to_numpy()).all():
        raise RuntimeError('Non-finite value in final miRNA matrix.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mrna_output.index.name = 'ID'
    mirna_output.index.name = 'ID'
    atomic_csv(args.output_dir / 'mRNA.csv', mrna_output, index=True)
    atomic_csv(args.output_dir / 'miRNA.csv', mirna_output, index=True)
    atomic_csv(args.output_dir / 'label.csv', labels)
    atomic_csv(args.output_dir / 'sample_metadata.csv', selected)
    atomic_csv(args.output_dir / 'sample_preprocessing_audit.csv', pd.DataFrame(audit_rows))
    feature_report = pd.DataFrame([{'modality': 'mRNA', 'training_features': len(mrna_features), 'external_features': mrna_output.shape[0], 'external_samples': mrna_output.shape[1], 'total_missing_values_imputed_from_training_median': int(sum((row['mrna_missing_reference_features'] for row in audit_rows))), 'external_fitted_feature_selection': False}, {'modality': 'miRNA', 'training_features': len(mirna_features), 'external_features': mirna_output.shape[0], 'external_samples': mirna_output.shape[1], 'total_missing_values_imputed_from_training_median': int(sum((row['mirna_missing_reference_features'] for row in audit_rows))), 'external_fitted_feature_selection': False}])
    atomic_csv(args.output_dir / 'feature_alignment_report.csv', feature_report)
    summary = {'script_version': SCRIPT_VERSION, 'dataset': 'CPTAC-CCRCC', 'purpose': f'zero-finetuning external stage validation from {args.train_dir.name}', 'samples': n_samples, 'label_counts': {str(key): int(value) for (key, value) in labels['label'].value_counts().sort_index().items()}, 'mRNA_shape': list(mrna_output.shape), 'miRNA_shape': list(mirna_output.shape), 'exact_RNA_aliquot_pairing': True, 'one_aliquot_per_patient': True, 'mRNA_transform': 'gene symbol mean, log2(TPM+1)', 'miRNA_transform': 'log2(RPM+1) per precursor row, original stage-pipeline miRNA-ID normalization, duplicate precursor mean', 'feature_space_source': str(args.train_dir), 'missing_feature_fill': f'{args.train_dir.name} feature median', 'external_fitted_variance_filter': False, 'external_fitted_scaling': False, 'external_labels_used_for_preprocessing': False}
    atomic_json(args.output_dir / 'preprocessing_summary.json', summary)
    print('CPTAC-CCRCC preprocessing complete.')
    print(f'Output : {args.output_dir}')
    print(f'Samples: {n_samples}')
    print(f"Labels : {summary['label_counts']}")
    print(f'mRNA   : {mrna_output.shape}')
    print(f'miRNA  : {mirna_output.shape}')

def parse_args() -> argparse.Namespace:
    root = package_root()
    parser = argparse.ArgumentParser(description='Prepare CPTAC-CCRCC for TCGA-THCA to CPTAC-CCRCC external validation.')
    parser.add_argument('--stage', choices=('manifests', 'download', 'preprocess', 'all'), default='all')
    parser.add_argument('--train-dir', type=Path, default=external_root() / 'stage' / 'THCA')
    parser.add_argument('--output-dir', type=Path, default=external_root() / 'external' / 'CPTAC_CCRCC')
    parser.add_argument('--work-dir', type=Path, default=external_root() / 'manifests' / 'CPTAC_CCRCC')
    parser.add_argument('--raw-dir', type=Path, default=external_root() / 'raw' / 'CPTAC_CCRCC')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--gdc-client', type=Path, default=Path('gdc-client.exe'))
    parser.add_argument('--use-python-downloader', action='store_true', help='Ignore an available GDC client and use the built-in downloader.')
    parser.add_argument('--skip-md5', action='store_true', help='Skip MD5 verification; size validation is still required.')
    parser.add_argument('--refresh-manifest', action='store_true', help='Query GDC again and preserve the previous frozen manifest. Do not use after inspecting external model results.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive.')
    if args.retries < 1:
        parser.error('--retries must be positive.')
    return args

def main() -> None:
    args = parse_args()
    args.train_dir = args.train_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.raw_dir = args.raw_dir.resolve()
    args.gdc_client = args.gdc_client.resolve()
    if not args.train_dir.is_dir():
        raise FileNotFoundError(f'TCGA-THCA training directory not found: {args.train_dir}')
    selected = create_or_load_manifest(args)
    if args.stage == 'manifests':
        return
    if args.stage in {'download', 'all'}:
        download_selected(args, selected)
    if args.stage in {'preprocess', 'all'}:
        preprocess(args, selected)
if __name__ == '__main__':
    main()
