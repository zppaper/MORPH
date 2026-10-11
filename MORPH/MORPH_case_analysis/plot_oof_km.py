from _settings import data_dir, prior_dir, output_dir, external_root
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
REQUIRED_COLUMNS = {'dataset', 'model', 'seed', 'fold', 'sample_id', 'time', 'event', 'risk'}

def parse_args():
    script_path = Path(__file__).resolve()
    package_root = script_path.parents[1]
    parser = argparse.ArgumentParser(description='Plot LAML/SKCM OOF Kaplan--Meier risk stratification.')
    parser.add_argument('--input', type=Path, help='OOF prediction CSV produced by the fixed survival protocol.', required=True)
    parser.add_argument('--output-dir', type=Path, default=output_dir('plot_oof_km'), help='Directory for figures and tables.')
    parser.add_argument('--model', default='MORPH-Cox')
    parser.add_argument('--datasets', nargs='+', default=['LAML', 'SKCM'], help='Cohorts to plot, in panel order.')
    parser.add_argument('--max-plot-years', type=float, default=15.0, help='Display limit only; statistical tests always use complete follow-up.')
    return parser.parse_args()

def validate_and_select(path, model, datasets):
    if not path.is_file():
        raise FileNotFoundError('OOF prediction file not found: {}'.format(path))
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError('Missing required columns: {}'.format(sorted(missing)))
    frame = frame.loc[frame['model'].eq(model) & frame['dataset'].isin(datasets)].copy()
    if frame.empty:
        raise ValueError('No rows found for model={} and datasets={}'.format(model, datasets))
    frame['time'] = pd.to_numeric(frame['time'], errors='raise')
    frame['event'] = pd.to_numeric(frame['event'], errors='raise').astype(int)
    frame['risk'] = pd.to_numeric(frame['risk'], errors='raise')
    if not np.isfinite(frame[['time', 'event', 'risk']].to_numpy()).all():
        raise ValueError('Non-finite time, event, or risk values were detected.')
    if (frame['time'] <= 0).any():
        raise ValueError('Survival times must be positive.')
    if not set(frame['event'].unique()).issubset({0, 1}):
        raise ValueError('Event must contain only 0/1 values.')
    duplicate_keys = frame.duplicated(['dataset', 'seed', 'fold', 'sample_id'], keep=False)
    if duplicate_keys.any():
        raise ValueError('Duplicate dataset/seed/fold/sample_id OOF rows were detected.')
    for (dataset, group) in frame.groupby('dataset', sort=False):
        label_counts = group.groupby('sample_id')[['time', 'event']].nunique()
        if (label_counts > 1).any().any():
            raise ValueError('Inconsistent outcomes across seeds for {}.'.format(dataset))
        expected_seeds = group['seed'].nunique()
        predictions_per_patient = group.groupby('sample_id')['seed'].nunique()
        if not predictions_per_patient.eq(expected_seeds).all():
            raise ValueError('Incomplete seed coverage for {}.'.format(dataset))
    absent = [dataset for dataset in datasets if dataset not in set(frame['dataset'])]
    if absent:
        raise ValueError('Requested datasets absent from OOF file: {}'.format(absent))
    return frame

def aggregate_patient_risk(frame):
    frame = frame.copy()
    frame['fold_risk_percentile'] = frame.groupby(['dataset', 'seed', 'fold'], sort=False)['risk'].transform(lambda values: values.rank(method='average', pct=True))
    patient = frame.groupby(['dataset', 'sample_id'], as_index=False).agg(time=('time', 'first'), event=('event', 'first'), oof_risk_percentile=('fold_risk_percentile', 'mean'), raw_oof_risk_mean=('risk', 'mean'), n_seeds=('seed', 'nunique')).sort_values(['dataset', 'sample_id']).reset_index(drop=True)
    cutoffs = patient.groupby('dataset')['oof_risk_percentile'].median()
    patient['median_cutoff'] = patient['dataset'].map(cutoffs)
    patient['risk_group'] = np.where(patient['oof_risk_percentile'] > patient['median_cutoff'], 'High risk', 'Low risk')
    return patient

def analyze_dataset(patient_frame, dataset):
    data = patient_frame.loc[patient_frame['dataset'].eq(dataset)].copy()
    high = data['risk_group'].eq('High risk')
    low = ~high
    logrank = logrank_test(data.loc[high, 'time'], data.loc[low, 'time'], event_observed_A=data.loc[high, 'event'], event_observed_B=data.loc[low, 'event'])
    cox_data = data[['time', 'event']].copy()
    cox_data['high_risk'] = high.astype(int)
    cox = CoxPHFitter()
    cox.fit(cox_data, duration_col='time', event_col='event')
    hr = float(np.exp(cox.params_.loc['high_risk']))
    ci_log = cox.confidence_intervals_.loc['high_risk']
    ci_low = float(np.exp(ci_log.iloc[0]))
    ci_high = float(np.exp(ci_log.iloc[1]))
    return {'dataset': dataset, 'patients': int(len(data)), 'events': int(data['event'].sum()), 'high_risk_n': int(high.sum()), 'high_risk_events': int(data.loc[high, 'event'].sum()), 'low_risk_n': int(low.sum()), 'low_risk_events': int(data.loc[low, 'event'].sum()), 'risk_cutoff': float(data['median_cutoff'].iloc[0]), 'hazard_ratio_high_vs_low': hr, 'hr_ci95_low': ci_low, 'hr_ci95_high': ci_high, 'logrank_p': float(logrank.p_value), 'follow_up_max_days': float(data['time'].max())}

def format_p_value(value):
    if value < 0.001:
        return '{:.2e}'.format(value)
    return '{:.3f}'.format(value)

def plot_panels(patient_frame, summaries, datasets, output_dir, max_plot_years):
    colors = {'Low risk': '#2878B5', 'High risk': '#D9534F'}
    (fig, axes) = plt.subplots(1, len(datasets), figsize=(7.0 * len(datasets), 5.5))
    if len(datasets) == 1:
        axes = [axes]
    for (panel_index, (ax, dataset)) in enumerate(zip(axes, datasets)):
        data = patient_frame.loc[patient_frame['dataset'].eq(dataset)].copy()
        data['time_years'] = data['time'] / 365.25
        summary = summaries[dataset]
        fitted = []
        legend_labels = []
        for group_name in ['Low risk', 'High risk']:
            group = data.loc[data['risk_group'].eq(group_name)]
            kmf = KaplanMeierFitter(label=group_name)
            kmf.fit(durations=group['time_years'], event_observed=group['event'])
            kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.13, color=colors[group_name], linewidth=2.1, show_censors=True, censor_styles={'ms': 4, 'marker': '+'})
            fitted.append(kmf)
            legend_labels.append('{} (n={})'.format(group_name, len(group)))
        observed_max = float(data['time_years'].max())
        x_max = min(max_plot_years, math.ceil(observed_max))
        ax.set_xlim(0, max(x_max, 1.0))
        ax.set_ylim(0, 1.03)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Overall survival probability' if panel_index == 0 else '')
        ax.set_title(dataset, fontsize=13, fontweight='bold')
        ax.grid(axis='y', color='#D9D9D9', linewidth=0.6, alpha=0.65)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        (legend_handles, _) = ax.get_legend_handles_labels()
        ax.legend(legend_handles, legend_labels, loc='lower left', frameon=False, fontsize=9)
        annotation = 'HR = {hr:.2f} (95% CI {lo:.2f}–{hi:.2f})\nLog-rank $P$ = {p}'.format(hr=summary['hazard_ratio_high_vs_low'], lo=summary['hr_ci95_low'], hi=summary['hr_ci95_high'], p=format_p_value(summary['logrank_p']))
        ax.text(0.98, 0.97, annotation, transform=ax.transAxes, ha='right', va='top', fontsize=9.5, bbox={'boxstyle': 'round,pad=0.35', 'fc': 'white', 'ec': '#BFBFBF'})
        add_at_risk_counts(*fitted, ax=ax, rows_to_show=['At risk'], ypos=-0.27)
    fig.suptitle('Out-of-fold MORPH-Cox risk stratification', fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.27, wspace=0.38)
    png_path = output_dir / 'MORPH_Cox_OOF_KM.png'
    pdf_path = output_dir / 'MORPH_Cox_OOF_KM.pdf'
    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return (png_path, pdf_path)

def write_method_note(output_dir, input_path, summaries, max_plot_years):
    lines = ['MORPH-Cox out-of-fold Kaplan-Meier analysis', '', 'Input: {}'.format(input_path), '', 'Method: For every seed and outer fold, MORPH-Cox risk scores were obtained only for patients excluded from model training. Scores were converted to percentiles within each held-out fold and averaged across the three seeds. Patients were divided using the cohort median aggregated score; survival outcomes were not used to select or optimize the cutoff. Hazard ratios compare high- with low-risk groups. Log-rank tests and Cox models used complete follow-up.', '', 'Figure display: Follow-up is displayed up to {:.1f} years when a cohort has a longer sparse tail; all statistical analyses retain complete follow-up.'.format(max_plot_years), '', 'Results:']
    for (dataset, summary) in summaries.items():
        lines.append('- {dataset}: n={patients}, events={events}, HR={hr:.4f} (95% CI {lo:.4f}-{hi:.4f}), log-rank P={p:.6g}.'.format(dataset=dataset, patients=summary['patients'], events=summary['events'], hr=summary['hazard_ratio_high_vs_low'], lo=summary['hr_ci95_low'], hi=summary['hr_ci95_high'], p=summary['logrank_p']))
    note_path = output_dir / 'methods_and_results.txt'
    note_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return note_path

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof = validate_and_select(args.input, args.model, args.datasets)
    patient = aggregate_patient_risk(oof)
    summaries = {dataset: analyze_dataset(patient, dataset) for dataset in args.datasets}
    patient_path = args.output_dir / 'MORPH_Cox_OOF_patient_risk.csv'
    summary_path = args.output_dir / 'MORPH_Cox_OOF_KM_summary.csv'
    audit_path = args.output_dir / 'analysis_audit.json'
    patient.to_csv(patient_path, index=False)
    pd.DataFrame([summaries[dataset] for dataset in args.datasets]).to_csv(summary_path, index=False)
    audit = {'input': str(args.input.resolve()), 'model': args.model, 'datasets': args.datasets, 'risk_normalization': 'percentile within dataset/seed/outer-test-fold', 'seed_aggregation': 'mean percentile per patient', 'group_cutoff': 'cohort median; outcomes not used', 'cutoff_optimization': False, 'statistical_follow_up': 'complete', 'max_plot_years': args.max_plot_years}
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (png_path, pdf_path) = plot_panels(patient, summaries, args.datasets, args.output_dir, args.max_plot_years)
    note_path = write_method_note(args.output_dir, args.input.resolve(), summaries, args.max_plot_years)
    print('OOF Kaplan-Meier analysis completed.')
    for dataset in args.datasets:
        summary = summaries[dataset]
        print('{dataset}: n={patients}, events={events}, HR={hr:.4f} (95% CI {lo:.4f}-{hi:.4f}), log-rank P={p:.6g}'.format(dataset=dataset, patients=summary['patients'], events=summary['events'], hr=summary['hazard_ratio_high_vs_low'], lo=summary['hr_ci95_low'], hi=summary['hr_ci95_high'], p=summary['logrank_p']))
    for path in [png_path, pdf_path, patient_path, summary_path, audit_path, note_path]:
        print(path)
if __name__ == '__main__':
    main()
