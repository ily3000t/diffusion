"""Render post-hoc channel attribution and frozen denoising traces."""
import argparse
import json
import os
from pathlib import Path
import sys
from sumodiff.data.dataset import write_json
from sumodiff.experiments.recorder import RunRecorder, file_identity

ORDER = ('raw', 'decoded', 'oracle_position', 'oracle_velocity',
         'oracle_heading', 'oracle_position_velocity', 'label_decoded',
         'current_constant_acceleration')
LABELS = ('Raw prediction', 'Decoded prediction', 'True positions',
          'True velocities', 'True heading', 'True positions + velocities',
          'True labels (decoded)', 'Current acceleration control')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def render(audit_path, output, formal=False):
    root = Path(audit_path).resolve(strict=True)
    manifest, metrics = read(root/'manifest.json'), read(root/'audit_summary.json')
    if read(root/'status.json')['state'] != 'completed':
        raise ValueError('Cannot plot an incomplete audit')
    if metrics['run_sha'] != manifest['git']['commit_sha']:
        raise ValueError('Source audit SHA mismatch')
    if formal and (not manifest['formal'] or manifest['git']['dirty']):
        raise ValueError('Formal report requires a clean formal audit')
    totals = {}
    for name, case in metrics['attribution'].items():
        keys = set().union(*(g.keys() for g in case['by_family'].values()))
        totals[name] = {k:sum(g.get(k, 0) for g in case['by_family'].values()) for k in keys}
        if totals[name]['scenes'] != metrics['planned_windows']:
            raise ValueError('Attribution denominator changed')
    files = [root/name for name in ('manifest.json', 'status.json',
                                    'audit_summary.json', 'resolved_config.yaml')]
    script = Path(__file__).resolve()
    files.append(script)
    config = dict(schema_version='sumodiff.quality.report.config.v1',
                  source_run=str(root), source_sha=metrics['run_sha'],
                  renderer=file_identity(script),
                  interpretation='Post-hoc label controls are not generated performance.',
                  scope='200-window attribution; six fixed denoising replays; no retraining')
    with RunRecorder(output, Path.cwd(), config, [sys.executable, *sys.orig_argv[1:]],
                     {'report':0}, purpose='trajectory_quality_audit_report',
                     data_files=files, checkpoint=manifest['checkpoint']['location'],
                     data_id=manifest['data']['id'], formal=formal) as run:
        os.environ['MPLCONFIGDIR'] = str(Path('artifacts/cache/matplotlib-quality').resolve())
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                             'axes.spines.right':False})
        colors = ['#777777', '#4477aa', *['#cc8844']*4, '#228833', '#777777']
        figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.5))
        for i, name in enumerate(ORDER):
            count = totals[name]['quality_pass']
            axes[0].barh(i, count, color=colors[i])
            axes[0].text(count+2, i, f'{count}/200', va='center', fontsize=9)
        axes[0].set_yticks(range(len(ORDER)), LABELS)
        axes[0].invert_yaxis()
        axes[0].set_xlim(0, 229)
        axes[0].set_xlabel('Scenes passing all independent quality checks')
        axes[0].set_title('Attribution controls: same cohort and thresholds')
        for i, name in enumerate(ORDER[:-1]):
            value = metrics['attribution'][name]['pooled']['interior_jerk']['p99']
            axes[1].scatter(value, i, color=colors[i], s=50)
            axes[1].annotate(f'{value:.3g}', (value, i), xytext=(7, -3),
                             textcoords='offset points', fontsize=9)
        axes[1].set_yticks(range(len(ORDER[:-1])), LABELS[:-1])
        axes[1].set_ylim(6.7, -.7)
        axes[1].set_xscale('log')
        axes[1].set_xlim(.1, 120000)
        axes[1].axvline(15, color='#aa3333', linestyle='--', label='Hard limit 15')
        axes[1].set_xlabel('Pooled interior jerk P99 (m/s^3), log scale')
        axes[1].set_title('Future t0 + 0.5 to 4.0 s')
        axes[1].legend(loc='lower right', fontsize=9)
        for ax in axes:
            ax.grid(axis='x', alpha=.2)
        figure.suptitle('Frozen checkpoint audit: future-label substitutions are diagnostic oracles', fontsize=12)
        figure.text(.5, .015, 'Current-acceleration control has near-zero jerk by construction; it does not establish behavioral realism.',
                    ha='center', fontsize=9)
        figure.tight_layout(rect=(0, .04, 1, .94))
        figure.savefig(run.output/'channel_attribution.png', dpi=160)
        figure.savefig(run.output/'channel_attribution.svg')
        plt.close(figure)
        families = sorted({t['family'] for t in metrics['traces']})
        figure, axes = plt.subplots(1, len(families), figsize=(14, 4.8), squeeze=False)
        names = {'ramp_merge':'Ramp merge', 'three_lane_straight':'Three-lane straight',
                 'unsignalized_intersection':'Unsignalized intersection'}
        for ax, family in zip(axes[0], families):
            for i, trace in enumerate(t for t in metrics['traces'] if t['family']==family):
                values = [s['decoded']['interior_jerk_p99'] for s in trace['steps']]
                ax.semilogy(range(1, len(values)+1), values, '-o', markersize=3,
                            label=trace['window_id'].rsplit('-', 1)[-1])
            ax.axhline(15, color='#aa3333', linestyle='--', label='Hard limit 15')
            ax.set_title(names[family])
            ax.set_xlabel('Denoising update (t=999 to t=0)')
            ax.set_ylabel('Decoded interior jerk P99 (m/s^3)')
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
        figure.suptitle('Six fixed pure-noise replays: exact source outputs, frozen model', fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, .92))
        figure.savefig(run.output/'denoising_trace.png', dpi=160)
        figure.savefig(run.output/'denoising_trace.svg')
        plt.close(figure)
        result = dict(schema_version='sumodiff.quality.report.v1',
                      run_sha=run.manifest['git']['commit_sha'], audit_sha=metrics['run_sha'],
                      planned_windows=metrics['planned_windows'], totals=totals,
                      figures={name:file_identity(run.output/name) for name in (
                          'channel_attribution.png', 'channel_attribution.svg',
                          'denoising_trace.png', 'denoising_trace.svg')},
                      interpretation=config['interpretation'])
        write_json(run.output/'summary.json', result)
        run.write_metrics(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--formal', action='store_true')
    args = parser.parse_args()
    render(args.audit, args.output, args.formal)
