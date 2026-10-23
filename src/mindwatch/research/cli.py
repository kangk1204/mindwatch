from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from mindwatch.arguments import PrivateArgumentParser


def main(argv=None):
    os.umask(0o077)
    p = PrivateArgumentParser(description='Confidential local multimodal research. Safe aggregate output only.')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--original', type=Path, default=Path('../02_original'))
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--config', type=Path, default=Path('configs/analysis.yaml'))
    q = sub.add_parser('develop')
    q.add_argument('--dataset', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--config', type=Path, default=Path('configs/analysis.yaml'))
    q.add_argument('--features', default='sensor', choices=['demo', 'sensor', 'sensor+ema', 'sensor+ema+history', 'sensor+audio', 'sensor+audio+text', 'sensor+audio+text+ema', 'sensor+audio+text+ema+history'])
    q.add_argument('--target', default='y_phq9', choices=['y_phq9', 'y_gad7', 'y_isi'])
    q.add_argument('--window', type=int, default=14, choices=[7, 14, 28])
    q = sub.add_parser('inventory-speech')
    q.add_argument('--original', type=Path, default=Path('../02_original'))
    q.add_argument('--dataset', type=Path, required=True)
    q = sub.add_parser('develop-sequences')
    q.add_argument('--dataset', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--config', type=Path, default=Path('configs/sequences.yaml'))
    q.add_argument('--window', type=int, default=14, choices=[7,14,28])
    q.add_argument('--target', default='y_phq9', choices=['y_phq9','y_gad7','y_isi'])
    q = sub.add_parser('speech')
    q.add_argument('args', nargs=argparse.REMAINDER)
    q = sub.add_parser('changes')
    q.add_argument('--dataset', type=Path, required=True)
    q = sub.add_parser('controls')
    q.add_argument('--dataset', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('followups')
    q.add_argument('--dataset', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('predict')
    q.add_argument('--input', type=Path, required=True)
    q.add_argument('--model-dir', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('pool-speech')
    q.add_argument('--dataset', type=Path, required=True)
    q.add_argument('--recordings', type=Path, required=True)
    q.add_argument('--window', type=int, choices=[7,14,28], default=14)
    q = sub.add_parser('finalize')
    q.add_argument('--dataset', type=Path, required=True)
    q.add_argument('--model-dir', type=Path, required=True)
    q.add_argument('--features', default=None)
    q.add_argument('--target', default=None)
    q.add_argument('--window', type=int, default=None)
    for name in ['develop-stacking','develop-fusion']:
        q=sub.add_parser(name)
        q.add_argument('--dataset',type=Path,required=True)
        q.add_argument('--output',type=Path,required=True)
        q.add_argument('--config',type=Path,default=Path('configs/sequences.yaml' if name=='develop-fusion' else 'configs/stacking_matched.yaml'))
        q.add_argument('--window',type=int,choices=[7,14,28],default=14)
        if name=='develop-stacking':
            q.add_argument('--features',choices=['sensor+audio','sensor+audio+text','sensor+audio+text+ema','sensor+audio+text+ema+history'],default='sensor+audio+text')
        else:
            q.add_argument('--task',choices=['free_speech','fixed_reading'],default='free_speech')
    for name in ['finalize-stacking','finalize-fusion']:
        q=sub.add_parser(name)
        q.add_argument('--dataset',type=Path,required=True)
        q.add_argument('--model-dir',type=Path,required=True)
    a = p.parse_args(argv)
    try:
        from . import pipeline
        if a.command == 'prepare':
            result = pipeline.prepare(a.original, a.output, a.config)
        elif a.command == 'develop':
            result = pipeline.run_development(a.dataset, a.output, a.config, a.features, a.target, a.window)
        elif a.command == 'inventory-speech':
            result = pipeline.inventory_speech(a.original, a.dataset)
        elif a.command == 'develop-sequences':
            from .sequence_run import develop_sequences
            result = develop_sequences(a.dataset, a.output, a.config, a.window, a.target)
        elif a.command == 'speech':
            from .speech import main as speech_main
            return speech_main(a.args)
        elif a.command == 'pool-speech':
            result = pipeline.pool_speech(a.dataset, a.recordings, a.window)
        elif a.command == 'changes':
            import pandas as pd
            from .features import personal_change
            from .privacy import write_parquet
            if (a.dataset / 'personal_changes.parquet').exists():
                raise FileExistsError('Change signals exist; preserve them and use a new dataset version.')
            changes = personal_change(pd.read_parquet(a.dataset / 'daily.parquet'))
            write_parquet(a.dataset / 'personal_changes.parquet', changes)
            result = {'status': 'exploratory_change_signals_computed', 'rows': len(changes), 'clinical_validation': False}
        elif a.command == 'controls':
            from .controls import run_controls
            result = run_controls(a.dataset, a.output)
        elif a.command == 'followups':
            from .followups import run_followups
            result = run_followups(a.dataset, a.output)
        elif a.command == 'predict':
            import pandas as pd
            from .inference import predict_research
            from .privacy import write_parquet
            prediction = predict_research(pd.read_parquet(a.input), a.model_dir)
            write_parquet(a.output, prediction)
            result = {'status':'research_predictions_saved', 'rows':len(prediction),
                      'abstained':int(prediction.insufficient_data.sum()), 'clinical_validation':False}
        elif a.command == 'develop-stacking':
            from .multimodal_run import develop_stacking
            result=develop_stacking(a.dataset,a.output,a.config,a.features,a.window)
        elif a.command == 'develop-fusion':
            from .multimodal_run import develop_fusion
            result=develop_fusion(a.dataset,a.output,a.config,a.window,a.task)
        elif a.command in {'finalize-stacking','finalize-fusion'}:
            from .multimodal_run import finalize_multimodal
            result=finalize_multimodal(a.dataset,a.model_dir,a.command.split('-')[1])
        else:
            from .benchmarks import finalize
            selection_path = a.model_dir / 'selection.json'
            if not selection_path.is_file():
                raise ValueError('persisted_development_selection_required_before_loading_test')
            selection = json.loads(selection_path.read_text())
            if selection.get('sealed_selection') is not True:
                raise ValueError('complete_frozen_development_selection_required')
            target = selection['target_column']
            if a.target is not None and a.target != target:
                raise ValueError('finalize_target_must_match_frozen_selection')
            selected_features = selection['feature_columns']
            is_sequence = any(name.startswith('sensor__seq_') for name in selected_features)
            if is_sequence:
                from .sequence_run import sequence_frame
                window = selection['config']['sequence_length']
                if a.window is not None and a.window != window:
                    raise ValueError('finalize_window_must_match_frozen_sequence_selection')
                frame, _, _ = sequence_frame(a.dataset, window, target, phase='finalize')
            else:
                feature_set = 'demo' if all(name.startswith('demo__') for name in selected_features) else 'sensor'
                for modality in ['audio', 'text', 'ema', 'history']:
                    if any(name.startswith(modality + '__') for name in selected_features):
                        feature_set += '+' + modality
                if a.features is not None and a.features != feature_set:
                    raise ValueError('finalize_features_must_match_frozen_selection')
                frame, _ = pipeline.load_model_frame(a.dataset, a.window or 14, target, feature_set, phase='finalize')
            result = finalize(frame, a.model_dir)
        # Return only safe summary metadata. Worker may return artifact paths or scalar scores.
        from .privacy import clean_json
        if isinstance(result, dict):
            safe = {k: v for k, v in result.items() if k not in {'predictions', 'frame', 'records', 'subjects', 'raw_paths'}}
        else:
            safe = {'status': 'completed', 'result_type': type(result).__name__}
        print(json.dumps(clean_json(safe), ensure_ascii=False, allow_nan=False, default=str))
        return 0
    except Exception as e:
        # Exception messages may contain patient paths/values; retain only exception type.
        print(json.dumps({'status': 'failed', 'error_type': type(e).__name__}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
