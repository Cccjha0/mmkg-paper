"""Verify the returned B01 audit against existing ledgers without model scoring."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = 'outputs/paper_a_safe_correction/data_checkpoint_review_v1'
RETURNED = 'outputs/paper_a_safe_correction/raw_score_contract_audit.json'
REVIEW = 'outputs/paper_a_safe_correction/raw_score_contract_review.json'
MANIFEST = 'paper_a_draft/raw_score_source_manifest.json'
CODE = ('scripts/audit_paper_a_raw_score_contract.py',
        'router/score_combination.py', 'scripts/eval_heterogeneous_complementarity.py')
COUNTERS = ('nonfinite_candidate_values', 'partial_nonfinite_rows', 'empty_finite_rows',
            'constant_finite_rows', 'nonfinite_gold_rows', 'nonfinite_moment_rows',
            'invalid_normalized_finite_values', 'nonfinite_normalized_gold_rows',
            'exceptional_rows')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def match_source_bytes(data, reported, allow_newlines=False):
    """Accept exact bytes, or an explicitly recorded whole-file newline conversion."""
    if sha(data) == reported:
        return 'exact_bytes'
    if allow_newlines:
        lf = data.replace(b'\r\n', b'\n')
        for name, candidate in [('LF', lf), ('CRLF', lf.replace(b'\n', b'\r\n'))]:
            if sha(candidate) == reported:
                return 'newline_only_to_' + name
    raise ValueError('Source bytes differ beyond allowed newline conversion')


def verify(root=ROOT, report=None):
    sources = {}

    def read_bytes(rel):
        data = (root / rel).read_bytes()
        sources[rel] = sha(data)
        return data

    baseline = json.loads(read_bytes('paper_a_draft/data_checkpoint_source_manifest.json'))['sources']

    def bound_bytes(rel):
        data = read_bytes(rel)
        require(sources[rel] == baseline.get(rel), 'Changed baseline input: ' + rel)
        return data

    original = json.loads(read_bytes(RETURNED))
    report = original if report is None else report
    runs = list(csv.DictReader(io.StringIO(bound_bytes(DATA + '/checkpoint_diagnostics.csv').decode())))
    ledger = list(csv.DictReader(io.StringIO(bound_bytes(DATA + '/split_counts.csv').decode())))
    require(len(runs) == 18 and len({r['run'] for r in runs}) == 18, 'Incomplete checkpoint ledger')
    expected_run_roles = {(d, m, s) for d in ('mkg_w', 'db15k')
                          for m in ('M-Hyper', 'NativE', 'AdaMF-MAT') for s in (1, 2, 3)}
    require({(r['dataset'], r['model'], int(r['seed'])) for r in runs} == expected_run_roles,
            'Unexpected checkpoint roles')
    split_counts = {(r['dataset'], r['split']): int(r['triples'])
                    for r in ledger if r['stage'] == 'canonical'}
    reported_sources = report['sources']
    expected_sources = {DATA + '/checkpoint_diagnostics.csv', DATA + '/split_counts.csv', *CODE}
    code_checks = []
    for rel in expected_sources:
        local = read_bytes(rel)
        mode = match_source_bytes(local, reported_sources[rel], allow_newlines=rel in CODE[1:])
        code_checks.append({'path': rel, 'match': mode, 'local_sha256': sha(local),
                            'server_sha256': reported_sources[rel]})
    for run in runs:
        checkpoint = run['run'] + '/best.ckpt'
        config = run['run'] + '/config_merged.json'
        expected_sources.update((checkpoint, config))
        require(reported_sources.get(checkpoint) == run['checkpoint_sha256'] == baseline.get(checkpoint),
                'Checkpoint identity mismatch: ' + checkpoint)
        config_bytes = bound_bytes(config)
        require(reported_sources.get(config) == sha(config_bytes), 'Config identity mismatch: ' + config)
    require(set(reported_sources) == expected_sources, 'Unexpected or missing server source paths')

    orders = {}
    for dataset in ('mkg_w', 'db15k'):
        rows = list(csv.DictReader(io.StringIO(gzip.decompress(
            bound_bytes(DATA + '/' + dataset + '_split_rows.csv.gz')).decode())))
        for split, canonical in (('dev', 'valid'), ('test', 'test')):
            selected = sorted((r for r in rows if r['canonical_split'] == canonical),
                              key=lambda r: int(r['canonical_row_index']))
            n = split_counts[dataset, canonical]
            require([int(r['canonical_row_index']) for r in selected] == list(range(n)),
                    'Canonical row coverage mismatch')
            triples = [[int(r[k]) for k in ('head', 'relation', 'tail')] for r in selected]
            orders[dataset, split] = (n, sha(json.dumps(triples, separators=(',', ':')).encode()))

    expected_cells = {(r['run'], split, direction): r for r in runs
                      for split in ('dev', 'test') for direction in ('head', 'tail')}
    cells = report['cells']
    keys = [(c['run'], c['split'], c['direction']) for c in cells]
    require(len(cells) == report['expected_cells'] == 72 and set(keys) == set(expected_cells),
            'Incomplete or duplicate audit cells')
    summaries = {}
    for cell, key in zip(cells, keys):
        run = expected_cells[key]
        require((cell['dataset'], cell['model'], cell['seed']) ==
                (run['dataset'], run['model'], int(run['seed'])), 'Cell role mismatch')
        n, order = orders[cell['dataset'], cell['split']]
        require(type(cell['rows']) is int and cell['rows'] == n, 'Cell row count mismatch')
        require(cell['triple_order_sha256'] == order, 'Triple order mismatch')
        for name in COUNTERS:
            require(type(cell[name]) is int and cell[name] == 0, 'Nonzero or invalid counter: ' + name)
        require(cell['examples'] == [], 'Unexpected exceptional examples')
        summary_key = (cell['dataset'], cell['split'])
        summary = summaries.setdefault(summary_key, dict(dataset=cell['dataset'], split=cell['split'],
                                                       cells=0, score_rows=0, triples=n,
                                                       triple_order_sha256=order))
        summary['cells'] += 1
        summary['score_rows'] += n
    require(report['status'] == 'raw_score_contract_passed' and report['exceptional_rows'] == 0,
            'Audit did not pass')
    require(report['training_runs'] == 0 and report['policy_selection'] is False
            and report['historical_rows_replaced'] is False, 'Unexpected training or result replacement')
    return dict(status='raw_score_return_checks_passed', sources=sources, checkpoints=18, cells=72,
                score_rows=sum(c['rows'] for c in cells), totals={k: sum(c[k] for c in cells) for k in COUNTERS},
                split_checks=list(summaries.values()), source_byte_checks=sorted(code_checks, key=lambda c: c['path']),
                checkpoint_identity='Returned hashes match both the bound checkpoint ledger and baseline source manifest; no local model loading.',
                raw_score_finiteness_certified=True, scope='frozen_checkpoint_reexecution_dev_test',
                historical_raw_bitwise_equality_established=False, abnormal_input_robustness_established=False,
                training_runs=0, test_used_for_selection=False, historical_rows_replaced=False,
                b01_status='closed_for_evaluated_checkpoints_and_splits',
                decision='No numerical-anomaly-driven repair or reevaluation is indicated by this audit.')


def main():
    review = verify()
    review['sources']['scripts/review_paper_a_raw_score_contract.py'] = sha(Path(__file__).read_bytes())
    review['sources']['tests/test_raw_score_contract_return.py'] = sha((ROOT / 'tests/test_raw_score_contract_return.py').read_bytes())
    (ROOT / REVIEW).write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8')
    sources = {**review['sources'], REVIEW: sha((ROOT / REVIEW).read_bytes())}
    (ROOT / MANIFEST).write_text(json.dumps(dict(version='raw_score_contract_review_v1', sources=sources),
                                          indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in review.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
