"""Check local dependencies and numerical inputs, render all PDF pages for QA."""
import csv, hashlib, json, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import pymupdf as fitz
from PIL import Image, ImageOps, ImageDraw

main=(ROOT/'main.tex').read_text(encoding='utf-8')
# Only active inputs belong to the current draft; historical tables remain archived.
tex_files=[ROOT/'main.tex']+[ROOT/rel for rel in re.findall(r'\\input\{([^}]+)\}',main)]
source='\n'.join(p.read_text(encoding='utf-8') for p in tex_files)
main=(ROOT/'main.tex').read_text(encoding='utf-8')
for rel in re.findall(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}',main):
    assert (ROOT/rel).is_file(),rel
labels=re.findall(r'\\label\{([^}]+)\}',source)
assert len(labels)==len(set(labels)), 'Duplicate label'
for key in re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',source): assert key in labels,key
bib=(ROOT/'references.bib').read_text(encoding='utf-8')
keys=set(re.findall(r'@\w+\{([^,]+),',bib))
for item in re.findall(r'\\cite\w*\{([^}]+)\}',main):
    assert set(item.split(','))<=keys,item
manifest=json.loads((ROOT/'rerun_source_manifest.json').read_text(encoding='utf-8'))
assert manifest['version']=='information_boundary_v2'
for rel,sha in manifest['sources'].items():
    assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==sha,rel
for name,sha in manifest['tables'].items():
    assert hashlib.sha256((ROOT/'tables/rerun'/name).read_bytes()).hexdigest()==sha,name
dyna=json.loads((ROOT/'dynasemble_source_manifest.json').read_text(encoding='utf-8'))
assert dyna['version']=='dynasemble_controls_v1_review'
for rel,sha in dyna['sources'].items():
    assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==sha,rel
for rel,sha in dyna['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables={ROOT/'tables/rerun'/name for name in manifest['tables']}|{ROOT/rel for rel in dyna['tables']}
conservative=json.loads((ROOT/'conservative_source_manifest.json').read_text(encoding='utf-8'))
assert conservative['version']=='conservative_radius_review_v1'
for rel,sha in conservative['sources'].items():
    assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==sha,rel
for collection in ('tables','figures'):
    for rel,sha in conservative[collection].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in conservative['tables']}
matched=json.loads((ROOT/'matched_source_manifest.json').read_text(encoding='utf-8'))
assert matched['version']=='matched_alternatives_v1'
for rel,sha in matched['sources'].items():
    assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==sha,rel
for rel,sha in matched['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in matched['tables']}
data_checkpoint=json.loads((ROOT/'data_checkpoint_source_manifest.json').read_text(encoding='utf-8'))
assert data_checkpoint['version']=='data_checkpoint_review_v1'
for rel,sha in data_checkpoint['sources'].items():
    # Stream checkpoints and exports; do not read every large source at once.
    digest=hashlib.sha256()
    with (ROOT.parent/rel).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==sha,rel
for rel,sha in data_checkpoint['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in data_checkpoint['tables']}
data_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/data_checkpoint_review_v1/audit.json').read_text())
assert data_audit['status']=='data_checkpoint_checks_passed' and not data_audit['failures']
assert data_audit['runs_checked']==18 and data_audit['exports_checked']==12
assert data_audit['test_used_for_selection'] is False
feature_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/data_checkpoint_review_v1/feature_provenance_audit.json').read_text())
assert feature_audit['status']=='source_identity_evidence_checks_passed' and not feature_audit['failures']
assert feature_audit['source_files_checked']==6 and feature_audit['sampled_objects_checked']==20
assert feature_audit['encoder_provenance_established'] is False
assert feature_audit['released_feature_download_source_documented'] is True
assert feature_audit['nonempty_root_attribute_maps']==feature_audit['nonempty_sampled_object_attribute_maps']==0
assert feature_audit['test_used_for_selection'] is False
history=json.loads((ROOT/'history_source_manifest.json').read_text(encoding='utf-8'))
assert history['version']=='test_history_review_v1'
for rel,sha in history['sources'].items():
    assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==sha,rel
for rel,sha in history['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in history['tables']}
history_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/test_history_review_v1/audit.json').read_text())
assert history_audit['status']=='history_evidence_checks_passed' and not history_audit['failures']
assert history_audit['timeline_events']==15
assert history_audit['historical_policy_bindings_checked']==4 and history_audit['corrected_policy_bindings_checked']==6
assert history_audit['current_confirmatory_status']=={'MKG-W':False,'DB15K':False}
assert not history_audit['historical_test_influence_on_design_excluded']
assert all(v is None for v in history_audit['first_human_test_inspection_time'].values())
assert history_audit['test_used_for_new_selection'] is False
complement=json.loads((ROOT/'complementarity_source_manifest.json').read_text(encoding='utf-8'))
assert complement['version']=='complementarity_review_v1'
for rel,sha in complement['sources'].items():
    digest=hashlib.sha256()
    with (ROOT.parent/rel).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==sha,rel
for collection in ('tables','figures'):
    for rel,sha in complement[collection].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in complement['tables']}
complement_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/complementarity_review_v1/audit.json').read_text())
assert complement_audit['status']=='complementarity_checks_passed' and not complement_audit['failures']
assert complement_audit['pairs']==6 and complement_audit['split_cells']==12
assert complement_audit['training_runs']==0 and not complement_audit['new_policy_selection']
assert not complement_audit['test_used_for_new_selection'] and not complement_audit['selector_features_changed']
assert all(c['recorded_aggregate_verified'] and all(c['decompositions'].values()) for c in complement_audit['checks'])
claims=json.loads((ROOT/'claims_cost_source_manifest.json').read_text(encoding='utf-8'))
assert claims['version']=='claims_cost_review_v1'
for rel,sha in claims['sources'].items():
    digest=hashlib.sha256()
    with (ROOT.parent/rel).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==sha,rel
for rel,sha in claims['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in claims['tables']}
claims_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/claims_cost_review_v1/audit.json').read_text())
assert claims_audit['status']=='claims_cost_checks_passed' and not claims_audit['failures']
assert claims_audit['primary_pairs']==4 and claims_audit['additional_pairs']==2
assert claims_audit['method_pair_cells']==76 and claims_audit['paired_effects']==70
assert claims_audit['paired_seed_direction_cells']==770 and claims_audit['historical_timing_rows_verified']==240
assert not claims_audit['test_used_for_new_selection'] and claims_audit['training_runs']==0
assert not claims_audit['pair_partition_before_test_established']
assert not claims_audit['current_corrected_timing_available'] and not claims_audit['historical_dyna_is_healthy_r3']
assert all(c['endpoint_and_summary_verified'] and c['complete_seed_direction_coverage'] for c in claims_audit['checks'])

inference=json.loads((ROOT/'inference_contract_source_manifest.json').read_text(encoding='utf-8'))
assert inference['version']=='inference_contract_review_v1'
for rel,sha in inference['sources'].items():
    digest=hashlib.sha256()
    with (ROOT.parent/rel).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==sha,rel
inference_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/inference_contract_review_v1/audit.json').read_text())
assert inference_audit['status']=='guarded_inference_and_anchor_checks_passed'
assert inference_audit['anchor_checks']==36 and inference_audit['observed_rows']==474732 and inference_audit['new_fallback_rows']==0
assert inference_audit['selector_fits']==0 and inference_audit['base_model_scoring_runs']==0
assert not inference_audit['test_used_for_selection'] and not inference_audit['historical_rows_replaced']
assert inference_audit['b10_status']=='anchor_and_shared_endpoint_tests_passed_real_normalized_endpoint_audit_pending'
assert len(inference_audit['checks'])==12 and all(c['adc_weights_exact'] and c['query_soft_weights_exact'] and c['fallback_exact'] and c['grid_endpoints_exact'] for c in inference_audit['checks'])

# The inference manifest above remains the dated pre-return replay; this audit
# supersedes its pending endpoint comparison without rewriting frozen evidence.
endpoint=json.loads((ROOT/'endpoint_contract_source_manifest.json').read_text(encoding='utf-8'))
assert endpoint['version']=='endpoint_contract_return_review_v1'
for rel,sha in endpoint['sources'].items():
    digest=hashlib.sha256()
    with (ROOT.parent/rel).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==sha,rel
for rel,sha in endpoint['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in endpoint['tables']}
endpoint_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/endpoint_contract_return_review_v1/audit.json').read_text())
assert endpoint_audit['status']=='endpoint_return_checks_passed' and endpoint_audit['cells']==72
assert endpoint_audit['returned_files']==146 and endpoint_audit['totals']['rows']==474732
assert endpoint_audit['totals']['normalized_vs_raw_mismatches']==1293 and not endpoint_audit['normalized_endpoint_equivalence']
assert all(endpoint_audit['totals'][k]==0 for k in ('export_vs_shared_mismatches','alpha_one_vs_shared_mismatches','alpha_zero_vs_shared_mismatches','raw_contract_exception_rows'))
assert endpoint_audit['primary_test_effects_unchanged_at_1e_12'] and len(endpoint_audit['paired_effect_checks'])==12
assert endpoint_audit['policy_rows']==144 and endpoint_audit['policy_cell_rows']==864 and len(endpoint_audit['historical_join_checks'])==24
assert endpoint_audit['selector_fits']==0 and endpoint_audit['local_scorer_runs']==0 and not endpoint_audit['test_used_for_selection']
assert not endpoint_audit['historical_results_replaced'] and not endpoint_audit['new_confidence_intervals']
assert [r['label'] for r in endpoint_audit['full_dev_anchor_checks'] if r['changed']]==['W-NA']
endpoint_assets=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/endpoint_contract_return_review_v1/asset_audit.json').read_text())
assert endpoint_assets['status']=='endpoint_assets_verified' and len(endpoint_assets['tables'])==3

winner=json.loads((ROOT/'winner_signal_source_manifest.json').read_text(encoding='utf-8'))
assert winner['version']=='winner_signal_review_v1'
for rel,sha in winner['sources'].items():
    digest=hashlib.sha256()
    with (ROOT.parent/rel).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==sha,rel
for rel,sha in winner['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in winner['tables']}
winner_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/winner_signal_review_v1/audit.json').read_text())
winner_dev=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/winner_signal_review_v1/dev_complete.json').read_text())
assert winner_audit['status']=='winner_signal_checks_passed' and not winner_audit['failures']
assert len(winner_audit['checks'])==12 and sum(c['observations'] for c in winner_audit['checks'])==474732
assert winner_audit['scorer_runs']==0 and winner_audit['small_model_fits']==60
for key in ('new_test_policy','test_used_for_selection','historical_results_replaced','calibrated_harm_probability'):
    assert not winner_audit[key],key
assert winner_dev['status']=='winner_signal_dev_checks_passed' and not winner_dev['test_opened']
assert winner_dev['balanced_reconstructions']==winner_dev['new_unweighted_fits']==30
assert winner_dev['candidate_evaluations']==2460 and winner_dev['final_models']==0
assert len(winner_dev['checks'])==6 and sum(c['observations'] for c in winner_dev['checks'])==219564
winner_assets=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/winner_signal_review_v1/asset_audit.json').read_text())
assert winner_assets['status']=='winner_signal_assets_verified' and winner_assets['tables']==winner['tables']
assert winner_assets['contingency_rows']==36 and winner_assets['probability_rows']==18 and winner_assets['policy_rows']==30
assert r'\label{eq:weighted-preference}' in main and r'\label{eq:winner-direction}' in main
assert 'not automatically the natural probability' in main and 'zero miss rates' in main
assert '5.57\\%' in main and '5.60\\%' in main and '7.66\\%' in main
scope=json.loads((ROOT/'inference_scope_source_manifest.json').read_text(encoding='utf-8'))
assert scope['version']=='inference_training_scope_v1'
for rel,sha in scope['sources'].items():
    digest=hashlib.sha256()
    with (ROOT.parent/rel).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==sha,rel
for rel,sha in scope['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in scope['tables']}
scope_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/inference_training_scope_v1/audit.json').read_text())
assert scope_audit['status']=='inference_training_scope_checks_passed'
assert scope_audit['adc_final_models']==6 and scope_audit['adc_full_geometry_oof_scopes']==30
assert scope_audit['dyna_historical_final_models']==12 and scope_audit['dyna_new_final_models']==180
assert scope_audit['dyna_cv_trajectories']==324
assert scope_audit['scientific_model_fits']==scope_audit['base_model_scoring_runs']==0
for key in ('test_used_for_selection','results_replaced','seed_is_selector_feature',
            'rank_cache_required_for_inference','checkpoint_generalization_tested',
            'training_information_matched_between_adc_and_dyna'):
    assert not scope_audit[key],key
assert all(p['pooled_preprocessing_count_and_moments_verified'] and p['full_selectors']==1 for p in scope_audit['adc_pairs'])
assert [p['fit_rows'] for p in scope_audit['adc_pairs']]==[19567,20564,37005,38925,20522,38515]
scope_assets=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/inference_training_scope_v1/asset_audit.json').read_text())
assert scope_assets['status']=='inference_scope_assets_verified' and scope_assets['method_groups']==10
assert scope_assets['tables']==scope['tables']
policy_algorithm=main.split(r'\begin{algorithm}',1)[1].split(r'\end{algorithm}',1)[0]
assert 'No gold or rank cache' in policy_algorithm and 'Return the weight and candidate scores' in policy_algorithm
assert 'return the corresponding standalone rank' not in policy_algorithm
assert 'one shared selector' in main and 'not three independently fitted ADC selectors' in main
assert r'\ref{tab:inference-method-scope}' in main and r'\label{app:inference-cache}' in main
grid=json.loads((ROOT/'grid_sensitivity_source_manifest.json').read_text(encoding='utf-8'))
assert grid['version']=='grid_sensitivity_return_review_v1'
for rel,sha in grid['sources'].items():
    digest=hashlib.sha256()
    with (ROOT.parent/rel).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==sha,rel
for rel,sha in grid['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in grid['tables']}
grid_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/grid_sensitivity_return_review_v1/audit.json').read_text())
assert grid_audit['status']=='grid_sensitivity_return_checks_passed' and not grid_audit['failures']
assert grid_audit['cells']==48 and grid_audit['observations']==316488 and grid_audit['original_grid_rank_mismatches']==0
assert not grid_audit['new_grid_selection'] and not grid_audit['historical_results_replaced'] and not grid_audit['new_confidence_intervals']
assert grid_audit['local_scorer_runs']==0 and len(grid_audit['checks'])==8
grid_assets=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/grid_sensitivity_return_review_v1/asset_audit.json').read_text())
assert grid_assets['status']=='grid_sensitivity_assets_verified' and grid_assets['original_paper_grid']==.05
assert grid_assets['dev_test_summary_rows']==56 and grid_assets['paired_effect_rows']==104 and grid_assets['seed_direction_rows']==336
dimension=json.loads((ROOT/'feature_dimension_source_manifest.json').read_text(encoding='utf-8'))
assert dimension['version']=='feature_dimension_review_v1'
for rel,sha in dimension['sources'].items():
    digest=hashlib.sha256()
    with (ROOT.parent/rel).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==sha,rel
for rel,sha in dimension['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in dimension['tables']}
dimension_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/feature_dimension_review_v1/test_audit.json').read_text())
assert dimension_audit['status']=='feature_dimension_checks_passed' and not dimension_audit['failures']
assert not dimension_audit['test_used_for_selection'] and len(dimension_audit['checks'])==4
dimension_assets=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/feature_dimension_review_v1/asset_audit.json').read_text())
assert dimension_assets['dev_candidate_evaluations']==1968 and dimension_assets['incomplete_feature_rows']==0
# The earlier feature-only audit remains insufficient; the separate returned
# raw-score audit below establishes a narrower, explicitly scoped condition.
assert not dimension_assets['raw_score_finiteness_certified']
raw_score=json.loads((ROOT/'raw_score_source_manifest.json').read_text(encoding='utf-8'))
assert raw_score['version']=='raw_score_contract_review_v1'
for rel,sha in raw_score['sources'].items():
    assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==sha,rel
sys.path.insert(0,str(ROOT.parent))
from scripts.review_paper_a_raw_score_contract import verify as verify_raw_score_return
raw_checked=verify_raw_score_return(ROOT.parent)
raw_review=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/raw_score_contract_review.json').read_text())
for key,value in raw_checked.items():
    if key!='sources': assert raw_review[key]==value,key
assert raw_review['scope']=='frozen_checkpoint_reexecution_dev_test'
assert raw_review['raw_score_finiteness_certified'] and not raw_review['historical_raw_bitwise_equality_established']
assert not raw_review['abnormal_input_robustness_established']
assert raw_review['score_rows']==474732 and raw_review['cells']==72 and raw_review['checkpoints']==18
assert '474,732 expert/query scoring rows' in main and 'audit is prepared but remains pending' not in main
action_semantics=json.loads((ROOT/'action_semantics_source_manifest.json').read_text(encoding='utf-8'))
assert action_semantics['version']=='action_semantics_review_v1'
for rel,sha in action_semantics['sources'].items():
    digest=hashlib.sha256()
    with (ROOT.parent/rel).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==sha,rel
for collection in ('tables','figures'):
    for rel,sha in action_semantics[collection].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in action_semantics['tables']}
coupling=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/action_semantics_review_v1/audit.json').read_text())
assert coupling['status']=='action_semantics_checks_passed' and len(coupling['checks'])==8
assert coupling['base_model_runs']==coupling['selector_fits']==0
assert not coupling['new_policy_selection'] and not coupling['test_used_for_selection']
assert not coupling['historical_metrics_replaced']
assert sum(c['rows'] for c in coupling['checks'])==316488
assert all(c['confidence_and_full_weights_replayed'] and c['identity_max_error']<1e-12 for c in coupling['checks'])
da=[c for c in coupling['checks'] if c['label']=='D-A']
assert len(da)==2 and all(abs(c['observed_min_nonzero_grid_displacement']-.30)<1e-12 for c in da)
assert r'\label{eq:confidence-displacement}' in main and '0.27523' in main
assert not re.search(r'no[ -]local[ -]bound|no bound',source,re.IGNORECASE)
for rel in action_semantics['tables']:
    content=(ROOT/rel).read_text(encoding='utf-8')
    assert content.count('& Expanded-radius &')==4 and 'fixes $\\beta=1$' in content
    assert r'& $\beta=1$ &' not in content
with fitz.open(ROOT/'figures/action_semantics/test_utility_loss.pdf') as figure:
    assert 'Expanded-radius' in '\n'.join(p.get_text() for p in figure)
# Summary sections must retain the adverse evidence as well as the main gains.
abstract=main.split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0]
conclusion=main.split(r'\section{Conclusion}',1)[1].split(r'\FloatBarrier',1)[0]
for section in (abstract,conclusion):
    assert 'four primary' in section and 'two additional' in section
    assert re.search(r'span(?:ning)? zero',section) and 'Relation' in section and 'W-N' in section
    assert 'slightly negative' in section and 'timing' in section
# The abstract uses rounded absolute MRR percentage points; full precision stays
# in the results. Check the unit conversion against bound effects.
with (ROOT.parent/'outputs/paper_a_safe_correction/claims_cost_review_v1/paired_effects.csv').open(encoding='utf-8') as handle:
    effects=list(csv.DictReader(handle))
primary_global=[r for r in effects if r['comparator']=='Global' and r['label'] in ('W-N','W-A','D-N','D-A')]
positive=[float(r['delta_adc_minus_comparator']) for r in primary_global if float(r['ci_low'])>0]
assert len(primary_global)==4 and len(positive)==3
headline=f'{100*min(positive):.2f}--{100*max(positive):.2f} MRR percentage points'
assert headline in abstract and 'three paired base seeds' in abstract
assert 'both additional-pair intervals span zero' in abstract
assert 'lower unconditional mean reciprocal-rank loss relative to Global' in abstract
with (ROOT.parent/'outputs/paper_a_safe_correction/claims_cost_review_v1/historical_cost.csv').open(encoding='utf-8') as handle:
    ratios=[float(r['adc_over_primary']) for r in csv.DictReader(handle)]
assert f'{min(ratios):.1f}--{max(ratios):.1f} times primary-only' in abstract
assert 'historical pre-repair' in abstract and 'corrected timing is unavailable' in abstract
# The conclusion states conditional use rather than repeating the result table;
# retain adverse evidence and an explicit link to all six paired intervals.
assert r'\ref{tab:claims-intervals}' in conclusion
assert 'both additional-pair intervals span zero' in conclusion
assert 'historical pre-repair timing' in conclusion and 'corrected end-to-end timing remains unmeasured' in conclusion
assert not re.search(r'\d+\.\d{6}',abstract)
assert 'requires a query-dependent policy' not in abstract.lower()
assert ROOT/'tables/claims_cost/main.tex' in tex_files
matched_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/matched_alternatives_v1/test_audit.json').read_text())
assert matched_audit['status']=='matched_alternative_checks_passed' and not matched_audit['failures']
assert matched_audit['test_used_for_selection'] is False
assert len(matched_audit['checks'])==4
assert set(tex_files[1:])<=bound_tables
radius_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/conservative_radius_review/audit.json').read_text())
assert radius_audit['status']=='conservative_radius_checks_passed' and not radius_audit['failures']
assert radius_audit['test_used_for_new_selection'] is False
assert len(radius_audit['pair_checks'])==4
assert all(c['five_fold_actions_rr_fallback_reproduced'] for c in radius_audit['pair_checks'])
dyna_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/dynasemble_review_audit/audit.json').read_text())
assert dyna_audit['status']=='review_artifact_checks_passed' and not dyna_audit['failures']
assert dyna_audit['small_cache_evidence_verified'], 'D01-D03 closure requires verified cache identities and selector replay'
audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/information_boundary_rerun_audit/audit.json').read_text())
assert audit['status']=='artifact_checks_passed_with_continuous_repeatability_caveat'
assert not audit['failures']
build=ROOT/'.build/qa'; build.mkdir(exist_ok=True,parents=True)
doc=fitz.open(ROOT/'main.pdf')
extracted='\n'.join(p.get_text() for p in doc)
assert '??' not in extracted,'Unresolved reference in PDF'
assert not re.search(r'\b(?:TODO|TBD|PLACEHOLDER)\b',extracted),'Placeholder found'
page_info=[]
for i,page in enumerate(doc):
    pix=page.get_pixmap(matrix=fitz.Matrix(1.1,1.1),alpha=False)
    pix.save(build/f'page_{i+1:02d}.png')
    page_info.append({'page':i+1,'characters':len(page.get_text()),'images':len(page.get_images())})
    for block in page.get_text('dict')['blocks']:
        if block['type']!=0: continue
        x0,y0,x1,y1=block['bbox']
        assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1,(i+1,block['bbox'])
for start in range(0,len(doc),6):
    sheet=Image.new('RGB',(3*398,2*588),'#d9dde0')
    draw=ImageDraw.Draw(sheet)
    for offset,i in enumerate(range(start,min(start+6,len(doc)))):
        im=Image.open(build/f'page_{i+1:02d}.png'); im.thumbnail((378,548))
        x=(offset%3)*398+10; y=(offset//3)*588+26
        sheet.paste(im,(x,y)); draw.text((x,y-20),f'Page {i+1}',fill='black')
    sheet.save(build/f'contact_{start//6+1}.png')
report={'pages':len(doc),'bibliography_entries':len(keys),'tables':len(re.findall(r'\\begin\{table\}',source)),
        'figures':len(re.findall(r'\\begin\{figure\}',source)),'source_snapshots':len(set(manifest['sources'])|set(dyna['sources'])|set(conservative['sources'])|set(matched['sources'])|set(data_checkpoint['sources'])|set(history['sources'])|set(complement['sources'])|set(claims['sources'])|set(dimension['sources'])|set(raw_score['sources'])|set(action_semantics['sources'])|set(grid['sources'])|set(inference['sources'])|set(endpoint['sources'])|set(scope['sources'])|set(winner['sources'])),
        'result_version':'information_boundary_v2 + dynasemble_controls_v1_review + conservative_radius_review_v1 + matched_alternatives_v1 + data_checkpoint_review_v1 + test_history_review_v1 + complementarity_review_v1 + claims_cost_review_v1 + feature_dimension_review_v1 + raw_score_contract_review_v1 + action_semantics_review_v1 + grid_sensitivity_return_review_v1 + inference_contract_review_v1 + endpoint_contract_return_review_v1 + inference_training_scope_v1 + winner_signal_review_v1',
        'small_cache_evidence_verified':dyna_audit['small_cache_evidence_verified'],
        'conservative_radius_checks_passed':True,'test_used_for_new_selection':False,
        'matched_alternative_checks_passed':True,
        'data_checkpoint_checks_passed':True,
        'source_feature_return_verified':True,'encoder_provenance_established':False,
        'released_feature_download_source_documented':True,
        'test_history_evidence_checks_passed':True,'current_confirmatory_status':history_audit['current_confirmatory_status'],
        'complementarity_checks_passed':True,'observable_modality_support_only':True,
        'claims_cost_checks_passed':True,'primary_plus_additional_pairs':[4,2],
        'feature_dimension_checks_passed':True,'raw_score_return_checks_passed':True,
        'action_semantics_checks_passed':True,'expanded_radius_definition_consistent':True,
        'grid_sensitivity_return_checks_passed':True,'grid_sensitivity_paired_effects':104,
        'grid_selection_after_sensitivity':False,'original_grid_rank_mismatches':0,
        'guarded_inference_replay_passed':True,'guarded_inference_rows':474732,'anchor_checks':36,
        'real_normalized_endpoint_equivalence_established':False,
        'endpoint_return_checks_passed':True,'normalized_endpoint_difference_rows':1293,
        'primary_fixed_policy_endpoint_effects_unchanged':True,'whole_workflow_endpoint_invariance_established':False,
        'inference_training_scope_checks_passed':True,'rank_cache_required_for_inference':False,
        'adc_final_selectors':6,'dyna_historical_final_selectors':12,'dyna_new_final_selectors':180,
        'adc_dyna_fitting_information_matched':False,'unseen_checkpoint_transfer_evaluated':False,
        'winner_signal_checks_passed':True,'winner_direction_diagnostic_rows':474732,
        'class_weight_control_small_fits':60,'class_weight_control_new_test_policy':False,
        'natural_winner_probability_or_harm_calibration_established':False,
        'raw_score_finiteness_certified':raw_review['raw_score_finiteness_certified'],
        'raw_score_finiteness_scope':raw_review['scope'],
        'historical_raw_bitwise_equality_established':False,'abnormal_input_robustness_established':False,
        'paired_comparisons_reported':70,'current_corrected_timing_available':False,
        'checks':'active input paths, references, rerun source/table hashes, audit status, no placeholders, page bounds: PASS',
        'page_details':page_info}
(ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
(build/'extracted_text.txt').write_text(extracted,encoding='utf-8')
print(json.dumps(report,indent=2))
