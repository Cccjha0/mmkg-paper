"""Real tiny architectures through the production score-export boundary, no training."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from ml.training.src.data.dataset_spec import DatasetBundle, FeatureBundle, MMKG_GENERAL_V1
from ml.training.src.models.build_model import build_model
from ml.training.src.models.recent_baselines.rotate_utils import rotate_score
from scripts.eval_heterogeneous_complementarity import score_expert_block, validate_pair, ranks_against_reference
from scripts.audit_paper_a_alignment_training import ordered_mapping, check_state_support
from scripts.rebuild_paper_a_base_model import rebuild_config


def bundle():
    torch.manual_seed(17)
    return DatasetBundle(name='toy', protocol_version=MMKG_GENERAL_V1, num_entities=6, num_relations=2,
        entity2id={'z':0,'b':1,'q':2,'a':3,'x':4,'c':5}, relation2id={'zrel':0,'arel':1},
        train_triples=[(0,0,1),(2,1,3)], valid_triples=[(4,0,5)], test_triples=[(5,1,0)],
        features=FeatureBundle(text_features=torch.randn(6,5), image_features=torch.randn(6,4),
            has_text=torch.tensor([1,1,0,1,0,1],dtype=torch.bool), has_img=torch.tensor([1,0,1,1,0,1],dtype=torch.bool)))


def expert(architecture='native', chunk=4, batch=2):
    b = bundle()
    cfg = dict(protocol={'version':MMKG_GENERAL_V1}, dataset={}, embedding={'d':4}, system={'seed':17}, training={},
        model=dict(name='mmkg_'+architecture, num_relations=2, dim=3, rank=2, pca_init=False))
    model, n = build_model(cfg, dataset_bundle=b)
    return SimpleNamespace(model=model.eval(), bundle=b, num_entities=n, seed=17, name=architecture,
                           chunk_size=chunk, query_batch_size=batch)


@pytest.mark.parametrize('architecture', ['mhyper','native','adamf_mat'])
@pytest.mark.parametrize('direction', ['head','tail'])
@torch.inference_mode()
def test_actual_models_export_id_columns_direction_sign_and_gold_independence(architecture, direction):
    e = expert(architecture)
    q = torch.tensor([[0,0,5],[5,1,0],[2,1,3]])
    _, gold, raw = score_expert_block(e,q,direction,{},'cpu',retain_unfiltered=True)
    scorer = e.model.score_head if direction=='head' else e.model.score_tail
    target_col = 0 if direction=='head' else 2
    # Deliberately non-sorted candidate order; compare by entity ID, not loop position.
    permutation = [5,0,3,1,4,2]
    for i, row in enumerate(q):
        triples = row.repeat(6,1); triples[:,target_col] = torch.tensor(permutation)
        expected = scorer(triples)
        torch.testing.assert_close(raw[i,permutation],expected,rtol=2e-6,atol=2e-6)
    assert torch.isfinite(raw).all() and raw.shape == (3,6)
    torch.testing.assert_close(raw[torch.arange(3),q[:,target_col]],gold,rtol=2e-6,atol=2e-6)
    # Chunking and a replacement hidden answer must not reindex/mutate the raw support.
    replacement=q.clone(); replacement[:,target_col]=(replacement[:,target_col]+1)%6
    e.chunk_size=2; e.query_batch_size=1
    _, _, changed=score_expert_block(e,replacement,direction,{},'cpu',retain_unfiltered=True)
    torch.testing.assert_close(changed,raw,rtol=2e-6,atol=2e-6)
    if architecture=='mhyper':
        reverse=q[:,[2,1,0]].clone(); reverse[:,1]+=2
        torch.testing.assert_close(e.model.score_head(q),e.model.score_tail(reverse))
        assert torch.equal(e.model.inverse_relation_ids,torch.tensor([2,3,0,1]))
        full=e.model.inference_all(reverse if direction=='head' else q)
        torch.testing.assert_close(raw,full,rtol=2e-6,atol=2e-6)
    else:
        torch.testing.assert_close(scorer(q),e.model.score(q))


@pytest.mark.parametrize('kind', ['entity','relation','count','split'])
def test_production_pair_validation_rejects_semantically_misaligned_same_sized_inputs(kind):
    a=expert(); b=deepcopy(a)
    if kind=='entity': b.bundle.entity2id['z'],b.bundle.entity2id['c']=5,0
    if kind=='relation': b.bundle.relation2id['zrel'],b.bundle.relation2id['arel']=1,0
    if kind=='count': b.num_entities=5
    if kind=='split': b.bundle.test_triples[:]=[(0,1,5)]
    with pytest.raises(RuntimeError): validate_pair(a,b)


def test_mapping_hash_order_is_id_order_not_alphabetic_or_insertion_order():
    assert ordered_mapping({'z':0,'a':1})==ordered_mapping({'a':1,'z':0})==['z','a']
    assert ordered_mapping({'z':1,'a':0})==['a','z']
    for bad in ({'a':0,'b':0},{'a':0,'b':2},{'a':-1}):
        with pytest.raises(ValueError): ordered_mapping(bad)


@pytest.mark.parametrize('architecture, label',[('mhyper','M-Hyper'),('native','NativE'),('adamf_mat','AdaMF-MAT')])
def test_checkpoint_support_rejects_missing_entities_and_wrong_relation_convention(architecture,label):
    state=expert(architecture).model.state_dict()
    check_state_support(state,6,2,label)
    bad=state.copy(); bad['text_feat']=state['text_feat'][:5]
    with pytest.raises(ValueError): check_state_support(bad,6,2,label)
    bad=state.copy()
    if architecture=='mhyper': bad['inverse_relation_ids']=torch.arange(4)
    else: bad['rel_embeddings.weight']=state['rel_embeddings.weight'][:1]
    with pytest.raises(ValueError): check_state_support(bad,6,2,label)


def test_rotational_score_sign_and_rank_greater_convention():
    h=torch.tensor([[1.,0.],[1.,0.]])
    t=torch.tensor([[1.,0.],[-1.,0.]])
    scores=rotate_score(h,torch.zeros(2,1),t,6.,1.)
    torch.testing.assert_close(scores,torch.tensor([6.,4.]))
    assert ranks_against_reference(scores[None],scores[:1]).item()==1
    assert ranks_against_reference(scores[None],scores[1:]).item()==2


def test_rebuild_preserves_training_and_seed_and_does_not_enable_test():
    cfg=dict(training={'lr':.1},system={'seed':3,'device':'cuda'},evaluation={'run_test':False},
             output={'root_dir':'original','exp_name':'d_m'},_config_paths={'exp':'old'})
    original=deepcopy(cfg)
    result=rebuild_config({'full_config':cfg},'outputs/base_model_rebuilds')
    assert result['training']==cfg['training'] and result['system']==cfg['system']
    assert result['evaluation']['run_test'] is False and cfg==original
    assert result['output']['root_dir']!='original' and '_config_paths' not in result
