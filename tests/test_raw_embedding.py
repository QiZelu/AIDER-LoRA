"""Raw task inputs, preserved experimental settings and checkpoint isolation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.aider import TaskEmbedding
from models.lora import lora_modules
from engine.audit import digest
from engine.checkpoint import load_model_checkpoint
from engine.preflight import sampling_audit
from utils.initialization import initialize_model
from utils.protocol import make_config,validate_config,validate_reference,CONTROL,CONTRACT,phase_for_epoch,best_eligible


class RawEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config=make_config(root='/tmp/aider_raw_clip_residual_gate_tests')
        cls.model=initialize_model(cls.config,build_backbones=False).eval()
        cls.initial=digest(cls.model.compact_state_dict())

    def test_task_table_is_unmodified_raw_fp32_without_weight_loading(self):
        before=torch.get_rng_state().clone()
        with patch('torch.load',side_effect=AssertionError('Task table must not load Adapter weights')):
            encoder=TaskEmbedding(self.config['task'])
        table=json.loads(Path(self.config['task']['table_path']).read_text())
        expected=torch.tensor(table['raw_projection'],dtype=torch.float32)
        self.assertTrue(torch.equal(encoder.encoded_feature,expected))
        self.assertTrue(torch.equal(encoder.original_feature,expected))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertFalse(encoder.encoded_feature.requires_grad)
        self.assertEqual(encoder.encoded_feature.dtype,torch.float32)
        self.assertTrue(bool((encoder.encoded_feature.norm(dim=1)>15).all()))
        self.assertEqual(torch.count_nonzero(encoder.task_mean),0)
        self.assertNotIn('intent_adapter',self.config['task'])

    def test_all_conditioners_receive_the_exact_raw_rows(self):
        model=self.model;seen={};handles=[]
        ids=torch.tensor([2,0,3,1]);expected=model.dimension_encoder.encoded_feature[ids]
        def capture(name):
            def hook(_module,args):seen.setdefault(name,[]).append(args[0].detach().clone())
            return hook
        for name in ('technical_projector','motion_projector','tv_projector'):
            handles.append(getattr(model,name).affine.register_forward_pre_hook(capture(name)))
        handles.append(model.alpha_gate.register_forward_pre_hook(capture('alpha')))
        for name,module in lora_modules(model):handles.append(module.adapter.gate.register_forward_pre_hook(capture(name)))
        try:
            model.configure_phase('adapter')
            with torch.no_grad(),torch.random.fork_rng(devices=[]):
                torch.manual_seed(100)
                raw={name:torch.randn(4,8,width) for name,width in [('technical',512),('motion',256),('text_video',768)]}
                model.forward_from_raw(raw,ids)
        finally:
            for handle in handles:handle.remove()
            model.configure_phase('dense')
        self.assertEqual(len(seen),16)
        for name,values in seen.items():
            for value in values:
                self.assertTrue(torch.equal(value,expected),name)
                self.assertEqual(value.dtype,torch.float32)

    def test_architecture_embedding_data_and_optimizer_settings_match_joint(self):
        old=yaml.safe_load((CONTROL/'config.yaml').read_text())
        for section in ('model','task','data','loader','loss_protocol','sharing_policy'):
            self.assertEqual(self.config[section],old[section],section)
        old_training=copy.deepcopy(old['training']);new_training=copy.deepcopy(self.config['training'])
        for training in (old_training,new_training):training.pop('output_dir')
        self.assertEqual(old_training,new_training)
        self.assertEqual(self.config['training_strategy'],'two_stage')
        self.assertEqual(self.config['training_phases'],[dict(phase='dense',epochs=[1,5]),dict(phase='adapter',epochs=[6,10])])
        self.assertEqual(self.config['evaluation']['best_epochs'],[6,10])

    def test_configuration_rejects_normalization_adapter_and_gate_changes(self):
        validate_config(self.config)
        for path,value in [(('model','task_embedding_transform'),'unit_clip'),
                           (('task','intent_adapter'),{}),
                           (('model','task_lora','modulation','kind'),'linear_2sigmoid'),
                           (('model','head_policy'),'shared'),(('training','lr'),1e-5)]:
            changed=copy.deepcopy(self.config);target=changed
            for key in path[:-1]:target=target[key]
            target[path[-1]]=value
            with self.assertRaises(ValueError):validate_config(changed)
        with self.assertRaises(ValueError):make_config(variant='joint')

    def test_two_stage_phases_post_stage_best_and_earliest_tie(self):
        from engine.trainer import select_best
        self.assertEqual([phase_for_epoch(self.config,i) for i in range(10)],['dense']*5+['adapter']*5)
        self.assertEqual([best_eligible(self.config,i) for i in range(10)],[False]*5+[True]*5)
        metrics={t:dict(count=115,plcc=.6,srocc=.7,krocc=.5,rmse=.2) for t in self.model.task_order}
        metrics['macro']=dict(plcc=.6,srocc=.7,krocc=.5,rmse=.2)
        with patch('engine.trainer.save_checkpoint') as save:
            for epoch in range(5):
                self.assertEqual(select_best(Path('/tmp'),None,self.config,epoch,metrics,-float('inf')),-float('inf'))
            save.assert_not_called()
            best=select_best(Path('/tmp'),None,self.config,5,metrics,-float('inf'))
            select_best(Path('/tmp'),None,self.config,6,metrics,best)
            save.assert_called_once()

    def test_checkpoint_contract_rejects_previous_models_and_other_runs(self):
        wrong=['aider_aigvqa_shared_film_task_conditioned_shared_lora_raw_clip_residual_gate_r8_v1',
               'aider_aigvqa_shared_film_task_conditioned_shared_lora_clip_intent_residual_gate_r8_v1',
               'aider_aigvqa_shared_film_task_conditioned_shared_lora_raw_clip_r8_v1']
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'wrong.pt'
            for contract in wrong:
                torch.save(dict(model_contract=contract,protocol=self.config['protocol']),path)
                with self.assertRaises(ValueError):load_model_checkpoint(self.model,path,self.config)
            other=copy.deepcopy(self.config['protocol']);other['output_root']='/tmp/different_run'
            torch.save(dict(model_contract=CONTRACT,protocol=other),path)
            with self.assertRaises(ValueError):load_model_checkpoint(self.model,path,self.config)

    def test_initialization_is_reproducible_and_all_heads_remain_private(self):
        model=initialize_model(self.config,build_backbones=False)
        self.assertEqual(digest(model.compact_state_dict()),self.initial)
        self.assertEqual(model.construction_rng_sha256,self.model.construction_rng_sha256)
        self.assertEqual(len(list(model.heads())),4)
        self.assertEqual(len(list(lora_modules(model))),12)
        self.assertFalse(model.initialization_audit['trained_checkpoint_loaded'])
        self.assertFalse(model.initialization_audit['intent_adapter_used'])
        self.assertTrue(model.initialization_audit['shared_transformer_base_equal_parent'])
        for _,module in lora_modules(model):
            self.assertTrue(torch.equal(module.adapter.channel_weights(model.dimension_encoder.encoded_feature),torch.ones(4,8)))

    def test_original_sampling_and_pinned_source_derivation(self):
        before=torch.get_rng_state().clone()
        self.assertEqual(len(sampling_audit()),10)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        validate_reference(check_parent_code=True)


if __name__=='__main__':unittest.main()
