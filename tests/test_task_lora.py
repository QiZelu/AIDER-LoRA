"""Behavioral checks for real task dispatch, frozen phases and strict identities."""
import copy
import gc
import json
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.fusion import TransformerFusion, TaskTransformerBlock
from models.lora import TaskLoRALinear, task_adapter_items
from models.registry import TASKS
from engine.audit import digest,optimizer_snapshot
from engine.optim import build_optimizer,_scheduler,transition_to_adapters
from engine.checkpoint import save_checkpoint,load_model_checkpoint
from engine.losses import regression_loss,quality_loss_protocol
from engine.preflight import sampling_audit
from engine.diagnostics import lora_diagnostics
from utils.protocol import OUTPUTS,load_prepared,validate_config,phase_for_epoch,best_eligible,validate_reference
from utils.initialization import initialize_model


def small_fusion():
    return TransformerFusion(42,torch.zeros(1,8,24),width=24,heads=4)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = small_fusion()
        self.readout = nn.Linear(24,1)
        self.fusion.configure_phase('dense')

    def configure_phase(self,phase):
        self.zero_grad(set_to_none=True)
        self.fusion.configure_phase(phase)

    def forward(self,x,ids):
        return self.readout(self.fusion(x,task_ids=ids)[0]).mean()


class LoRABehaviorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_zero_delta_matches_ordinary_pre_ln_transformer(self):
        block = TaskTransformerBlock(42,0,width=24,heads=4).eval()
        mha = nn.MultiheadAttention(24,4,dropout=0,batch_first=True)
        with torch.no_grad():
            mha.in_proj_weight.copy_(torch.cat([getattr(block.attention,n).base.weight for n in ('q_proj','k_proj','v_proj')]))
            mha.in_proj_bias.copy_(torch.cat([getattr(block.attention,n).base.bias for n in ('q_proj','k_proj','v_proj')]))
            mha.out_proj.load_state_dict(block.attention.out_proj.base.state_dict())
        x, ids = torch.randn(8,8,24),torch.tensor([0,2,1,3,2,1,0,3])
        normalized = block.norm1(x)
        u = x + mha(normalized,normalized,normalized,need_weights=False)[0]
        expected = u + block.fc2.base(F.gelu(block.fc1.base(block.norm2(u))))
        torch.testing.assert_close(block(x,ids),expected,atol=1e-6,rtol=1e-6)

    def test_mixed_tasks_match_separate_dispatch_and_keep_row_order(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,alpha=4,dropout=0).eval()
        with torch.no_grad():
            for i,adapter in enumerate(layer.adapters.values()):
                adapter.B.weight.fill_(.1*(i+1))
        x, ids = torch.randn(8,3,6),torch.tensor([3,0,2,0,1,3,1,2])
        expected = torch.stack([layer.base(x[i])+layer.adapters[TASKS[t]](x[i]) for i,t in enumerate(ids.tolist())])
        torch.testing.assert_close(layer(x,ids),expected)
        self.assertFalse(torch.equal(layer(x[:1],torch.tensor([0])),layer(x[:1],torch.tensor([1]))))

    def test_absent_task_adam_and_weight_decay_are_unchanged(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,dropout=0)
        optimizer = torch.optim.AdamW(layer.parameters(),lr=.01,weight_decay=.1)
        x = torch.randn(8,3,6)
        # Give every task an Adam state, then omit three tasks on the next step.
        layer(x,torch.tensor([0,1,2,3,0,1,2,3])).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        old = {task:optimizer_snapshot(adapter,optimizer) for task,adapter in layer.adapters.items()}
        layer(x,torch.zeros(8,dtype=torch.int64)).square().mean().backward()
        optimizer.step()
        for task in TASKS[1:]:
            self.assertEqual(optimizer_snapshot(layer.adapters[task],optimizer),old[task])
            self.assertTrue(all(p.grad is None for p in layer.adapters[task].parameters()))
        self.assertNotEqual(optimizer_snapshot(layer.adapters[TASKS[0]],optimizer),old[TASKS[0]])

    def test_first_step_B_gradient_and_later_A_gradient(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,dropout=0)
        optimizer = torch.optim.AdamW(layer.parameters(),lr=.01)
        x,ids = torch.randn(4,2,6),torch.zeros(4,dtype=torch.int64)
        layer(x,ids).square().mean().backward()
        adapter = layer.adapters['static']
        self.assertEqual(torch.count_nonzero(adapter.A.weight.grad),0)
        self.assertGreater(torch.count_nonzero(adapter.B.weight.grad),0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        layer(x,ids).square().mean().backward()
        self.assertGreater(torch.count_nonzero(adapter.A.weight.grad),0)

    def test_adapter_construction_does_not_consume_base_rng(self):
        torch.manual_seed(42)
        one = TaskLoRALinear(6,5,('static',),42)
        state = torch.get_rng_state().clone()
        torch.manual_seed(42)
        four = TaskLoRALinear(6,5,TASKS,42)
        self.assertTrue(torch.equal(one.base.weight,four.base.weight))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertFalse(torch.equal(four.adapters['static'].A.weight,four.adapters['temporal'].A.weight))
        pointers = [p.data_ptr() for p in four.adapters.parameters()]
        self.assertEqual(len(pointers),len(set(pointers)))

    def test_checkpoint_recomputation_uses_correct_task_and_dropout_rng(self):
        checkpointed = small_fusion().train()
        direct = copy.deepcopy(checkpointed)
        direct.use_checkpoint = False
        for _,_,adapter in task_adapter_items(checkpointed):
            with torch.no_grad():
                adapter.B.weight.normal_(0,.01)
        direct.load_state_dict(checkpointed.state_dict())
        x,ids = torch.randn(8,8,24),torch.tensor([0,3,1,2,3,0,2,1])
        for model in (checkpointed,direct):
            torch.manual_seed(100)
            model(x,task_ids=ids)[0][...,0].mean().backward()
        for (name,left),(_,right) in zip(checkpointed.named_parameters(),direct.named_parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad,right.grad,atol=1e-7,rtol=1e-5,msg=name)
            else:
                self.assertIsNone(right.grad)

    def test_dense_bypass_does_not_consume_lora_dropout_rng(self):
        model = small_fusion().train()
        model.configure_phase('dense')
        x,ids = torch.randn(8,8,24),torch.zeros(8,dtype=torch.int64)
        before = torch.get_rng_state().clone()
        model(x,task_ids=ids)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(all(not p.requires_grad for _,_,a in task_adapter_items(model) for p in a.parameters()))

    def test_invalid_task_ids_are_rejected(self):
        model = small_fusion()
        for ids in (None,torch.zeros(8),torch.full((8,),4,dtype=torch.int64),torch.zeros(7,dtype=torch.int64)):
            with self.assertRaises(ValueError):
                model(torch.randn(8,8,24),task_ids=ids)

    def test_delta_weight_diagnostics_matches_materialized_matrix(self):
        model = small_fusion().eval()
        for _,_,adapter in task_adapter_items(model):
            with torch.no_grad():
                adapter.B.weight.normal_(0,.01)
        report = lora_diagnostics(type('Container',(),{'fusion':model,'named_modules':model.named_modules})())
        name,module = next((n,m) for n,m in model.named_modules() if isinstance(m,TaskLoRALinear))
        matrices = [a.B.weight @ a.A.weight * a.scale for a in module.adapters.values()]
        item = report['projections'][name]
        for i,task in enumerate(TASKS):
            self.assertAlmostEqual(item['tasks'][task]['delta_weight_norm'],float(matrices[i].norm()),places=5)
            for j in range(4):
                expected = float(F.cosine_similarity(matrices[i].flatten(),matrices[j].flatten(),dim=0))
                self.assertAlmostEqual(item['delta_weight_cosines'][i][j],expected,places=5)


class StageTransitionTests(unittest.TestCase):
    def test_freeze_scope_adam_continuity_and_no_lr_restart(self):
        torch.manual_seed(42)
        model = TinyModel()
        config = dict(training=dict(lr=1e-4,weight_decay=.05,module_lrs={},betas=[.9,.999]))
        optimizer = build_optimizer(model,config)
        scheduler = _scheduler(optimizer,2310,.05,warmup_steps=116)
        x,ids = torch.randn(8,8,24),torch.tensor([0,1,2,3,0,1,2,3])
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(x,ids).backward()
            optimizer.step()
            scheduler.step()
        while scheduler.last_epoch < 1155:
            scheduler.step()
        old_base = digest(dict(model.fusion.base_parameters()))
        old_readout_state = {k:v.clone() for k,v in optimizer.state[model.readout.weight].items()}
        old_lr = scheduler.get_last_lr()
        optimizer,scheduler,report = transition_to_adapters(model,optimizer,scheduler,config)
        self.assertEqual(scheduler.last_epoch,1155)
        self.assertEqual(old_lr,scheduler.get_last_lr())
        self.assertEqual(report['lr_before'],report['lr_after'])
        self.assertFalse(report['learning_rate_restarted'])
        self.assertTrue(model.readout.weight.requires_grad)
        for key,value in old_readout_state.items():
            self.assertTrue(torch.equal(optimizer.state[model.readout.weight][key],value))
        self.assertTrue(all(not p.requires_grad and p.grad is None for _,p in model.fusion.base_parameters()))
        self.assertTrue(all(p not in optimizer.state for _,_,a in task_adapter_items(model) for p in a.parameters()))
        model(x,ids).backward()
        optimizer.step()
        scheduler.step()
        self.assertEqual(digest(dict(model.fusion.base_parameters())),old_base)
        self.assertEqual(scheduler.last_epoch,1156)
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for _,_,a in task_adapter_items(model) for p in a.parameters()))
        reference_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))],lr=1e-4)
        reference_scheduler = _scheduler(reference_optimizer,2310,.05,warmup_steps=116)
        for _ in range(1156):
            reference_optimizer.step()
            reference_scheduler.step()
        self.assertTrue(all(lr == reference_scheduler.get_last_lr()[0] for lr in scheduler.get_last_lr()))

    def test_transition_rejects_repeated_activation(self):
        model = TinyModel()
        config = dict(training=dict(lr=1e-4,module_lrs={}))
        optimizer = build_optimizer(model,config)
        scheduler = _scheduler(optimizer,2310,.05,warmup_steps=116)
        optimizer,scheduler,_ = transition_to_adapters(model,optimizer,scheduler,config)
        with self.assertRaises(ValueError):
            transition_to_adapters(model,optimizer,scheduler,config)


class ProtocolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.configs = {v:load_prepared(p) for v,p in OUTPUTS.items()}

    def test_original_sampling_and_no_model_rng_consumption(self):
        self.assertEqual(len(sampling_audit()),10)

    def test_phase_and_best_windows(self):
        joint,two = self.configs['joint'],self.configs['two_stage']
        self.assertEqual([phase_for_epoch(two,i) for i in range(10)],['dense']*5+['adapter']*5)
        self.assertTrue(all(best_eligible(joint,i) for i in range(10)))
        self.assertEqual([best_eligible(two,i) for i in range(10)],[False]*5+[True]*5)

    def test_config_and_reference_are_pinned(self):
        validate_reference(check_parent_code=True)
        for variant,original in self.configs.items():
            validate_config(original)
            for keys,value in ((('model','readout_spec','depth'),12),(('model','task_lora','rank'),16),
                               (('training','moe_mi_weight'),.3),(('training','epochs'),20)):
                changed = copy.deepcopy(original)
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_regression_only_fp32_and_no_mi_protocol(self):
        predictions = torch.tensor([.2,.8],dtype=torch.bfloat16,requires_grad=True)
        targets = torch.tensor([.1,.6])
        loss,_ = regression_loss(predictions,targets)
        self.assertEqual(loss.dtype,torch.float32)
        torch.testing.assert_close(loss,F.smooth_l1_loss(predictions.float(),targets,beta=.1))
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        for config in self.configs.values():
            self.assertIsNone(quality_loss_protocol(config)['auxiliary'])

    def test_public_model_factory_supports_the_new_fusion(self):
        from models import build_aider
        model = build_aider(self.configs['joint'],build_backbones=False)
        self.assertEqual(len(list(task_adapter_items(model))),48)
        self.assertFalse(any('moe' in name for name,_ in model.named_parameters()))
        del model

    def test_best_selection_skips_dense_stage_and_keeps_earlier_ties(self):
        from engine.trainer import select_best
        metrics = {task:dict(count=115,plcc=.6,srocc=.7,krocc=.5,rmse=.2) for task in TASKS}
        metrics['macro'] = dict(plcc=.6,srocc=.7,krocc=.5,rmse=.2)
        with patch('engine.trainer.save_checkpoint') as save:
            best = select_best(Path('/tmp'),None,self.configs['two_stage'],4,metrics,-float('inf'))
            self.assertEqual(best,-float('inf'))
            save.assert_not_called()
            best = select_best(Path('/tmp'),None,self.configs['two_stage'],5,metrics,best)
            self.assertEqual(best,.7)
            save.assert_called_once()
            select_best(Path('/tmp'),None,self.configs['two_stage'],6,metrics,best)
            save.assert_called_once()

    def test_identical_full_initial_state_and_rng_without_backbones(self):
        values = []
        for variant in ('joint','two_stage'):
            model = initialize_model(self.configs[variant],build_backbones=False)
            values.append((digest(model.compact_state_dict()),model.construction_rng_sha256))
            self.assertTrue(model.initialization_audit['retained_modules_equal_parent'])
            self.assertEqual(model.initialization_audit['task_lora_parameters'],1179648)
            del model
            gc.collect()
        self.assertEqual(values[0],values[1])

    def test_checkpoint_identity_phase_and_all_frozen_fusion_weights(self):
        config = self.configs['two_stage']
        model = initialize_model(config,build_backbones=False)
        model.source_hashes = {'unit_test':'fixed'}
        model.configure_phase('adapter')
        raw = {name:torch.randn(4,8,width) for name,width in [('technical',512),('motion',256),('text_video',768)]}
        ids = torch.arange(4)
        model.eval()
        with torch.no_grad():
            expected = model.forward_from_raw(raw,ids)['predictions'].clone()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'model.pt'
            save_checkpoint(checkpoint,model,config,5)
            saved = torch.load(checkpoint,weights_only=False,mmap=True)
            self.assertTrue(all('fusion.'+n in saved['model'] for n,_ in model.fusion.named_parameters()))
            del saved
            with torch.no_grad():
                model.fusion.blocks[0].fc1.base.weight.add_(.1)
                model.fusion.blocks[0].fc1.adapters['static'].B.weight.add_(.1)
            model.configure_phase('dense')
            load_model_checkpoint(model,checkpoint,config)
            self.assertEqual(model.fusion.phase,'adapter')
            with torch.no_grad():
                actual = model.forward_from_raw(raw,ids)['predictions']
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
            with self.assertRaises(ValueError):
                load_model_checkpoint(model,checkpoint,self.configs['joint'])
            model.source_hashes = {'unit_test':'changed'}
            with self.assertRaises(ValueError):
                load_model_checkpoint(model,checkpoint,config)
            model.source_hashes = {'unit_test':'fixed'}
            bad = Path(tmp) / 'old_moe.pt'
            torch.save(dict(model_contract='old_moe',model={}),bad)
            with self.assertRaises(ValueError):
                load_model_checkpoint(model,bad,config)
            with self.assertRaises(ValueError):
                save_checkpoint(Path(tmp)/'wrong_phase.pt',model,config,2)
        del model
        gc.collect()


if __name__ == '__main__':
    unittest.main()
