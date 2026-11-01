"""Exercise real phase switches, numeric Adam preservation and complete checkpoints."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.audit import digest,optimizer_snapshot
from engine.checkpoint import save_checkpoint,load_model_checkpoint,validate_phase
from engine.optim import build_optimizer,_scheduler
from engine.preflight import check_private_activity
from engine.stages import switch_to_adapters,lora_state_digest
from models.fusion import TransformerFusion
from models.lora import adapter_items
from models.registry import TASKS
from utils.initialization import initialize_model
from utils.protocol import make_config,validate_config,CONTROL


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion=TransformerFusion(42,torch.zeros(1,8,24),width=24,heads=4)
        self.fusion.heads=nn.ModuleDict({task:nn.Linear(24,1) for task in TASKS})
        self.technical_projector=nn.Linear(24,24)
        self.private_heads=True
        self.task_order=TASKS
        self.configure_phase('dense')

    def configure_phase(self,phase):
        self.zero_grad(set_to_none=True)
        self.fusion.configure_phase(phase)

    def heads(self):return self.fusion.heads.items()

    def forward(self,x,z,ids):
        value=self.fusion(self.technical_projector(x),z,ids)[0]
        outputs=[self.fusion.heads[TASKS[int(ids[i])]](value[i]).mean() for i in range(len(ids))]
        return torch.stack(outputs)


def configuration():
    return dict(training=dict(lr=1e-4,weight_decay=.05,module_lrs={},betas=[.9,.999]))


class StageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        self.model=TinyModel()
        self.config=configuration()
        self.x,self.z,self.ids=torch.randn(8,8,24),torch.randn(8,768),torch.tensor([0,1,2,3,0,1,2,3])

    def step(self,optimizer,scheduler):
        optimizer.zero_grad(set_to_none=True)
        (self.model(self.x,self.z,self.ids)-.5).square().mean().backward()
        optimizer.step();scheduler.step()

    def test_dense_updates_base_and_other_modules_without_lora_adam_or_dropout_rng(self):
        opt=build_optimizer(self.model,self.config);sched=_scheduler(opt,2310,.05,116)
        adapter_before=lora_state_digest(self.model)
        base_before=digest(dict(self.model.fusion.base_parameters()))
        for _ in range(3):self.step(opt,sched)
        self.assertEqual(lora_state_digest(self.model),adapter_before)
        self.assertNotEqual(digest(dict(self.model.fusion.base_parameters())),base_before)
        self.assertTrue(all(not p.requires_grad and p.grad is None and p not in opt.state
                            for _,a in adapter_items(self.model) for p in a.parameters()))
        before=torch.get_rng_state().clone()
        self.model(self.x,self.z,self.ids)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_switch_at_1155_preserves_numeric_adam_and_uninterrupted_lr(self):
        opt=build_optimizer(self.model,self.config);sched=_scheduler(opt,2310,.05,116)
        for _ in range(3):self.step(opt,sched)
        while sched.last_epoch<1155:opt.step();sched.step()
        continued={name:p for name,p in self.model.named_parameters()
                   if p.requires_grad and not name.startswith('fusion.blocks.') and not name.startswith('fusion.final_norm.')}
        state={name:{key:value.clone() if isinstance(value,torch.Tensor) else value for key,value in opt.state.get(p,{}).items()}
               for name,p in continued.items()}
        before_rng=torch.get_rng_state().clone();before_lr=sched.get_last_lr()
        base_before=digest(dict(self.model.fusion.base_parameters()))
        opt,sched,report=switch_to_adapters(self.model,opt,sched,self.config)
        self.assertEqual(sched.last_epoch,1155);self.assertEqual(before_lr,sched.get_last_lr())
        self.assertTrue(torch.equal(before_rng,torch.get_rng_state()))
        self.assertTrue(report['continuing_adam_tensor_identity_and_versions_preserved'])
        for name,p in continued.items():
            self.assertTrue(p.requires_grad)
            for key,value in state[name].items():torch.testing.assert_close(opt.state[p][key],value,atol=0,rtol=0)
        self.assertTrue(all(p not in opt.state for _,a in adapter_items(self.model) for p in a.parameters()))
        for step in range(2):
            self.step(opt,sched)
            for _,a in adapter_items(self.model):
                self.assertGreater(torch.count_nonzero(a.B.weight.grad),0)
                if step:
                    self.assertGreater(torch.count_nonzero(a.A.weight.grad),0)
                    self.assertGreater(torch.count_nonzero(a.gate.weight.grad),0)
        self.assertEqual(base_before,digest(dict(self.model.fusion.base_parameters())))
        self.assertTrue(all(p.grad is None and not p.requires_grad for _,p in self.model.fusion.base_parameters()))
        reference_opt=torch.optim.AdamW([nn.Parameter(torch.ones(1))],lr=1e-4)
        reference_sched=_scheduler(reference_opt,2310,.05,116)
        for _ in range(1157):reference_opt.step();reference_sched.step()
        self.assertTrue(all(lr==reference_sched.get_last_lr()[0] for lr in sched.get_last_lr()))
        for p in self.model.parameters():
            for key in ('exp_avg','exp_avg_sq'):
                if key in opt.state.get(p,{}):self.assertEqual(opt.state[p][key].dtype,torch.float32)

    def test_switch_rejects_wrong_update_phase_or_changed_adapter(self):
        opt=build_optimizer(self.model,self.config);sched=_scheduler(opt,2310,.05,116)
        with self.assertRaises(ValueError):switch_to_adapters(self.model,opt,sched,self.config)
        with torch.no_grad():next(iter(adapter_items(self.model)))[1].B.weight.add_(.1)
        with self.assertRaises(AssertionError):switch_to_adapters(self.model,opt,sched,self.config,expected_updates=0)

    def test_absent_private_heads_and_adam_remain_unchanged_in_both_phases(self):
        opt=build_optimizer(self.model,self.config);sched=_scheduler(opt,2310,.05,116)
        self.step(opt,sched)
        for phase in ('dense','adapter'):
            if phase=='adapter':opt,sched,_=switch_to_adapters(self.model,opt,sched,self.config,expected_updates=sched.last_epoch)
            self.ids.zero_()
            opt.zero_grad(set_to_none=True)
            before={task:optimizer_snapshot(head,opt) for task,head in self.model.heads()}
            self.model(self.x,self.z,self.ids).square().mean().backward()
            self.assertEqual(sum(check_private_activity(self.model,self.ids).values()),1)
            opt.step();sched.step()
            for task in TASKS[1:]:self.assertEqual(before[task],optimizer_snapshot(self.model.fusion.heads[task],opt))

    def test_complete_scheduler_keeps_single_warmup_and_2310_updates(self):
        opt=build_optimizer(self.model,self.config);sched=_scheduler(opt,2310,.05,116)
        for step in range(2310):
            if step==1155:opt,sched,report=switch_to_adapters(self.model,opt,sched,self.config)
            opt.step();sched.step()
        self.assertEqual(sched.last_epoch,2310)
        self.assertTrue(all(lr==0 for lr in sched.get_last_lr()))


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config=make_config(root='/tmp/aider_raw_stages_checkpoint_test')
        cls.model=initialize_model(cls.config,build_backbones=False).eval()
        cls.model.source_hashes={'unit':'fixed'}

    def test_checkpoint_phase_boundary_validation(self):
        for phase,epoch in (('dense',0),('dense',4),('adapter',5),('adapter',9)):validate_phase(self.config,phase,epoch)
        for phase,epoch in (('adapter',4),('dense',5),('joint',0)):
            with self.assertRaises(ValueError):validate_phase(self.config,phase,epoch)

    def test_roundtrip_restores_frozen_base_lora_and_private_heads_in_both_phases(self):
        model=self.model
        original={name:t.clone() for name,t in model.compact_state_dict().items()}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for phase,epoch in (('dense',4),('adapter',5)):
                    model.load_state_dict(original,strict=False);model.configure_phase(phase)
                    path=Path(tmp)/(phase+'.pt')
                    save_checkpoint(path,model,self.config,epoch)
                    expected=digest(model.compact_state_dict())
                    with torch.no_grad():
                        model.fusion.blocks[0].fc1.base.weight.add_(.2)
                        model.fusion.blocks[0].fc1.adapter.B.weight.add_(.1)
                        next(model.fusion.heads.parameters()).add_(.3)
                    load_model_checkpoint(model,path,self.config)
                    self.assertEqual(digest(model.compact_state_dict()),expected)
                    self.assertEqual(model.fusion.phase,phase)
        finally:model.load_state_dict(original,strict=False);model.configure_phase('dense')

    def test_contract_and_phase_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wrong.pt'
            torch.save(dict(model_contract='aider_aigvqa_shared_film_task_conditioned_shared_lora_raw_clip_residual_gate_r8_v1'),path)
            with self.assertRaises(ValueError):load_model_checkpoint(self.model,path,self.config)
            changed=copy.deepcopy(self.config);changed['phase_transition']['reset_learning_rate']=True
            with self.assertRaises(ValueError):validate_config(changed)


if __name__=='__main__':unittest.main()
