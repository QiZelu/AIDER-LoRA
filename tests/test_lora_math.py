"""Shared FiLM, conditional low-rank math, optimizer phases and strict artifacts."""
import copy
import csv
import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.fusion import TransformerFusion,TaskTransformerBlock
from models.lora import TaskLoRALinear,adapter_items,lora_modules
from models.registry import TASKS
from models.task_private import private_module_items
from engine.audit import digest,optimizer_snapshot
from engine.optim import build_optimizer,_scheduler,transition_to_adapters
from engine.checkpoint import save_checkpoint,load_model_checkpoint
from engine.preflight import sampling_audit,check_private_activity
from engine.diagnostics import lora_diagnostics
from engine.losses import regression_loss,quality_loss_protocol,TrainingLossSummary
from utils.protocol import (make_config,validate_config,validate_data,validate_reference,
                            phase_for_epoch,best_eligible,SOURCE)
from utils.initialization import initialize_model


def small_fusion(tasks=TASKS):
    return TransformerFusion(42,torch.zeros(1,8,24),width=24,heads=4,tasks=tasks)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = small_fusion()
        self.fusion.heads = nn.ModuleDict({task:nn.Linear(24,1) for task in TASKS})
        self.private_heads = True
        self.task_order = TASKS
        self.configure_phase('dense')

    def configure_phase(self,phase):
        self.zero_grad(set_to_none=True)
        self.fusion.configure_phase(phase)

    def heads(self):
        yield from self.fusion.heads.items()

    def forward(self,x,z,ids):
        features = self.fusion(x,z,ids)[0]
        values,positions = [],[]
        for index,task in enumerate(TASKS):
            selected = torch.where(ids==index)[0]
            if selected.numel():
                values.append(self.fusion.heads[task](features[selected]).mean((1,2)))
                positions.append(selected)
        return torch.cat(values)[torch.cat(positions).argsort()]


class ConditionalLoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_initial_gate_is_one_and_zero_delta_matches_base(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,dropout=0,condition_width=4).eval()
        x,z = torch.randn(8,3,6),torch.randn(8,4)*10
        self.assertTrue(torch.equal(layer.adapter.channel_residual(z),torch.zeros(8,2)))
        self.assertEqual(torch.count_nonzero(layer.adapter.gate.weight),0)
        self.assertEqual(torch.count_nonzero(layer.adapter.gate.bias),0)
        self.assertTrue(torch.equal(layer.adapter.channel_weights(z),torch.ones(8,2)))
        self.assertTrue(torch.equal(layer(x,z),layer.base(x)))

    def test_residual_modulation_allows_signed_and_unbounded_channels(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,dropout=0,condition_width=4)
        with torch.no_grad():
            layer.adapter.gate.weight.zero_()
            layer.adapter.gate.weight[0,0] = 2.
            layer.adapter.gate.weight[1,1] = -3.
            layer.adapter.gate.bias.copy_(torch.tensor([-.5,.25]))
        z = torch.tensor([[2.,1.,0.,0.],[-1.,-2.,0.,0.]])
        residual = torch.tensor([[3.5,-2.75],[-2.5,6.25]])
        torch.testing.assert_close(layer.adapter.channel_residual(z),residual,atol=0,rtol=0)
        torch.testing.assert_close(layer.adapter.channel_weights(z),1+residual,atol=0,rtol=0)

    def test_residual_modulator_keeps_unit_gradient_at_large_outputs(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,dropout=0,condition_width=4)
        with torch.no_grad():
            layer.adapter.gate.bias.copy_(torch.tensor([-100.,100.]))
        z = torch.tensor([[1.,-2.,3.,-4.]])
        output = layer.adapter.channel_weights(z)
        torch.testing.assert_close(output,torch.tensor([[-99.,101.]]),atol=0,rtol=0)
        output.sum().backward()
        torch.testing.assert_close(layer.adapter.gate.bias.grad,torch.ones(2),atol=0,rtol=0)
        torch.testing.assert_close(layer.adapter.gate.weight.grad,z.expand(2,-1),atol=0,rtol=0)

    def test_bias_shift_matches_legacy_linear_outputs_and_gradients(self):
        residual = TaskLoRALinear(6,5,TASKS,42,rank=2,dropout=0,condition_width=4).eval()
        with torch.no_grad():
            residual.adapter.B.weight.normal_(0,.1)
            residual.adapter.gate.weight.normal_(0,.1)
            residual.adapter.gate.bias.normal_(0,.1)
        legacy = copy.deepcopy(residual)
        with torch.no_grad():legacy.adapter.gate.bias.add_(1.)
        z = torch.randn(8,4)
        torch.testing.assert_close(residual.adapter.channel_weights(z),legacy.adapter.gate(z),atol=1e-6,rtol=0)
        for x in (torch.randn(8,6),torch.randn(8,3,6)):
            residual.zero_grad(set_to_none=True)
            legacy.zero_grad(set_to_none=True)
            hidden = legacy.adapter.A(x)
            broadcast = [len(x)]+[1]*(x.ndim-2)+[legacy.adapter.rank]
            direct = legacy.base(x)+legacy.adapter.B(hidden*legacy.adapter.gate(z).reshape(broadcast))*legacy.adapter.scale
            actual = residual(x,z)
            torch.testing.assert_close(actual,direct,atol=1e-6,rtol=0)
            actual.square().mean().backward()
            direct.square().mean().backward()
            for (name,p),(_,q) in zip(residual.named_parameters(),legacy.named_parameters()):
                torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=0,msg=name)

    def test_formula_matches_materialized_diagonal_for_2d_and_3d_inputs(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,alpha=4,dropout=0,condition_width=4).eval()
        with torch.no_grad():
            layer.adapter.B.weight.normal_(0,.1)
            layer.adapter.gate.weight.normal_(0,.1)
        z = torch.randn(8,4)
        g = layer.adapter.channel_weights(z)
        for x in (torch.randn(8,6),torch.randn(8,3,6)):
            expected = []
            for i in range(8):
                delta = layer.adapter.B.weight @ torch.diag(g[i]) @ layer.adapter.A.weight * layer.adapter.scale
                expected.append(F.linear(x[i],layer.base.weight+delta,layer.base.bias))
            torch.testing.assert_close(layer(x,z),torch.stack(expected),atol=1e-6,rtol=1e-5)

    def test_mixed_batch_matches_per_sample_with_frame_broadcast(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,dropout=0,condition_width=4).eval()
        with torch.no_grad():
            layer.adapter.B.weight.normal_(0,.1)
            layer.adapter.gate.weight.normal_(0,.2)
        x,z = torch.randn(8,8,6),torch.randn(8,4)
        expected = torch.cat([layer(x[i:i+1],z[i:i+1]) for i in range(8)])
        torch.testing.assert_close(layer(x,z),expected,atol=1e-6,rtol=1e-5)

    def test_task_embedding_controls_update_after_modulator_learns(self):
        layer = TaskLoRALinear(2,2,TASKS,42,rank=1,dropout=0,condition_width=2).eval()
        with torch.no_grad():
            layer.adapter.A.weight.fill_(1.)
            layer.adapter.B.weight.fill_(1.)
            layer.adapter.gate.weight.copy_(torch.tensor([[1.,0.]]))
        x,z = torch.ones(2,3,2),torch.tensor([[1.,0.],[-1.,0.]])
        self.assertFalse(torch.equal(layer(x,z)[0],layer(x,z)[1]))
        torch.testing.assert_close(layer(x,z[:1].expand(2,-1))[0],layer(x,z[:1].expand(2,-1))[1])

    def test_parameter_count_and_base_rng_do_not_depend_on_task_count(self):
        torch.manual_seed(42)
        one = TaskLoRALinear(6,5,('one',),42,rank=2,condition_width=4)
        state = torch.get_rng_state().clone()
        torch.manual_seed(42)
        eight = TaskLoRALinear(6,5,tuple(str(i) for i in range(8)),42,rank=2,condition_width=4)
        self.assertEqual(sum(p.numel() for p in one.parameters()),sum(p.numel() for p in eight.parameters()))
        self.assertEqual(digest(one.state_dict()),digest(eight.state_dict()))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertFalse(any('.adapters.' in name or '.by_task.' in name for name,_ in eight.named_parameters()))

    def test_B_then_A_and_modulator_gradients_open(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,dropout=0,condition_width=4)
        layer.base.requires_grad_(False)
        optimizer = torch.optim.AdamW(layer.adapter.parameters(),lr=.01,weight_decay=0)
        x,z = torch.randn(8,3,6),torch.randn(8,4)
        layer(x,z).square().mean().backward()
        self.assertEqual(torch.count_nonzero(layer.adapter.A.weight.grad),0)
        self.assertEqual(torch.count_nonzero(layer.adapter.gate.weight.grad),0)
        self.assertGreater(torch.count_nonzero(layer.adapter.B.weight.grad),0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        layer(x,z).square().mean().backward()
        self.assertGreater(torch.count_nonzero(layer.adapter.A.weight.grad),0)
        self.assertGreater(torch.count_nonzero(layer.adapter.gate.weight.grad),0)
        self.assertGreater(torch.count_nonzero(layer.adapter.gate.bias.grad),0)

    def test_shared_gradients_equal_sum_of_task_subset_gradients(self):
        mixed = TaskLoRALinear(6,5,TASKS,42,rank=2,dropout=0,condition_width=4)
        with torch.no_grad():
            mixed.adapter.B.weight.normal_(0,.1)
        separate = copy.deepcopy(mixed)
        x,z = torch.randn(8,3,6),torch.randn(8,4)
        mixed(x,z).square().sum().backward()
        for subset in (slice(0,4),slice(4,8)):
            separate(x[subset],z[subset]).square().sum().backward()
        for (name,p),(_,q) in zip(mixed.named_parameters(),separate.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-5,msg=name)

    def test_modulator_stays_fp32_under_autocast(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,condition_width=4)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            z = torch.randn(8,4)
            residual = layer.adapter.channel_residual(z)
            gate = layer.adapter.channel_weights(z)
        self.assertEqual(residual.dtype,torch.float32)
        self.assertEqual(gate.dtype,torch.float32)
        self.assertTrue(torch.isfinite(gate).all())

    def test_invalid_conditions_and_nonfinite_embeddings_are_rejected(self):
        layer = TaskLoRALinear(6,5,TASKS,42,rank=2,condition_width=4)
        for z in (None,torch.randn(7,4),torch.randn(8,3)):
            with self.assertRaises(ValueError):
                layer(torch.randn(8,3,6),z)
        with self.assertRaises(FloatingPointError):
            layer(torch.randn(8,3,6),torch.full((8,4),float('nan')))

    def test_zero_delta_block_equals_ordinary_pre_ln_transformer(self):
        block = TaskTransformerBlock(42,0,width=24,heads=4).eval()
        mha = nn.MultiheadAttention(24,4,dropout=0,batch_first=True)
        with torch.no_grad():
            mha.in_proj_weight.copy_(torch.cat([getattr(block.attention,n).base.weight for n in ('q_proj','k_proj','v_proj')]))
            mha.in_proj_bias.copy_(torch.cat([getattr(block.attention,n).base.bias for n in ('q_proj','k_proj','v_proj')]))
            mha.out_proj.load_state_dict(block.attention.out_proj.base.state_dict())
        x,z = torch.randn(8,8,24),torch.randn(8,768)
        normalized = block.norm1(x)
        u = x + mha(normalized,normalized,normalized,need_weights=False)[0]
        expected = u + block.fc2.base(F.gelu(block.fc1.base(block.norm2(u))))
        torch.testing.assert_close(block(x,z),expected,atol=1e-6,rtol=1e-6)

    def test_checkpoint_recomputation_preserves_condition_and_dropout(self):
        checkpointed = small_fusion().train()
        for _,adapter in adapter_items(checkpointed):
            with torch.no_grad():
                adapter.B.weight.normal_(0,.01)
                adapter.gate.weight.normal_(0,.01)
        direct = copy.deepcopy(checkpointed)
        direct.use_checkpoint = False
        x,z,ids = torch.randn(8,8,24),torch.randn(8,768),torch.tensor([0,3,1,2,3,0,2,1])
        for model in (checkpointed,direct):
            torch.manual_seed(100)
            model(x,z,ids)[0][...,0].mean().backward()
        for (name,p),(_,q) in zip(checkpointed.named_parameters(),direct.named_parameters()):
            if p.grad is None:
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-5,msg=name)

    def test_disabled_adapter_has_no_gradients_or_rng_consumption(self):
        model = small_fusion().train()
        model.configure_phase('dense')
        x,z,ids = torch.randn(8,8,24),torch.randn(8,768),torch.zeros(8,dtype=torch.int64)
        before = torch.get_rng_state().clone()
        model(x,z,ids)[0][...,0].mean().backward()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for _,a in adapter_items(model) for p in a.parameters()))

    def test_task_delta_diagnostics_match_materialized_gated_matrices(self):
        model = TinyModel().eval()
        model.configure_phase('adapter')
        model.dimension_encoder = nn.Module()
        model.dimension_encoder.register_buffer('encoded_feature',torch.randn(4,768))
        for _,adapter in adapter_items(model):
            with torch.no_grad():
                adapter.B.weight.normal_(0,.01)
                adapter.gate.weight.normal_(0,.01)
        report = lora_diagnostics(model)
        name,module = next(lora_modules(model))
        g = module.adapter.channel_weights(model.dimension_encoder.encoded_feature)
        residual = module.adapter.channel_residual(model.dimension_encoder.encoded_feature)
        matrices = [module.adapter.B.weight @ torch.diag(g[i]) @ module.adapter.A.weight * module.adapter.scale for i in range(4)]
        corrections = [module.adapter.B.weight @ torch.diag(residual[i]) @ module.adapter.A.weight * module.adapter.scale for i in range(4)]
        shared = module.adapter.B.weight @ module.adapter.A.weight * module.adapter.scale
        item = report['projections'][name]
        self.assertAlmostEqual(item['unit_gate_shared_weight_norm'],float(shared.norm()),places=5)
        for i,task in enumerate(TASKS):
            self.assertAlmostEqual(item['tasks'][task]['delta_weight_norm'],float(matrices[i].norm()),places=5)
            self.assertEqual(item['tasks'][task]['channel_residual'],residual[i].tolist())
            self.assertAlmostEqual(item['tasks'][task]['channel_residual_norm'],float(residual[i].norm()),places=5)
            self.assertAlmostEqual(item['tasks'][task]['task_residual_weight_norm'],float(corrections[i].norm()),places=5)
            torch.testing.assert_close(matrices[i],shared+corrections[i],atol=1e-6,rtol=1e-5)
            for j in range(4):
                expected = float(F.cosine_similarity(matrices[i].flatten(),matrices[j].flatten(),dim=0))
                self.assertAlmostEqual(item['delta_weight_cosines'][i][j],expected,places=5)
                expected_residual = float(F.cosine_similarity(residual[i],residual[j],dim=0))
                self.assertAlmostEqual(item['channel_residual_task_cosines'][i][j],expected_residual,places=5)
                expected_correction = float(F.cosine_similarity(corrections[i].flatten(),corrections[j].flatten(),dim=0))
                self.assertAlmostEqual(item['task_residual_weight_cosines'][i][j],expected_correction,places=5)

    def test_zero_residual_and_weight_update_cosines_are_null(self):
        model = TinyModel().eval()
        model.configure_phase('joint')
        model.dimension_encoder = nn.Module()
        model.dimension_encoder.register_buffer('encoded_feature',torch.randn(4,768))
        report = lora_diagnostics(model)
        self.assertEqual(len(report['projections']),12)
        null_matrix = [[None]*4 for _ in range(4)]
        for item in report['projections'].values():
            self.assertEqual(item['channel_residual_task_cosines'],null_matrix)
            self.assertEqual(item['task_residual_weight_cosines'],null_matrix)
            self.assertEqual(item['delta_weight_cosines'],null_matrix)
            for row in item['gate_task_cosines']:
                for value in row:self.assertAlmostEqual(value,1.,places=12)
            for task in TASKS:
                self.assertEqual(item['tasks'][task]['channel_residual'],[0.]*8)
                self.assertIsNone(item['tasks'][task]['task_residual_to_unit_gate_weight_norm'])

    def test_zero_effective_channels_have_null_cosines(self):
        model = TinyModel().eval()
        model.configure_phase('joint')
        model.dimension_encoder = nn.Module()
        model.dimension_encoder.register_buffer('encoded_feature',torch.randn(4,768))
        for _,adapter in adapter_items(model):
            with torch.no_grad():
                adapter.gate.bias.fill_(-1.)
                adapter.B.weight.normal_(0,.01)
        for item in lora_diagnostics(model)['projections'].values():
            self.assertEqual(item['gate_task_cosines'],[[None]*4 for _ in range(4)])
            self.assertEqual(item['delta_weight_cosines'],[[None]*4 for _ in range(4)])
            self.assertGreater(item['tasks'][TASKS[0]]['task_residual_weight_norm'],0)


if __name__ == "__main__":unittest.main()
