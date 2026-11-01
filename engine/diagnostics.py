"""Separate unit-baseline LoRA updates from task-conditioned channel residuals."""
import torch
from models.registry import BRANCHES,ATTRIBUTES
from models.task_private import private_module_items,task_linear
from models.lora import adapter_items,lora_modules


def module_gradient_norms(model):
    def norm(module):
        values = [p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
        return float(torch.stack(values).sum().sqrt()) if values else None
    base = [p.grad.detach().double().square().sum() for _,p in model.fusion.base_parameters() if p.grad is not None]
    return dict(private_detail={name:norm(module) for name,_,module in private_module_items(model)},
        shared_film={branch:norm(getattr(model,attr)) for branch,attr in zip(BRANCHES,ATTRIBUTES)},
        shared_alpha=norm(model.alpha_gate),video_summaries=norm(model.video_summaries),
        shared_lora={name:dict(A=norm(a.A),B=norm(a.B),channel_modulator=norm(a.gate)) for name,a in adapter_items(model)},
        fusion_base=float(torch.stack(base).sum().sqrt()) if base else None)


def _gram_cosines(cross):
    norms = cross.diag().clamp_min(0).sqrt()
    return [[None if float(norms[i]*norms[j]) == 0 else
             max(-1.,min(1.,float(cross[i,j]/(norms[i]*norms[j]))))
             for j in range(len(norms))] for i in range(len(norms))]


@torch.no_grad()
def lora_diagnostics(model):
    condition = model.dimension_encoder.encoded_feature
    result = dict(phase=model.fusion.phase,adapters_enabled=model.fusion.phase != 'dense',
                  sharing='shared_A_B_and_modulator_across_tasks',modulation='linear_residual',
                  task_residual_reference='unit channels with current learned A/B, not initial A/B',
                  zero_vector_cosine=None,projections={})
    for name,module in lora_modules(model):
        adapter = module.adapter
        a,b = adapter.A.weight.double(),adapter.B.weight.double()
        residual = adapter.channel_residual(condition).double()
        gate = adapter.channel_weights(condition).double()
        metric = (b.T @ b) * (a @ a.T).T
        cross = (gate @ metric @ gate.T) * adapter.scale**2
        norms = cross.diag().clamp_min(0).sqrt()
        residual_cross = (residual @ metric @ residual.T) * adapter.scale**2
        residual_weight_norms = residual_cross.diag().clamp_min(0).sqrt()
        residual_norms = residual.norm(dim=1)
        unit_gate_weight_norm = float((metric.sum()*adapter.scale**2).clamp_min(0).sqrt())
        tasks = {}
        base_norm = float(module.base.weight.float().norm())
        for index,task in enumerate(model.task_order):
            tasks[task] = dict(gate=gate[index].float().cpu().tolist(),gate_min=float(gate[index].min()),
                gate_max=float(gate[index].max()),gate_mean=float(gate[index].mean()),
                channel_residual=residual[index].float().cpu().tolist(),
                channel_residual_norm=float(residual_norms[index]),
                channel_residual_rms=float(residual_norms[index])/adapter.rank**.5,
                delta_weight_norm=float(norms[index]),delta_to_base_norm=float(norms[index])/max(base_norm,1e-30),
                task_residual_weight_norm=float(residual_weight_norms[index]),
                task_residual_to_unit_gate_weight_norm=(float(residual_weight_norms[index])/unit_gate_weight_norm
                                                       if unit_gate_weight_norm else None))
        result['projections'][name] = dict(tasks=tasks,task_order=list(model.task_order),
            shared_A_norm=float(a.norm()),shared_B_norm=float(b.norm()),
            channel_modulator_trainable=adapter.gate.weight.requires_grad,
            gate_task_cosines=_gram_cosines(gate @ gate.T),
            gate_std_across_tasks=gate.std(0,unbiased=False).float().cpu().tolist(),
            channel_residual_task_cosines=_gram_cosines(residual @ residual.T),
            channel_residual_std_across_tasks=residual.std(0,unbiased=False).float().cpu().tolist(),
            unit_gate_shared_weight_norm=unit_gate_weight_norm,
            task_residual_weight_cosines=_gram_cosines(residual_cross),
            delta_weight_cosines=_gram_cosines(cross))
    return result


@torch.no_grad()
def task_coefficients(model):
    ids = torch.arange(len(model.task_order),device=next(model.parameters()).device)
    condition = model.dimension_encoder(ids)
    with torch.autocast(condition.device.type,enabled=False):
        alpha = task_linear(model.alpha_gate,condition.float(),ids).sigmoid()
        result = {task:dict(alpha=float(alpha[i,0]),branches={}) for i,task in enumerate(model.task_order)}
        for branch,attribute in zip(BRANCHES,ATTRIBUTES):
            gamma,beta = getattr(model,attribute).coefficients(condition,ids)
            for i,task in enumerate(model.task_order):
                result[task]['branches'][branch] = dict(gamma=gamma[i].cpu().tolist(),beta=beta[i].cpu().tolist())
    return result
