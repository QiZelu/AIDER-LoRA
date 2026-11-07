"""Task FiLM and residual/effective QK channel diagnostics."""
import torch
from aider_modulation.registry import BRANCHES,ATTRIBUTES
from aider_modulation.task_private import private_module_items,task_linear
from aider_modulation.qk_gate import gate_items


def module_gradient_norms(model):
    def norm(module):
        if module is None: return None
        values=[p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
        return float(torch.stack(values).sum().sqrt()) if values else None
    base=[p.grad.detach().double().square().sum() for _,p in model.fusion.base_parameters() if p.grad is not None]
    return dict(private_detail={n:norm(m) for n,_,m in private_module_items(model)},
        shared_film={b:norm(getattr(model,a)) for b,a in zip(BRANCHES,ATTRIBUTES)},
        shared_alpha=norm(model.alpha_gate),video_summaries=norm(model.video_summaries),
        qk_gates={n:norm(g) for n,g in gate_items(model)},
        fusion_base=float(torch.stack(base).sum().sqrt()) if base else None)


def cosines(x):
    x=x.double();cross=x@x.T;norm=x.norm(dim=1)
    return [[None if float(norm[i]*norm[j])==0 else float((cross[i,j]/(norm[i]*norm[j])).clamp(-1,1))
             for j in range(len(x))] for i in range(len(x))]


@torch.no_grad()
def gate_diagnostics(model):
    z=model.dimension_encoder.encoded_feature
    result=dict(phase=model.fusion.phase,gates_enabled=model.fusion.phase!='dense' and model.case!='no_qk',
        gate_definition='Q diag(1+Linear(z)) K^T; 128 channels shared across heads',
        zero_vector_cosine=None,task_order=list(model.task_order),layers={})
    for name,g in gate_items(model):
        raw=g.residual(z);effective=1+raw
        stats={}
        for i,t in enumerate(model.task_order):
            stats[t]=dict(residual=raw[i].cpu().tolist(),effective=effective[i].cpu().tolist(),
                residual_norm=float(raw[i].norm()),effective_norm=float(effective[i].norm()),
                residual_min=float(raw[i].min()),residual_max=float(raw[i].max()),
                effective_min=float(effective[i].min()),effective_max=float(effective[i].max()),
                effective_negative_fraction=float((effective[i]<0).float().mean()))
        result['layers'][name]=dict(tasks=stats,residual_cosines=cosines(raw),effective_cosines=cosines(effective),
            modulator_trainable=g.linear.weight.requires_grad)
    return result


@torch.no_grad()
def task_coefficients(model):
    ids = torch.arange(len(model.task_order),device=next(model.parameters()).device)
    condition = model.dimension_encoder(ids)
    with torch.autocast(condition.device.type,enabled=False):
        alpha = task_linear(model.alpha_gate,condition.float(),ids).sigmoid() if model.alpha_gate is not None else condition.new_zeros((4,1))
        result = {task:dict(alpha=float(alpha[i,0]),branches={}) for i,task in enumerate(model.task_order)}
        for branch,attribute in zip(BRANCHES,ATTRIBUTES):
            module=getattr(model,attribute)
            if module is None: continue
            gamma,beta = module.coefficients(condition,ids)
            for i,task in enumerate(model.task_order):
                result[task]['branches'][branch] = dict(gamma=gamma[i].cpu().tolist(),beta=beta[i].cpu().tolist())
    return result
