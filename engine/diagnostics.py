"""Task-private LoRA/FiLM gradients and low-rank weight-delta diagnostics."""
import torch
from models.registry import BRANCHES, ATTRIBUTES
from models.task_private import private_module_items, task_linear
from models.lora import task_adapter_items, lora_modules


def module_gradient_norms(model):
    def norm(module):
        values = [p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
        return float(torch.stack(values).sum().sqrt()) if values else None
    base = [p.grad.detach().double().square().sum() for _,p in model.fusion.base_parameters() if p.grad is not None]
    return dict(private_detail={name:norm(module) for name,_,module in private_module_items(model)},
        lora={name:norm(adapter) for name,_,adapter in task_adapter_items(model)},
        fusion_base=float(torch.stack(base).sum().sqrt()) if base else None)


@torch.no_grad()
def lora_diagnostics(model):
    result = dict(phase=model.fusion.phase,adapters_enabled=model.fusion.phase != 'dense',projections={})
    for name, module in lora_modules(model):
        tasks, norms = {}, []
        adapters = list(module.adapters.values())
        for task, adapter in module.adapters.items():
            a, b = adapter.A.weight.float(), adapter.B.weight.float()
            square = ((b.T @ b) * (a @ a.T).T).sum().clamp_min(0)
            delta_norm = float(square.sqrt() * adapter.scale)
            base_norm = float(module.base.weight.float().norm())
            tasks[task] = dict(A_norm=float(a.norm()),B_norm=float(b.norm()),
                delta_weight_norm=delta_norm,delta_to_base_norm=delta_norm/max(base_norm,1e-30),
                trainable=adapter.A.weight.requires_grad)
            norms.append(delta_norm)
        cosines = []
        for i, left in enumerate(adapters):
            row = []
            for j, right in enumerate(adapters):
                if norms[i] == 0 or norms[j] == 0:
                    row.append(None)
                else:
                    cross = ((left.B.weight.float().T @ right.B.weight.float()) *
                             (right.A.weight.float() @ left.A.weight.float().T).T).sum()
                    value = float(cross * left.scale * right.scale) / (norms[i]*norms[j])
                    row.append(max(-1.,min(1.,value)))
            cosines.append(row)
        result['projections'][name] = dict(tasks=tasks,task_order=list(module.task_order),delta_weight_cosines=cosines)
    return result


@torch.no_grad()
def task_coefficients(model):
    ids=torch.arange(len(model.task_order),device=next(model.parameters()).device)
    condition=model.dimension_encoder(ids)
    with torch.autocast(condition.device.type,enabled=False):
        alpha=task_linear(model.alpha_gate,condition.float(),ids).sigmoid()
        result={task:dict(alpha=float(alpha[i,0]),branches={}) for i,task in enumerate(model.task_order)}
        for branch,attribute in zip(BRANCHES,ATTRIBUTES):
            gamma,beta=getattr(model,attribute).coefficients(condition,ids)
            for i,task in enumerate(model.task_order):
                result[task]['branches'][branch]=dict(gamma=gamma[i].cpu().tolist(),beta=beta[i].cpu().tolist())
    return result
