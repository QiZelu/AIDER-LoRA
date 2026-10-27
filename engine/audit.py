"""Small state/parameter audits used by baseline verification and preflight."""
import hashlib
import torch


def digest(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(f'{name}|{value.dtype}|{tuple(value.shape)}'.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def parameter_counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        modules={name: dict(total=sum(p.numel() for p in module.parameters()),
                           trainable=sum(p.numel() for p in module.parameters() if p.requires_grad))
                 for name, module in model.named_children()})


def optimizer_snapshot(module, optimizer):
    states = {f'{i}.{key}': value for i, parameter in enumerate(module.parameters())
              for key, value in optimizer.state.get(parameter, {}).items()}
    return dict(parameters=digest(module.state_dict()), optimizer=digest(states))
