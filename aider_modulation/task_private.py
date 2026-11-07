"""Only scoring Heads remain task-private; all FiLM conditioning is shared."""


def task_linear(module, features, task_ids=None):
    # task_ids validate metadata upstream; shared maps depend on embedding values.
    return module(features)


def private_module_items(model):
    if getattr(model, 'private_heads', False):
        for task, head in model.heads():
            yield 'head.' + task, task, head
