from pathlib import Path
import torch
from training.dataset.loader import build_loader
from training.engine.memory import initialize_process_memory,MemoryWatch,check_memory
from training.engine.checkpoint import load_model_checkpoint
from training.engine.evaluation import evaluate,export
from training.engine.diagnostics import gate_diagnostics,task_coefficients
from training.engine.io import write_json
from training.protocol import initialize_model
from training.protocol import selected_source_hashes


def evaluate_one(config,checkpoint,gpu):
    root=Path(config['training']['output_dir'])
    out=root/'evaluations'/Path(checkpoint).stem;out.mkdir(parents=True,exist_ok=False)
    initialize_process_memory()
    torch.cuda.set_per_process_memory_fraction(14*1024**3/torch.cuda.get_device_properties(0).total_memory)
    with MemoryWatch() as watch:
        model=initialize_model(config);model.source_hashes=selected_source_hashes(config)
        payload=load_model_checkpoint(model,checkpoint,config);epoch=payload['epoch'];del payload
        model.cuda();rows,metrics,routes=evaluate(model,build_loader(config,'val'),torch.device('cuda:0'))
        export(out,'evaluation',rows,metrics,routes,epoch,checkpoint)
        write_json(out/'qk.json',gate_diagnostics(model));write_json(out/'film.json',task_coefficients(model))
        check_memory(watch.report());write_json(out/'memory.json',watch.report())
