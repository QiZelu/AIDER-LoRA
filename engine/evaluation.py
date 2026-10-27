"""Four-task metrics and video-conditional FiLM diagnostics for task LoRAs."""

import numpy as np
import torch
from models.registry import TASKS, BRANCHES, ATTRIBUTES
from engine.metrics import regression_metrics
from engine.runtime import autocast_context, move_to_device
from engine.io import write_csv, write_json


@torch.no_grad()
def evaluate(model, loader, device):
    tasks = tuple(model.task['order'])
    model.eval()
    rows, routes = [], []
    for cpu in loader:
        batch = move_to_device(cpu, device)
        with autocast_context(device):
            result = model(batch)
        pred = result["predictions"].float().cpu()
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Nonfinite validation predictions")
        routing = result["routing"]
        for i, video_id in enumerate(cpu["video_id"]):
            task_id = int(cpu["task_id"][i])
            task = tasks[task_id]
            rows.append(
                dict(
                    video_id=video_id,
                    task=task,
                    task_id=task_id,
                    dimension_text=model.task["descriptions"][task],
                    prompt=cpu["prompt"][i],
                    target=float(cpu["target"][i]),
                    prediction=float(pred[i]),
                )
            )
            record = dict(
                video_id=video_id,
                task=task,
                film_video_condition=model.film_video_condition,
                frame_scores=result["frame_scores"][i].float().cpu().tolist(),
                fusion_phase=routing['fusion_phase'],
                adapters_enabled=routing['adapters_enabled'],
            )
            context_key = "cross_branch_input" if model.film_video_condition == "cross_branch_linear" else "video_summary"
            if isinstance(routing.get(context_key), dict):
                record[context_key] = {
                    b: routing[context_key][b][i].float().cpu().tolist()
                    for b in BRANCHES
                }
                record["alpha"] = float(routing["alpha"][i, 0])
                condition = routing["dimension_features"]
                for b, attr in zip(BRANCHES, ATTRIBUTES):
                    modulation = getattr(model, attr).modulation(
                        condition, routing["video_condition"][b], routing["alpha"], batch['task_id']
                    )
                    record.setdefault("film", {})[b] = {
                        k + "_norm": float(v[i].float().norm())
                        for k, v in modulation.items()
                    }
                    if model.film_video_condition == "cross_branch_linear":
                        stats = record["film"][b]
                        width = modulation["task_gamma"].shape[-1]
                        stats.update({
                            "scaled_delta_gamma_rms": float((routing["alpha"] * modulation["delta_gamma"])[i].norm() / width ** 0.5),
                            "scaled_delta_beta_rms": float((routing["alpha"] * modulation["delta_beta"])[i].norm() / width ** 0.5),
                            "video_feature_change_rms": float(routing["video_feature_change_rms"][b][i]),
                        })
            routes.append(record)
    if len(rows) != len(loader.dataset) or len(
        {(r["video_id"], r["task"]) for r in rows}
    ) != len(rows):
        raise ValueError("Evaluation coverage mismatch")
    metrics = {}
    for task in tasks:
        subset = [r for r in rows if r["task"] == task]
        values = regression_metrics(
            np.asarray([r["prediction"] for r in subset], np.float32)[:, None],
            np.asarray([r["target"] for r in subset], np.float32)[:, None],
            ["score"],
        )["score"]
        metrics[task] = dict(
            count=len(subset),
            **{
                k: float(values[k]) if np.isfinite(values[k]) else None
                for k in ("plcc", "srocc", "krocc", "rmse")
            },
        )
    metrics["macro"] = {
        k: float(np.mean([metrics[t][k] for t in tasks]))
        if all(metrics[t][k] is not None for t in tasks)
        else None
        for k in ("plcc", "srocc", "krocc", "rmse")
    }
    return rows, metrics, routes


def validate_metrics(metrics):
    from models.registry import TASKS
    if set(metrics) != {*TASKS, 'macro'}:
        raise ValueError('Incomplete four-task metrics')
    for task in TASKS:
        if metrics[task]['count'] != 115:
            raise ValueError('Expected 115 test videos per task')
    for key in ('plcc', 'srocc', 'krocc', 'rmse'):
        values = [metrics[t][key] for t in TASKS]
        if any(v is None or not np.isfinite(v) for v in values):
            raise ValueError('Nonfinite test metric: ' + key)
        if metrics['macro'][key] is None or not np.isclose(metrics['macro'][key], np.mean(values), atol=1e-12, rtol=0):
            raise ValueError('Incorrect macro metric: ' + key)


def export(out, kind, rows, metrics, routes, epoch, checkpoint=None):
    from pathlib import Path
    from engine.audit import file_digest
    out = Path(out)
    validate_metrics(metrics)
    tasks = tuple(t for t in metrics if t != 'macro')
    write_json(
        out / f"{kind}_evaluation.json",
        dict(
            completed=True,
            checkpoint_epoch=epoch + 1,
            checkpoint_sha256=file_digest(checkpoint) if checkpoint is not None else None,
            metrics=metrics,
            samples=rows,
            selection_uses_same_heldout=True,
        ),
    )
    write_csv(out / f"{kind}_predictions_long.csv", rows)
    lookup = {(r["video_id"], r["task"]): r for r in rows}
    wide = []
    for video_id in sorted({r["video_id"] for r in rows}):
        record = dict(video_id=video_id, prompt=lookup[video_id, tasks[0]]["prompt"])
        for task in tasks:
            record[task + "_target"] = lookup[video_id, task]["target"]
            record[task + "_prediction"] = lookup[video_id, task]["prediction"]
        wide.append(record)
    write_csv(out / f"{kind}_predictions_wide.csv", wide)
    write_json(out / f"{kind}_film_diagnostics.json", routes)
