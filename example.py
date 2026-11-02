import argparse
import csv
import json
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import DecoupledPhysicalMSE, GATTransformer
from training import train_epoch


def load_example(directory):
    directory = Path(directory)
    graph = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
    case = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    with (directory / "results" / "displacement.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        results = list(csv.DictReader(stream))
    if [int(row["node_id"]) for row in results] != graph["node_ids"]:
        raise ValueError("Node order mismatch")
    if [int(row["graph_index"]) for row in results] != list(range(len(results))):
        raise ValueError("Graph indices must be contiguous")
    if any(float(row["time_s"]) != case["result_time_s"] for row in results):
        raise ValueError("Result time mismatch")
    scale = torch.tensor(case["target_scale_m"], dtype=torch.float32)
    if scale.shape != (3,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid displacement scale")
    displacement = torch.tensor(
        [[float(row[key]) for key in ("ux_m", "uy_m", "uz_m")] for row in results],
        dtype=torch.float32,
    )
    data = Data(
        x=torch.tensor(graph["x"], dtype=torch.float32),
        edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(graph["edge_attr"], dtype=torch.float32),
        global_attr=torch.tensor(graph["global_attr"], dtype=torch.float32),
        y=displacement / scale,
        pos=torch.tensor(graph["positions_m"], dtype=torch.float32),
        node_ids=torch.tensor(graph["node_ids"], dtype=torch.long),
    )
    if (
        data.x.shape != (len(results), 15)
        or data.edge_attr.shape != (data.num_edges, 8)
        or data.global_attr.shape != (1, 13)
    ):
        raise ValueError("Invalid feature dimensions")
    for values in (data.x, data.edge_attr, data.global_attr, data.y, data.pos):
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite graph values")
    if data.edge_index.min() < 0 or data.edge_index.max() >= data.num_nodes:
        raise ValueError("Invalid edge endpoint")
    return data, case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path(__file__).parent / "example")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--export-pt", type=Path)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("steps must be nonnegative")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    data, case = load_example(args.case)
    print(
        f"case={case['name']}, nodes={data.num_nodes}, directed_edges={data.num_edges}"
    )
    print(
        f"features={tuple(data.x.shape)}, targets={tuple(data.y.shape)}, time_s={case['result_time_s']}"
    )
    if args.export_pt:
        args.export_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, args.export_pt)
    if args.steps == 0:
        return
    device = torch.device(args.device)
    model = GATTransformer().to(device)
    criterion = DecoupledPhysicalMSE(case["target_scale_m"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = DataLoader([data], batch_size=1)
    for step in range(args.steps):
        loss = train_epoch(model, loader, optimizer, scaler, criterion)
        print(f"step={step + 1}, loss_m2={loss:.8f}")


if __name__ == "__main__":
    main()
