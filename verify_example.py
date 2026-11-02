import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from example import load_example


def verify(directory):
    directory = Path(directory)
    data, case = load_example(directory)
    payload = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
    deck = (directory / "model.k").read_text(encoding="ascii")
    if not deck.strip().startswith("*KEYWORD") or not deck.strip().endswith("*END"):
        raise ValueError("Incomplete keyword deck")
    nodes, references = {}, []
    counts = Counter()
    keyword = ""
    for line in deck.splitlines():
        if line.lstrip().startswith("$") or not line.strip():
            continue
        if line.startswith("*"):
            keyword = line.upper()
            if keyword.startswith("*INCLUDE"):
                raise ValueError("External keyword dependency")
            continue
        fields = line.split()
        if keyword == "*NODE":
            node_id = int(fields[0])
            if node_id in nodes:
                raise ValueError("Duplicate FE node")
            nodes[node_id] = [float(value) for value in fields[1:4]]
        elif keyword in ("*ELEMENT_BEAM", "*ELEMENT_SHELL"):
            width = 3 if keyword == "*ELEMENT_BEAM" else 4
            references.extend(
                int(value) for value in fields[2 : 2 + width] if int(value) > 0
            )
            counts[keyword] += 1
    if any(node_id not in nodes for node_id in references):
        raise ValueError("Unresolved element node")
    for node_id, xyz in zip(payload["node_ids"], payload["positions_m"]):
        if node_id not in nodes or any(
            not math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
            for a, b in zip(nodes[node_id], xyz)
        ):
            raise ValueError("FE/graph coordinate mismatch")
    directed = {}
    for i, pair in enumerate(zip(*payload["edge_index"])):
        if pair in directed:
            raise ValueError("Duplicate directed edge")
        directed[pair] = payload["edge_attr"][i]
    for (source, target), attributes in directed.items():
        if directed.get((target, source)) != attributes:
            raise ValueError("Missing or inconsistent reverse edge")
    with (directory / "edges.csv").open(newline="", encoding="utf-8") as stream:
        connections = list(csv.DictReader(stream))
    if len(connections) * 2 != len(directed):
        raise ValueError("Connection count mismatch")
    for row in connections:
        source, target = int(row["source"]), int(row["target"])
        if (
            (source, target) not in directed
            or payload["node_ids"][source] != int(row["source_node_id"])
            or payload["node_ids"][target] != int(row["target_node_id"])
        ):
            raise ValueError("Edge mapping mismatch")
    raw_values = {}
    raw_output = (directory / "results" / "nodout_terminal.txt").read_text(
        encoding="ascii"
    )
    times = re.findall(r"\(\s*at\s+time\s+([-+0-9.Ee]+)\s*\)", raw_output)
    if len(times) != 1 or float(times[0]) != case["result_time_s"]:
        raise ValueError("Nodout time mismatch")
    for line in raw_output.splitlines():
        fields = line.split()
        if len(fields) == 13 and fields[0].isdigit():
            raw_values[int(fields[0])] = [float(value) for value in fields[1:4]]
    with (directory / "results" / "displacement.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            if raw_values.get(int(row["node_id"])) != [
                float(row[key]) for key in ("ux_m", "uy_m", "uz_m")
            ]:
                raise ValueError("CSV/nodout displacement mismatch")
    if (
        len(nodes) != case["fe_nodes"]
        or counts["*ELEMENT_BEAM"] != case["beam_elements"]
        or counts["*ELEMENT_SHELL"] != case["shell_elements"]
    ):
        raise ValueError("FE mesh count mismatch")
    print(
        f"Verified {len(nodes)} FE nodes, {len(references)} element-node references, {data.num_nodes} graph nodes and {len(directed)} directed graph edges"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path(__file__).parent / "example")
    verify(parser.parse_args().case)
