"""Convert the supplied LogisticRHS evidence inputs to runtime numeric files.

The private model and data are not embedded in this repository. The input is
the frozen-inputs.json in the supplied logistic-joint-fullmodel archive.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys

source, output = map(Path, sys.argv[1:3])
models = json.loads(source.read_text())["models"]
output.mkdir(parents=True, exist_ok=True)
(output / "source.sha256").write_text(hashlib.sha256(source.read_bytes()).hexdigest() + "\n")
for name, model in models.items():
    dest = output / name
    dest.mkdir(exist_ok=True)
    args, points = model["args"], model["points"]
    n, d = args["x"]["size"]
    assert len(args["x"]["data"]) == n * d
    assert len(args["y"]) == n
    (dest / "shape.txt").write_text(f"{n}\n{d}\n{len(points)}\n")

    def write_floats(filename, values):
        (dest / filename).write_bytes(struct.pack("<" + "d" * len(values), *values))

    # The archive's data is already Julia column-major storage.
    write_floats("x.bin", args["x"]["data"])
    write_floats("y.bin", args["y"])
    write_floats("settings.bin", [args[k] for k in
        ("scale_icept", "scale_global", "nu_global", "nu_local", "slab_scale", "slab_df")])
    names = []
    for coordinate in points[0]["coordinates"]:
        key = "caux" if coordinate["name"] == "raw_caux" else coordinate["name"]
        first, last = coordinate["range"]
        names.extend([key] if first == last else [f"{key}.{i}" for i in range(1, last-first+2)])
    (dest / "coordinates.txt").write_text("\n".join(names) + "\n")
    for i, point in enumerate(points, 1):
        c = point["constrained"]
        assert len(c["z"]) == len(c["lambda"]) == d
        write_floats(f"point-{i}.bin", [c["beta0"], c["tau"], c["raw_caux"], *c["z"], *c["lambda"]])
        write_floats(f"reference-{i}.bin", [point["lp"], *point["gradient"]])
    print(name, n, d, len(points))
