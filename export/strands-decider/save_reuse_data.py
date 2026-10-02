"""Saver that reuses a pre-streamed .data file.

After torch.onnx.export returns `program` and the torch model is deleted:
    from save_reuse_data import save_reuse_data
    save_reuse_data(program, "export/model_fp16.onnx", "offsets.json")

This gets initializer metadata (name/dtype/shape), drops the tensor data from
the IR (freeing 3.77GB), serializes the weightless graph, and attaches
external-data references using the precomputed offsets from offsets.json.
The .data file must already exist with tensors in the same order.
"""
import os
import json
import gc


def save_reuse_data(program, path, offsets_path):
    import onnx
    import numpy as np
    from onnxscript.ir import serde

    with open(offsets_path) as f:
        offsets = json.load(f)
    # Map name -> (offset, size)
    offmap = {o["name"]: (o["offset"], o["size"]) for o in offsets}
    print(f"loaded {len(offmap)} offsets", flush=True)

    data_path = path + ".data"
    data_basename = os.path.basename(data_path)
    assert os.path.exists(data_path), f"missing {data_path}"
    model = program.model
    initializers = model.graph.initializers

    # Collect metadata and drop tensors from IR (recursively, including
    # subgraphs like the while_loop body).
    def _clear_graph(graph, prefix=""):
        names = list(graph.initializers.keys())
        for name in names:
            init = graph.initializers[name]
            full = f"{prefix}{name}"
            # dtype/shape from IR
            ir_dtype = init.dtype
            ir_shape = tuple(init.shape)
            dtype_str = str(ir_dtype).split(".")[-1]
            _map = {"FLOAT16": "float16", "FLOAT": "float32", "DOUBLE": "float64",
                    "INT64": "int64", "INT32": "int32", "BOOL": "bool",
                    "UINT8": "uint8", "INT8": "int8"}
            dtype = np.dtype(_map.get(dtype_str, "float32"))
            if full in offmap:
                offset, size = offmap[full]
                infos.append((full, offset, size, str(dtype), ir_shape))
            # Drop the tensor
            init.const_value = None
            del graph.initializers[name]
            del init
        # Recurse into node attributes that contain graphs
        for node in graph:
            for attr in node.attributes.values():
                if hasattr(attr, "as_graph"):
                    try:
                        g = attr.as_graph()
                        _clear_graph(g, prefix=f"{prefix}{node.name}/")
                    except Exception:
                        pass

    infos = []
    print(f"collecting metadata (recursive)...", flush=True)
    _clear_graph(model.graph)
    print(f"collected {len(infos)} initializers; tensors dropped", flush=True)
    gc.collect()

    # Serialize the weightless graph
    proto = serde.serialize_model(model)
    print(f"graph serialized: {len(proto.graph.node)} nodes", flush=True)
    gc.collect()

    # Attach external-data initializers
    for name, off, size, dtype_str, shape in infos:
        tensor = proto.graph.initializer.add()
        tensor.name = name
        tensor.data_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype_str))
        tensor.dims.extend(shape)
        tensor.data_location = onnx.TensorProto.EXTERNAL
        ext = tensor.external_data.add()
        ext.key = "location"
        ext.value = data_basename
        ext = tensor.external_data.add()
        ext.key = "offset"
        ext.value = str(off)
        ext = tensor.external_data.add()
        ext.key = "length"
        ext.value = str(size)

    # Save using onnx.save_model (standard API, handles large protos).
    # The proto has external-data initializers; the .data file already exists.
    onnx.save_model(proto, path)
    print(f"done: {path} (reuses {data_path})", flush=True)
