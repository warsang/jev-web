"""Memory-efficient saver: graph without weights, then stream weights to .data.

Usage (in the export process, after program is built and torch model freed):
    from save_graph_first import save_graph_then_weights
    save_graph_then_weights(program, "export/model_fp16.onnx")
"""
import os
import gc


def save_graph_then_weights(program, path):
    import onnx
    import numpy as np

    data_path = path + ".data"
    data_basename = os.path.basename(data_path)

    # Step 1: Save the graph WITHOUT initializers (small, ~200MB).
    # This does not serialize the 3.77GB of weights.
    print("saving graph without initializers...", flush=True)
    program.save(path, include_initializers=False)
    print("graph saved", flush=True)
    gc.collect()

    # Step 2: Stream each initializer's bytes to the .data file, recording
    # name/offset/length/dtype/shape. Only one tensor in flight at a time.
    print("streaming weights to external data...", flush=True)
    infos = []  # (name, offset, length, dtype, shape)
    offset = 0
    initializers = program.model.graph.initializers
    n = len(initializers)
    with open(data_path, "wb") as f:
        for idx, (name, init) in enumerate(initializers.items()):
            # Get the tensor as numpy (zero-copy where possible)
            t = init.const_value
            # t is a TorchTensor or similar; convert to numpy
            if hasattr(t, "numpy"):
                arr = t.numpy()
            else:
                import torch
                arr = np.asarray(t)
            raw = arr.tobytes()
            f.write(raw)
            length = len(raw)
            infos.append((name, offset, length, arr.dtype, arr.shape))
            offset += length
            del arr, raw, t
            if idx % 50 == 0:
                print(f"  {idx}/{n} ...", flush=True)
                gc.collect()
    print(f"weights streamed: {offset} bytes", flush=True)
    gc.collect()

    # Step 3: Reload the graph, attach external-data initializers, save.
    print("attaching external initializers...", flush=True)
    model = onnx.load(path, load_external_data=False)
    for name, off, length, dtype, shape in infos:
        tensor = onnx.TensorProto()
        tensor.name = name
        tensor.data_type = onnx.helper.np_dtype_to_tensor_dtype(dtype)
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
        ext.value = str(length)
        model.graph.initializer.append(tensor)
    # Save the final model (graph + external refs, small)
    onnx.save_model(model, path)
    print(f"done: {path} (+ {data_path})", flush=True)
