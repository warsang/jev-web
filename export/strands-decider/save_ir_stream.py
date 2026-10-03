"""Stream tensors out of an ONNXProgram's IR model one-by-one, freeing memory.

After torch.onnx.export returns `program` and the torch model is deleted:
    from save_ir_stream import save_ir_streaming
    save_ir_streaming(program, "export/model_fp16.onnx")

This writes each initializer's bytes to the .data file and DROPS the tensor
from the IR immediately, so peak memory stays low. Finally the (now small)
graph is serialized and saved with external-data references.
"""
import os
import gc


def save_ir_streaming(program, path):
    import onnx
    import numpy as np
    from onnxscript import ir

    data_path = path + ".data"
    data_basename = os.path.basename(data_path)
    model = program.model  # ir.Model
    initializers = model.graph.initializers

    # Step 1: record (name, dtype, shape) and stream bytes to .data,
    # dropping each tensor from the IR right after writing.
    infos = []
    offset = 0
    names = list(initializers.keys())
    n = len(names)
    print(f"streaming {n} initializers...", flush=True)
    with open(data_path, "wb") as f:
        for idx, name in enumerate(names):
            init = initializers[name]
            t = init.const_value
            # Get nbytes without materializing (TorchTensor.nbytes property)
            try:
                nbytes = int(t.nbytes)
            except Exception:
                nbytes = -1
            print(f"  [{idx}/{n}] {name}: {nbytes} bytes", flush=True)
            # Get dtype/shape from the IR initializer metadata
            # (SymbolicTensor has dtype/shape properties).
            try:
                ir_dtype = init.dtype
                ir_shape = tuple(init.shape)
                dtype_str = str(ir_dtype).split(".")[-1]
                _map = {"FLOAT16": "float16", "FLOAT": "float32", "DOUBLE": "float64",
                        "INT64": "int64", "INT32": "int32", "BOOL": "bool",
                        "UINT8": "uint8", "INT8": "int8"}
                dtype = np.dtype(_map.get(dtype_str, "float32"))
            except Exception:
                arr0 = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
                dtype, ir_shape = arr0.dtype, tuple(arr0.shape)
                del arr0
            shape = ir_shape
            # Write bytes directly via TorchTensor.tofile() (avoids a Python
            # bytes copy); fall back to tobytes().
            if hasattr(t, "tofile"):
                t.tofile(f)
                length = nbytes if nbytes > 0 else 0
            elif hasattr(t, "tobytes"):
                raw = t.tobytes()
                f.write(raw)
                length = len(raw)
                del raw
            else:
                arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
                raw = arr.tobytes()
                f.write(raw)
                length = len(raw)
                del arr, raw
            infos.append((name, offset, length, str(dtype), shape))
            offset += length
            # Free: drop the tensor value and remove the initializer entry
            del t
            init.const_value = None
            del initializers[name]
            del init
            if idx % 50 == 0:
                print(f"  {idx}/{n} ...", flush=True)
            gc.collect()
    print(f"weights streamed: {offset} bytes; IR now weightless", flush=True)
    gc.collect()

    # Step 2: serialize the (small) weightless IR model to a ModelProto.
    from onnxscript.ir import serde
    proto = serde.serialize_model(model)
    print(f"graph serialized: {len(proto.graph.node)} nodes", flush=True)

    # Step 3: attach external-data initializers.
    for name, off, length, dtype_str, shape in infos:
        tensor = proto.graph.initializer.add()
        tensor.name = name
        # Map numpy dtype string to ONNX TensorProto.DataType
        dtype = np.dtype(dtype_str)
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

    # Step 4: save the (small) proto using SerializeToFileDescriptor to avoid
    # creating a huge Python bytes object.
    import io
    with open(path, "wb") as f:
        # Get the file descriptor and serialize directly to it
        fd = f.fileno()
        # protobuf's SerializeToFileDescriptor writes without a Python copy
        proto.SerializeToFileDescriptor(fd)
    print(f"done: {path} (+ {data_path})", flush=True)
