"""Fallback streaming saver for large ONNX models.

If torch.onnx.export's program.save() OOMs, use this to save the
ONNXProgram's model_proto with tensors streamed to external data
without holding duplicate copies in memory.

Usage (in the export process, after program is built):
    from save_streaming import save_program_streaming
    save_program_streaming(program, "export/model_fp16.onnx")
"""
import os


def save_program_streaming(program, path):
    import onnx

    model_proto = program.model_proto
    data_path = path + ".data"

    # Stream each initializer's raw bytes to the .data file, then replace
    # the tensor's data with an external reference. This avoids holding
    # both the in-memory tensor and the serialized copy simultaneously.
    offset = 0
    with open(data_path, "wb") as f:
        for tensor in model_proto.graph.initializer:
            # Get raw bytes (convert from raw_data or from float_data etc.)
            if tensor.HasField("raw_data"):
                raw = tensor.raw_data
            else:
                # Fallback: use onnx.numpy_helper (small tensors only)
                import numpy as np
                from onnx import numpy_helper
                raw = numpy_helper.to_array(tensor).tobytes()

            f.write(raw)
            length = len(raw)

            # Replace with external reference
            tensor.ClearField("raw_data")
            # Clear any non-raw data fields to be safe
            for field in ("float_data", "double_data", "int32_data",
                          "int64_data", "uint64_data", "string_data"):
                if tensor.HasField(field):
                    tensor.ClearField(field)
            ext = tensor.external_data.add()
            ext.key = "location"
            ext.value = os.path.basename(data_path)
            ext = tensor.external_data.add()
            ext.key = "offset"
            ext.value = str(offset)
            ext = tensor.external_data.add()
            ext.key = "length"
            ext.value = str(length)
            tensor.data_location = 1  # EXTERNAL

            offset += length
            # Free the bytes immediately
            del raw

    # Now the proto is small (graph only); save it.
    # Use save_model with external data already set.
    with open(path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"streamed save complete: {path} (+ {data_path}, {offset} bytes)")
