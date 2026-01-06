import os
import sys

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_FILE_PATH = "models/model.onnx"
ENGINE_FILE_PATH = "models/model.trt"


def build_engine(onnx_file_path):
    print(f"Reading ONNX: {onnx_file_path}")

    # 1. Setup Builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()

    # 2. Parse ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not os.path.exists(onnx_file_path):
        print(f"Error: {onnx_file_path} not found.")
        sys.exit(1)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 3. Optimize
    # We set the max workspace size (e.g., 1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Enable FP16 if supported (faster inference)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 4. Build Engine
    print("Building TensorRT Engine... (this may take a while)")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except AttributeError:
        # Fallback for older TensorRT versions
        engine = builder.build_engine(network, config)
        serialized_engine = engine.serialize()

    return serialized_engine


def main():
    if not os.path.exists("models"):
        os.makedirs("models")

    engine = build_engine(ONNX_FILE_PATH)

    if engine:
        with open(ENGINE_FILE_PATH, "wb") as f:
            f.write(engine)
        print(f"Success! TensorRT engine saved to {ENGINE_FILE_PATH}")
    else:
        print("Failed to build engine.")


if __name__ == "__main__":
    main()
