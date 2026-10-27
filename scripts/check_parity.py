"""Compare the port with the segmentation subgraph stored by the authors."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from mouse_sleep.config import Settings, MODEL_STEPS, ROOT, atomic_json
from mouse_sleep.segmentation import Segmenter

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "verification")
    args = parser.parse_args(argv)
    results = []
    for model_name in MODEL_STEPS:
        checkpoint = str(Settings(model=model_name).checkpoint)
        reader = tf.train.load_checkpoint(checkpoint)
        meta = tf.compat.v1.MetaGraphDef()
        meta.ParseFromString(Path(checkpoint + ".meta").read_bytes())
        graph_def = meta.graph_def
        first = next(node for node in graph_def.node if node.op == "Conv2D" and node.name.startswith("Network/SegmentEncoder/Conv/"))
        # Use the same standardized input for both graphs, bypassing the saved
        # graph's fixed-size training input queue and its 50-image map operation.
        first.input[0] = "parity_input"
        node = graph_def.node.add()
        node.name, node.op = "parity_input", "Placeholder"
        node.attr["dtype"].type = tf.float32.as_datatype_enum
        node.attr["shape"].shape.CopyFrom(tf.TensorShape([1, 480, 480, 1]).as_proto())
        graph_def = tf.compat.v1.graph_util.extract_sub_graph(graph_def, ["Network/SegmentDecoder/seg/Relu"])
        for node in graph_def.node:
            node.attr.pop("_output_shapes", None)
            if node.op == "Const":
                value = tf.make_ndarray(node.attr["value"].tensor)
                if value.shape == (4,) and value[0] == 50 and "SegmentDecoder" in node.name:
                    value[0] = 1
                    node.attr["value"].tensor.CopyFrom(tf.make_tensor_proto(value))
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
            reference = tf.nn.softmax(graph.get_tensor_by_name("Network/SegmentDecoder/seg/Relu:0"), axis=-1)[..., 0]
            restores = [tf.compat.v1.assign(graph.get_tensor_by_name(node.name + ":0"), reader.get_tensor(node.name))
                        for node in graph_def.node if node.op == "VariableV2"]
        rng = np.random.default_rng(42)
        image = rng.normal(130, 12, (1, 480, 480, 1)).astype(np.float32)
        image[:, 150:260, 200:270] = 15
        with Segmenter(checkpoint) as port:
            centered = image - image.mean()
            standardized = centered / max(image.std(), 1 / np.sqrt(image.size))
            predicted = port.session.run(port.probabilities, {port.inputs: image})
        with tf.compat.v1.Session(graph=graph) as session:
            session.run(restores)
            feed = {graph.get_tensor_by_name("parity_input:0"): standardized}
            for operation in graph.get_operations():
                if operation.type == "Placeholder" and operation.name != "parity_input":
                    feed[operation.outputs[0]] = False
            expected = session.run(reference, feed)
        error = float(np.max(np.abs(expected - predicted)))
        np.testing.assert_allclose(predicted, expected, atol=3e-4, rtol=3e-4)
        results.append({"model": model_name, "maximum_probability_difference": error, "passed": True})
        print(json.dumps(results[-1]), flush=True)
    atomic_json(args.output / "checkpoint_parity.json", {"checks": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
