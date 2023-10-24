from typing import Dict, Any, NoReturn
import json
from pathlib import Path
import sys

current_parent = Path(__file__).resolve().parent
sys.path.append(str(current_parent))

import numpy as np
from tritonclient import http

URL = 'localhost:8000'
MODEL_NAME = 'add_sub'

def load_json(path_file: str) -> Dict[str, Any]:
    with open(path_file, 'r') as f:
        json_object = json.load(f)
    return json_object


def inference_test() -> NoReturn:
    triton_client = http.InferenceServerClient(url=URL)
    path_example = 'test_example.json'
    test_example = load_json(path_example)
    base64str = test_example['base64str']
    base64np = np.asarray([base64str], dtype=object)

    inputs, outputs = [], []
    infer_input = http.InferInput(
        "image_base64",
        [1],
        "BYTES"
    )
    infer_input.set_data_from_numpy(base64np.reshape([1]), binary_data=False)
    inputs.append(infer_input)

    infer_output = http.InferRequestedOutput(
        "output__0",
        binary_data=False
    )
    outputs.append(infer_output)

    result = triton_client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )
    prediction = result.as_numpy("output__0").item()
    print(f'Output from model {prediction}')

    triton_client.close()



if __name__ == '__main__':
    inference_test()