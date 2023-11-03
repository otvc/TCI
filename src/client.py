from typing import Dict, Any, NoReturn
import json
from pathlib import Path
import sys
import os
import argparse
import logging

current_parent = Path(__file__).resolve().parent
sys.path.append(str(current_parent))

import numpy as np
from tritonclient import http

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('dag_creator')
logger.setLevel(logging.DEBUG)

def load_json(path_file: str) -> Dict[str, Any]:
    with open(path_file, 'r') as f:
        json_object = json.load(f)
    return json_object


def inference_test(url: str = 'localhost:8000',
                   model_name = 'add_sub') -> NoReturn:
    triton_client = http.InferenceServerClient(url=url)
    path_example = 'test_example.json'
    test_example = load_json(path_example)
    logger.info("Example is loaded.")
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

    logger.info(f"Send message to {url} on model with name {model_name}")
    result = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
    )
    prediction = result.as_numpy("output__0").item()
    print(f'Output from model {prediction}')

    triton_client.close()



if __name__ == '__main__':
    logger.info("Client is initialized.")
    address = os.environ['ADDRESS']
    model_name = os.environ['MODEL_NAME']
    inference_test(address, model_name)
