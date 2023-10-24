from typing import Dict, Any, NoReturn, List
import json
import base64
from io import BytesIO
import codecs

import PIL
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np

import triton_python_backend_utils as pb_utils

from utils_data import postprocessed


class TritonPythonModel:

    def initialize(self, args:Dict[str, Any]) -> NoReturn:
        parameters = json.loads(args['model_config'])['parameters']
        base_params = parameters['base_params']
        path_model = base_params['string_value']
        self.processor = TrOCRProcessor.from_pretrained(path_model)
        self.model = VisionEncoderDecoderModel.from_pretrained(path_model)
    
    def execute(self, requests):
        responses = []
        for request in requests:
            image_base64_out = pb_utils.get_input_tensor_by_name(request, "image_base64")
            image_base64_str = image_base64_out.as_numpy().item()
            image_decoded = base64.b64decode((image_base64_str))
            image_bytes = BytesIO(image_decoded)
            image = PIL.Image.open(image_bytes).convert("RGB")
            pixel_values = self.processor(image, return_tensors='pt').pixel_values
            param = next(self.model.parameters())
            pixel_values = pixel_values.to(param)
            with torch.inference_mode():
                output = self.model.generate(pixel_values)
                preds = postprocessed(output, self.processor)
                np_preds = np.asarray([preds], dtype=object)
            output = pb_utils.InferenceResponse(
                pb_utils.Tensor("output__0", np_preds)
            )
            responses.append(output)
        return responses

