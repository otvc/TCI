from typing import List
import re

from transformers import TrOCRProcessor
import torch

def postprocessed(output: torch.Tensor, processor: TrOCRProcessor) -> List[str]:
    """
    Clean texts in predictions;

    Args:
        output (torch.Tensor): predictions;
        processor (TrOCRProcessor): processor for TrOCR.

    Returns:
        List[str]: cleaned predictions;
    """
    preds = processor.batch_decode(output)
    preds_postprocessed = list(map(lambda x: re.sub(r'\D', '', x), preds))
    return preds_postprocessed
