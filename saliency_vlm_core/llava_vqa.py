import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

class LLaVAVQA:
    """Simple wrapper around the LLaVA-7B model for VQA."""

    def __init__(self, model_id: str = "llava-hf/llava-v1.5-7b", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id).to(self.device)

    def answer(self, image: Image.Image, question: str, context: str) -> str:
        prompt = question
        if context:
            prompt += f"\n{context}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        return self.processor.decode(outputs[0], skip_special_tokens=True).strip()