import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

"""Simple wrapper around the LLaVA-7B model for VQA."""

class LLaVAVQA:
    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.5-7b-hf",
        device: str | None = None,
        use_quantization: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)

        quantization_config = None
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float16
            )

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,  # 양자화 사용 시에도 torch_dtype 명시 권장
            device_map="auto" if quantization_config else None,
        ).to(self.device if not quantization_config else None)


# 예시: generate 파라미터 유연성 확보
# LLaVAVQA 클래스 내부의 answer 함수

    def answer(
        self, image: Image.Image, question: str, context: str, **generation_kwargs
    ) -> str:        
        """
        # 질문의 의도를 명확히 하는 단순 프롬프트로 
        prompt = (
            f"USER: <image>\n"
            f"answer the following question: {question}"
        )
        if context:
            prompt += f"\n{context}"
        # 모델의 역할을 '비서'로자유로운 답변을 유도
        prompt += "\nASSISTANT:"
"""
        prompt = (
            f"<image>\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        # 참고: 프롬프트가 길어졌으므로, 모델의 최대 컨텍스트 길이를 넘지 않도록 주의해야 합니다.

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device
        )
        # 답변 부분만 잘라내기 위해 입력 토큰 길이를 미리 계산합니다.
        input_token_len = inputs.input_ids.shape[1]

        # 단답형 답변을 원하므로 max_new_tokens를 이전보다 줄여도 좋습니다. (예: 64)
        default_generation_kwargs = {"max_new_tokens": 128}
        default_generation_kwargs.update(generation_kwargs)

        outputs = self.model.generate(**inputs, **default_generation_kwargs)

                # 모델을 통해 답변 ID 생성
        generate_ids = self.model.generate(**inputs, **default_generation_kwargs)

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★ 여기에 디버깅 코드 블록을 추가합니다 ★★★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        print("\n--- LLaVA_VQA Debug Info ---")
        
        # 1. 프로세서에 전달된 최종 프롬프트 확인
        print(f"\n[Prompt Sent to Processor]:\n---\n{prompt}\n---")
        
        # 2. 모델이 생성한 '날 것'의 토큰 ID 전체를 확인
        #print(f"\n[Raw Generated IDs (Prompt + Answer)]:\n{generate_ids}")
        
        # 3. 특수 토큰(<s>, </s>)을 포함하여 전체 생성 내용을 디코딩
        # 이 부분을 통해 모델이 답변 없이 바로 문장 종료(</s>)를 하는지 확인할 수 있습니다.
        full_decoded_text = self.processor.batch_decode(generate_ids, skip_special_tokens=False)[0]
        #print(f"\n[Full Decoded Output (with special tokens)]:\n---\n{full_decoded_text}\n---")
        
        print("--------------------------\n")

        # 새로 생성된 토큰만 잘라냅니다.
        new_tokens = outputs[0, input_token_len:]

        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()