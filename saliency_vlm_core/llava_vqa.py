import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

"""Simple wrapper around the LLaVA-7B model for VQA."""

class LLaVAVQA:
    def __init__(self, model_id: str = "llava-hf/llava-v1.5-7b-hf", device: str | None = None, use_quantization: bool = False):
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
            torch_dtype=torch.float16, # 양자화 사용 시에도 torch_dtype 명시 권장
            device_map="auto" if quantization_config else None
        ).to(self.device if not quantization_config else None)


# 예시: generate 파라미터 유연성 확보
# LLaVAVQA 클래스 내부의 answer 함수

    def answer(self, image: Image.Image, question: str, context: str, **generation_kwargs) -> str:
        # --- Start of Level 3 Few-Shot Prompt ---

        # 1. [예시 제공] 원하는 출력 형식(단답형)을 학습시키기 위한 예제들을 정의합니다.
        # 이 예시들은 모델에게 답변의 '패턴'을 알려주는 역할을 합니다.
        # 실제 이미지를 필요로 하지 않으므로 <image> 토큰은 사용하지 않습니다.
        few_shot_examples = """USER:
    Question: What is the primary activity shown in the picture?
    Answer: A person reading a book.

    USER:
    Question: Identify the main subject.
    Answer: A brown cat sleeping on a sofa.

    """  # 예시와 실제 질문을 구분하기 위해 끝에 빈 줄을 추가합니다.

        # 2. [프롬프트 최종 조합] 예시 뒤에 실제 질문을 붙여 전체 프롬프트를 구성합니다.
        # 예시와 일관성을 맞추기 위해 'ASSISTANT:' 대신 'Answer:'를 사용합니다.
        prompt = few_shot_examples
        prompt += f"USER: <image>\nQuestion: {question}"

        if context:
            # 컨텍스트도 질문의 일부로 포함시켜줍니다.
            prompt += f"\n{context}"

        prompt += "\nAnswer:"

        # --- End of Level 3 Few-Shot Prompt ---


        # --- (이하 코드는 이전과 동일) 모델 추론 및 답변 정제 로직 ---
        # 참고: 프롬프트가 길어졌으므로, 모델의 최대 컨텍스트 길이를 넘지 않도록 주의해야 합니다.

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # 답변 부분만 잘라내기 위해 입력 토큰 길이를 미리 계산합니다.
        input_token_len = inputs.input_ids.shape[1]

        # 단답형 답변을 원하므로 max_new_tokens를 이전보다 줄여도 좋습니다. (예: 64)
        default_generation_kwargs = {"max_new_tokens": 64}
        default_generation_kwargs.update(generation_kwargs)

        outputs = self.model.generate(**inputs, **default_generation_kwargs)

        # 새로 생성된 토큰만 잘라냅니다.
        new_tokens = outputs[0, input_token_len:]

        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()