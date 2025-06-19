import os
import yaml
from PIL import Image

# saliency_vlm_core 폴더의 모듈들을 임포트하기 위해 경로 추가
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from saliency_vlm_core.image_processing import pre_normalize_image_size
from saliency_vlm_core.saliency import SaliencyMapper
from saliency_vlm_core.retriever import WikiRetriever  # ★★★ 새로 만든 WikiRetriever 임포트 ★★★
from saliency_vlm_core.llava_vqa import LLaVAVQA

def main():
    # 1. 설정 파일 로드
    print("--- 단계 1: 설정 파일 로드 ---")
    with open("configs/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("설정 로드 완료.\n")

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 2. WikiRetriever 객체 생성 (모델, 데이터 로딩) ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # 이 한 줄로 모델, 프로세서, FAISS 인덱스, 위키 데이터가 모두 로드됩니다.
    retriever = WikiRetriever(config)

    # 3. 이미지 처리 및 최종 쿼리 벡터 생성
    print("--- 단계 3: 이미지 처리 및 최종 쿼리 벡터 생성 ---")
    image_path = os.path.join(config['data_dir'], config['image_dir'], config['image_filename'])
    original_image = Image.open(image_path).convert("RGB")
    
    # 사전 정규화
    normalized_image = pre_normalize_image_size(original_image, target_size=config['pre_normalize_size'])
    
    # SaliencyMapper 객체 생성 및 실행
    # 이제 model, processor, device를 retriever 객체로부터 가져옵니다.
    saliency_mapper = SaliencyMapper(retriever.model, retriever.processor, retriever.device)
    final_image_to_encode = saliency_mapper.get_saliency_cropped_image(
        image=normalized_image,
        query=config['saliency_query'],
        window_size=config['window_size'],
        stride=config['stride'],
        keep_top_percent=config['keep_top_percent']
    )
    
    # ★★★ 최종 이미지를 retriever를 통해 인코딩 ★★★
    query_vector = retriever.encode_image(final_image_to_encode)
    print("최종 쿼리 벡터 생성 완료.\n")

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 4. Retriever를 통해 FAISS 검색 및 결과 출력 ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    print("--- 단계 4: 최종 벡터로 위키피디아 문서 검색 ---")
    results = retriever.search(query_vector, top_k=config['top_k'])
    

    # 가장 높은 유사도를 보이는 문서의 일부만 사용
    wiki_context = results[0]['text'] if results else ""

    # VQA 모델 초기화 및 질문 수행
    llava = LLaVAVQA(model_id=config.get('vlm_model_id', 'llava-hf/llava-1.5-7b-hf'))
    answer = llava.answer(final_image_to_encode, config.get('vqa_question', ''), wiki_context)
    print("VQA Answer:", answer)
    
    """    print("\n--- 최종 검색 결과 ---")
    if not results:
        print("검색 결과가 없습니다.")
    else:
        for res in results:
            print(f"🔍 순위 {res['rank']}: {res['title']}")
            print(f"✨ 유사도: {res['similarity']:.4f}")
            text_preview = res['text'].replace("\n", " ").strip()
            print(f"📖 내용 미리보기:\n{text_preview[:250]}...\n")
    """


if __name__ == "__main__":
    main()