#2_run_retrieval_test.py
import os
import yaml
from PIL import Image

# saliency_vlm_core 폴더의 모듈들을 임포트하기 위해 경로 추가
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from saliency_vlm_core.image_processing import pre_normalize_image_size
from saliency_vlm_core.saliency import SaliencyMapper
from saliency_vlm_core.retriever import WikiRetriever  
from saliency_vlm_core.llava_vqa import LLaVAVQA
from saliency_vlm_core.contriever_reranker import ContrieverReranker

# DEBUG 플래그: 출력 문구를 손쉽게 끄기 위한 변수
VERBOSE = True

def main():
    # 1. 설정 파일 로드
    print("--- 설정 파일 로드 ---")
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("설정 로드 완료.\n")

    # WikiRetriever 객체 생성 (모델, 데이터 로딩) 
    # 이 한 줄로 모델, 프로세서, FAISS 인덱스, 위키 데이터가 모두 로드됩니다.
    retriever = WikiRetriever(config)

    # 이미지 처리 및 최종 쿼리 벡터 생성
    print("--- 이미지 처리 및 최종 쿼리 벡터 생성 ---")
    image_path = os.path.join(
        config["data_dir"], config["image_dir"], config["image_filename"]
    )
    original_image = Image.open(image_path).convert("RGB")
    
    # 사전 정규화
    normalized_image = pre_normalize_image_size(
        original_image, target_size=config["pre_normalize_size"]
    )    
    # SaliencyMapper 객체 생성 및 실행
    # 이제 model, processor, device를 retriever 객체로부터 가져옵니다.
    saliency_mapper = SaliencyMapper(
        retriever.model, retriever.processor, retriever.device
    )
    final_image_to_encode = saliency_mapper.get_saliency_cropped_image(
        image=normalized_image,
        query=config["saliency_query"],
        window_size=config["window_size"],
        stride=config["stride"],
        keep_top_percent=config["keep_top_percent"],
    )
    
    # 최종 이미지를 retriever를 통해 인코딩
    query_vector = retriever.encode_image(final_image_to_encode)
    print("최종 쿼리 벡터 생성 완료.\n")

    # Retriever를 통해 FAISS 검색 및 결과 출력
    print("--- 최종 벡터로 위키피디아 문서 검색 ---")
    results = retriever.search(query_vector, top_k=config["top_k"])

    top_m = config.get("top_m", len(results))
    if VERBOSE:
        print(f"\n[DEBUG] Top-{top_m} 검색 결과")
        for res in results[:top_m]:
            print(
                f"🔍 순위 {res['rank']}: {res['title']} (유사도: {res['similarity']:.4f})"
            )
        print()

    # 검색 결과가 있는지 확인하고, 가장 유사도가 높은 문서의 제목을 'title'을 출력합니다.
    if results:
        top_result = results[0]
        print(f"가장 유사한 문서 제목을 찾았습니다: '{top_result['title']}'\n")
        
        #~~~~~~~~~~
        # 수정된 부분: 컨텍스트 길이 축약 로직 추가, 임시코드임
        
        # 1. 문서 전체 텍스트를 가져옵니다.
        full_text = top_result["text"]
        
        # 2. 설정 파일에서 최대 길이를 가져옵니다 (없으면 기본값 1000 사용).
        max_len = config.get("max_context_chars", 1000)
        
        # 3. 파이썬 문자열 슬라이싱을 이용해 앞에서부터 max_len 만큼만 잘라 사용합니다.
        wiki_context = full_text[:max_len]
        
        # 4. 사용자에게 어떻게 축약되었는지 알려주는 로그 출력
        print(f"컨텍스트를 {max_len}자로 축약하여 사용합니다:")
        print(f'"""\n{wiki_context}...\n"""\n')
    else:
        for res in results:
            print(f"🔍 순위 {res['rank']}: {res['title']} (유사도: {res['similarity']:.4f})")

    # Contriever를 사용한 문장 랭킹
    contriever = ContrieverReranker(device=retriever.device)
    top_sentence_k = config.get("top_m", 10)
    ranked_sentences = contriever.rank_sentences(
        config.get("vqa_question", ""), results, top_k=top_sentence_k
    )

    if VERBOSE:
        print(f"\n[DEBUG] 텍스트 질문과 가장 유사한 Top-{top_sentence_k} 문장")
        if not ranked_sentences:
            print("관련성 높은 문장을 찾을 수 없습니다.")
        else:
            for idx, s in enumerate(ranked_sentences):
                print(
                    f"🎯 문장 순위 {idx + 1} | 출처 문서: {s['source_title']} | 섹션: {s['section']} | 유사도: {s['similarity']:.4f}"
                )
                print(f"   -> {s['sentence']}\n")

    wiki_context = ranked_sentences[0]["sentence"] if ranked_sentences else ""


    # VQA 모델 초기화 및 질문 수행

    llava = LLaVAVQA(model_id=config.get("vlm_model_id", "llava-hf/llava-1.5-7b-hf"))
    '''
    answer = llava.answer(
        final_image_to_encode, config.get("vqa_question", ""), wiki_context
    )
    '''
    answer = llava.answer(
        original_image, config.get("vqa_question", ""), wiki_context
    )
    print("VQA Answer:", answer)


if __name__ == "__main__":
    main()