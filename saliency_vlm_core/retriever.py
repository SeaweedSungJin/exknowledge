import os
import torch
import faiss
import pickle
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class WikiRetriever:
    """
    CLIP 모델과 FAISS 인덱스를 로드하고 관리하며,
    주어진 이미지나 벡터로 위키피디아 문서를 검색하는 클래스.
    """
    def __init__(self, config: dict):
        """
        클래스 초기화 시 모델, 프로세서, FAISS 인덱스, 위키 데이터를 로드합니다.
        
        Args:
            config (dict): config.yaml 파일에서 로드된 설정 딕셔너리.
        """
        print("--- WikiRetriever 초기화 시작 ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config

        # 1. CLIP 모델 및 프로세서 로드
        print(f"CLIP 모델 로딩: {self.config['model_id']}")
        self.model = CLIPModel.from_pretrained(self.config['model_id']).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.config['model_id'])
        self.model.eval()

        # 2. FAISS 인덱스 로드
        faiss_path = os.path.join(self.config['data_dir'], "wikipedia_preprocessed", self.config['faiss_index_name'])
        print(f"FAISS 인덱스 로딩: {faiss_path}")
        self.index = faiss.read_index(faiss_path)

        # 3. 위키피디아 데이터(제목, 본문) 로드
        data_path = os.path.join(self.config['data_dir'], "wikipedia_preprocessed", self.config['wiki_data_name'])
        print(f"위키피디아 데이터 로딩: {data_path}")
        with open(data_path, "rb") as f_in:
            wiki_data = pickle.load(f_in)
        self.titles = wiki_data["titles"]
        self.texts = wiki_data["texts"]
        
        print("--- WikiRetriever 초기화 완료 ---\n")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        주어진 PIL 이미지를 인코딩하여 정규화된 NumPy 벡터를 반환합니다.
        """
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().squeeze()

    def search(self, query_vector: np.ndarray, top_k: int) -> list:
        """
        주어진 쿼리 벡터로 FAISS 인덱스를 검색하여 상위 K개의 결과를 반환합니다.
        """
        query_vector = np.array([query_vector]).astype("float32")
        similarities, indices = self.index.search(query_vector, top_k)
        
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                "rank": rank + 1,
                "title": self.titles[idx],
                "similarity": float(similarities[0][rank]),
                "text": self.texts[idx]
            })
        return results