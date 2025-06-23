import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
from .image_processing import add_auto_padding # 같은 폴더의 image_processing.py에서 함수 임포트

class SaliencyMapper:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def get_saliency_cropped_image(
        self,
        image: Image.Image,
        query: str,
        window_size: int,
        stride: int,
        keep_top_percent: int,
    ) -> Image.Image:
        """
        주어진 이미지와 텍스트 쿼리를 기반으로 중요 영역을 찾아 잘라낸 이미지를 반환합니다.
        """
        print("--- 중요 영역 탐지 시작 ---")
        
        # 1. 텍스트 쿼리 인코딩
        with torch.no_grad():
            text_inputs = self.processor(
                text=[query], return_tensors="pt", padding=True
            ).to(self.device)
            text_feat = self.model.get_text_features(**text_inputs).float()
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

        # 2. 패딩 및 텐서 변환
        padded_image, _ = add_auto_padding(image, window_size, stride)
        image_tensor = ToTensor()(padded_image)
        H, W = image_tensor.shape[1:]

        # 3. 슬라이딩 윈도우로 유사도 계산
        patch_similarities = []
        for y in tqdm(range(0, H - window_size + 1, stride), desc="Analyzing Patches"):
            for x in range(0, W - window_size + 1, stride):
                patch = image_tensor[:, y : y + window_size, x : x + window_size]                
                patch_pil = ToPILImage()(patch)
                with torch.no_grad():
                    inputs = self.processor(images=[patch_pil], return_tensors="pt").to(
                        self.device
                    )                    
                    image_feat = self.model.get_image_features(**inputs).float()
                    image_feat_norm = image_feat / image_feat.norm(dim=-1, keepdim=True)
                    sim = (image_feat_norm @ text_feat.T).item()
                patch_similarities.append(
                    {
                        "similarity": sim,
                        "center": (x + window_size // 2, y + window_size // 2),
                    }
                )

        # 4. 히트맵 및 마스크 생성
        heatmap = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        for p in patch_similarities:
            cx, cy = p["center"]
            x_start, y_start = cx - window_size // 2, cy - window_size // 2
            heatmap[
                y_start : y_start + window_size, x_start : x_start + window_size
            ] += p["similarity"]
            count_map[
                y_start : y_start + window_size, x_start : x_start + window_size
            ] += 1

        heatmap /= count_map + 1e-6
        percentile_value = 100 - keep_top_percent
        threshold = np.percentile(heatmap, percentile_value)
        padded_mask = (heatmap >= threshold).astype(np.uint8)
        
        # 5. 마스크 기반 이미지 자르기
        H_norm, W_norm = image.height, image.width
        mask = padded_mask[:H_norm, :W_norm]
        
        np_img = np.array(image)
        white_bg = np.ones_like(np_img) * 255
        mask_3ch = np.stack([mask] * 3, axis=-1)
        masked_np = np.where(mask_3ch == 1, np_img, white_bg)
        masked_image_pil = Image.fromarray(masked_np)
        
        rows, cols = np.where(mask == 1)
        if rows.size > 0:
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            final_image = masked_image_pil.crop((x_min, y_min, x_max + 1, y_max + 1))
            print(
                f"중요 영역 BBox:({x_min}, {y_min}, {x_max}, {y_max}) 로 이미지를 잘랐습니다."
            )        
        else:
            final_image = image
            print("경고: 중요 영역을 찾지 못해 원본 이미지를 그대로 사용합니다.")
            
        return final_image