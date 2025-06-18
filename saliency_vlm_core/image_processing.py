
from PIL import Image
from torchvision.transforms import functional as TF

def pre_normalize_image_size(image: Image.Image, target_size: int) -> Image.Image:
    """
    이미지의 가로세로 비율을 유지하면서, 짧은 쪽의 길이를 target_size에 맞게 조절합니다.
    """
    w, h = image.size
    if h < w:
        new_h = target_size
        ratio = new_h / h
        new_w = int(w * ratio)
    else:
        new_w = target_size
        ratio = new_w / w
        new_h = int(h * ratio)
    
    #resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_image = image.resize((new_w, new_h), Image.LANCZOS)

    return resized_image

def add_auto_padding(image: Image.Image, window_size: int, stride: int) -> tuple[Image.Image, tuple[int, int]]:
    """
    이미지에 슬라이딩 윈도우를 적용할 때 가장자리에 남는 픽셀이 없도록,
    오른쪽과 아래쪽에 '반사' 패딩을 자동으로 추가합니다.
    """
    W, H = image.size
    pad_w, pad_h = 0, 0
    if W < window_size:
        pad_w = window_size - W
    else:
        rem_w = (W - window_size) % stride
        if rem_w != 0:
            pad_w = stride - rem_w

    if H < window_size:
        pad_h = window_size - H
    else:
        rem_h = (H - window_size) % stride
        if rem_h != 0:
            pad_h = stride - rem_h

    if pad_w > 0 or pad_h > 0:
        padded_image = TF.pad(image, padding=(0, 0, pad_w, pad_h), padding_mode='reflect')
    else:
        padded_image = image
        
    return padded_image, (pad_w, pad_h)