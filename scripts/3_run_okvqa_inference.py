import os
import json
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for missing PyYAML
    yaml = None
try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - missing dependency
    raise SystemExit(
        "The Pillow library is required to load images. Please install it via 'pip install Pillow'."
    ) from exc
import sys

# allow imports from repository root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from saliency_vlm_core import (
    pre_normalize_image_size,
    SaliencyMapper,
    WikiRetriever,
    LLaVAVQA,
    ContrieverReranker,
)


def load_okvqa(config):
    root = config["okvqa_root"]
    split = config.get("split", "val2014")

    # question and annotation paths
    q_path = os.path.join(
        root,
        "questions",
        f"OpenEnded_mscoco_{split}_questions.json",
    )
    a_path = os.path.join(
        root,
        "annotations",
        f"mscoco_{split}_annotations.json",
    )

    with open(q_path, "r") as f:
        questions = json.load(f)["questions"]

    with open(a_path, "r") as f:
        anns = json.load(f)["annotations"]
    # map question id -> annotation entry
    ann_map = {a["question_id"]: a for a in anns}
    return questions, ann_map, split, root


def build_image_path(root, split, image_id):
    fname = f"COCO_{split}_{image_id:012d}.jpg"
    return os.path.join(root, "images", split, fname)


def run_inference(question_entry, ann_entry, retriever, config, split, root):
    image_id = question_entry["image_id"]
    question = question_entry["question"]
    image_path = build_image_path(root, split, image_id)

    image = Image.open(image_path).convert("RGB")
    image = pre_normalize_image_size(image, config["pre_normalize_size"])

    saliency_mapper = SaliencyMapper(retriever.model, retriever.processor, retriever.device)
    cropped = saliency_mapper.get_saliency_cropped_image(
        image=image,
        query=question,
        window_size=config["window_size"],
        stride=config["stride"],
        keep_top_percent=config["keep_top_percent"],
    )
    query_vec = retriever.encode_image(cropped)
    results = retriever.search(query_vec, top_k=config["top_k"])

    contriever = ContrieverReranker(device=retriever.device)
    ranked = contriever.rank_sentences(question, results, top_k=config.get("top_m", 5))
    context = ranked[0]["sentence"] if ranked else ""

    llava = LLaVAVQA(model_id=config.get("vlm_model_id"))
    #answer = llava.answer(cropped, question, context)
    answer = llava.answer(image, question, context)


    gt_answers = [a["answer"] for a in ann_entry.get("answers", [])]

    print("질문:", question)
    print("이미지번호:", image_id)
    print("언어모델의대답:", answer)
    if gt_answers:
        print("실제정답:", ", ".join(gt_answers))
    else:
        print("실제정답:")
    print()


def _simple_yaml_load(path):
    data = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip().strip('"').strip("'")
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            data[key.strip()] = value
    return data


def main():
    if yaml is not None:
        with open("configs/okvqa_config.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        config = _simple_yaml_load("configs/okvqa_config.yaml")

    retriever = WikiRetriever(config)
    questions, ann_map, split, root = load_okvqa(config)
    start = config.get("start_index", 0)
    num = config.get("num_samples", 1)

    # You can change 'start_index' and 'num_samples' in the config
    # file to process only a slice of the dataset.

    for q in questions[start : start + num]:
        ann = ann_map.get(q["question_id"], {})
        run_inference(q, ann, retriever, config, split, root)


if __name__ == "__main__":
    main()