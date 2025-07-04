import re
import nltk
import torch
from sentence_transformers import SentenceTransformer, util

class ContrieverReranker:
    """Rank sentences from documents using Contriever."""

    def __init__(self, model_name: str = "facebook/contriever-msmarco", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
       # Ensure NLTK tokenizers are available.  Newer NLTK versions require
        # both ``punkt`` and ``punkt_tab`` so we download them if missing.
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:  # pragma: no cover - local environment setup
            nltk.download("punkt")
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:  # pragma: no cover - local environment setup
            nltk.download("punkt_tab")

    @staticmethod
    def _extract_sentences_with_section(text: str):
        sections = []
        current_title = ""
        buffer = []
        for line in text.splitlines():
            match = re.match(r"={2,}\s*(.+?)\s*={2,}", line)
            if match:
                if buffer:
                    sections.append((current_title, "\n".join(buffer)))
                    buffer = []
                current_title = match.group(1).strip()
            else:
                buffer.append(line)
        if buffer:
            sections.append((current_title, "\n".join(buffer)))

        sentences_data = []
        for title, sec_text in sections:
            sentences = nltk.sent_tokenize(sec_text)
            for sent in sentences:
                stripped = sent.strip()
                if stripped:
                    sentences_data.append({"sentence": stripped, "section": title})
        return sentences_data

    def rank_sentences(self, query: str, docs: list[dict], top_k: int = 5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        all_sentences = []
        for doc in docs:
            for item in self._extract_sentences_with_section(doc["text"]):
                item["source_title"] = doc["title"]
                all_sentences.append(item)

        if not all_sentences:
            return []

        sentences = [s["sentence"] for s in all_sentences]
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
        for s, score in zip(all_sentences, cosine_scores):
            s["similarity"] = score.item()

        sorted_sents = sorted(all_sentences, key=lambda x: x["similarity"], reverse=True)
        return sorted_sents[:top_k]