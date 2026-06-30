from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, BlipForConditionalGeneration, BlipProcessor


BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CROSS_ENCODER_MODEL_NAME = "roberta-base"
BI_ENCODER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TEXT_LEN = 32


class CrossEncoderScorer(nn.Module):
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL_NAME, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_embedding).squeeze(-1)


class BiEncoderHead(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CaptionSelectionPipeline:
    def __init__(self, models_dir: str = "models", device: str | None = None):
        self.models_dir = Path(models_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(self.device)
        self.blip_model.eval()

        self.cross_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL_NAME)
        self.cross_model = self._load_cross_encoder()

        self.sentence_model = SentenceTransformer(BI_ENCODER_MODEL_NAME, device=str(self.device))
        self.bi_model = self._load_bi_encoder()

    def _load_cross_encoder(self) -> CrossEncoderScorer | None:
        path = self.models_dir / "cross_encoder_scorer.pt"
        if not path.exists():
            return None

        model = CrossEncoderScorer(CROSS_ENCODER_MODEL_NAME).to(self.device)
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_bi_encoder(self) -> BiEncoderHead | None:
        path = self.models_dir / "bi_encoder_head.pt"
        if not path.exists():
            return None

        model = BiEncoderHead().to(self.device)
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def generate_captions(self, image: Image.Image, num_captions: int = 5) -> List[str]:
        image = image.convert("RGB")
        inputs = self.blip_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.blip_model.generate(
            **inputs,
            num_beams=max(5, num_captions),
            num_return_sequences=num_captions,
            max_new_tokens=30,
            early_stopping=True,
        )
        captions = self.blip_processor.batch_decode(outputs, skip_special_tokens=True)
        captions = [caption.strip() for caption in captions]
        return list(dict.fromkeys(captions))

    @torch.inference_mode()
    def score_cross_encoder(self, captions: List[str]) -> List[float]:
        if self.cross_model is None:
            return [float("nan")] * len(captions)

        encoded = self.cross_tokenizer(
            captions,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        scores = self.cross_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        return scores.detach().cpu().numpy().astype(float).tolist()

    @torch.inference_mode()
    def score_bi_encoder(self, captions: List[str]) -> List[float]:
        if self.bi_model is None:
            return [float("nan")] * len(captions)

        embeddings = self.sentence_model.encode(
            captions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        batch = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        scores = self.bi_model(batch)
        return scores.detach().cpu().numpy().astype(float).tolist()

    def score_consensus(self, captions: List[str]) -> List[float]:
        if len(captions) == 1:
            return [1.0]

        embeddings = self.sentence_model.encode(
            captions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sim_matrix = embeddings @ embeddings.T
        np.fill_diagonal(sim_matrix, np.nan)
        return np.nanmean(sim_matrix, axis=1).astype(float).tolist()

    def select_caption(self, image: Image.Image, method: str, num_captions: int = 5) -> Tuple[str, list[dict], str]:
        captions = self.generate_captions(image, num_captions=num_captions)
        if len(captions) == 0:
            return "", [], "BLIP не сгенерировал подписи."

        cross_scores = self.score_cross_encoder(captions)
        bi_scores = self.score_bi_encoder(captions)
        consensus_scores = self.score_consensus(captions)

        rows = []
        for index, caption in enumerate(captions, start=1):
            rows.append(
                {
                    "caption_number": index,
                    "caption": caption,
                    "cross_encoder_score": cross_scores[index - 1],
                    "bi_encoder_score": bi_scores[index - 1],
                    "consensus_score": consensus_scores[index - 1],
                }
            )

        score_map = {
            "cross_encoder": cross_scores,
            "bi_encoder": bi_scores,
            "consensus": consensus_scores,
            "first_blip": [1.0] + [0.0] * (len(captions) - 1),
        }

        chosen_method = method
        scores = score_map[chosen_method]
        if all(np.isnan(score) for score in scores):
            chosen_method = "consensus"
            scores = score_map[chosen_method]

        best_index = int(np.nanargmax(np.asarray(scores, dtype=float)))
        rows[best_index]["selected"] = True
        for index, row in enumerate(rows):
            row.setdefault("selected", index == best_index)

        warning = ""
        if method == "cross_encoder" and self.cross_model is None:
            warning = "Файл models/cross_encoder_scorer.pt не найден. Использован consensus."
        if method == "bi_encoder" and self.bi_model is None:
            warning = "Файл models/bi_encoder_head.pt не найден. Использован consensus."

        return captions[best_index], rows, warning
