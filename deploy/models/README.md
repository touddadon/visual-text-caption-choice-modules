# Model files

Сюда кладутся веса моделей:

- `cross_encoder_scorer.pt` — текстовый cross-encoder скорер на базе `roberta-base`.
- `bi_encoder_head.pt` — голова bi-encoder поверх `sentence-transformers/all-MiniLM-L6-v2`.

Без этих файлов приложение всё равно запускается, но при выборе `cross_encoder` или `bi_encoder` будет использовать fallback `consensus`.
