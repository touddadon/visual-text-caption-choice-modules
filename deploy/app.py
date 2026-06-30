from functools import lru_cache

import gradio as gr
import pandas as pd
from PIL import Image

from modeling import CaptionSelectionPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> CaptionSelectionPipeline:
    return CaptionSelectionPipeline(models_dir="models")


def run_caption_selection(image: Image.Image, method: str, num_captions: int):
    if image is None:
        return "Загрузите изображение.", pd.DataFrame(), ""

    pipeline = get_pipeline()
    best_caption, rows, warning = pipeline.select_caption(
        image=image,
        method=method,
        num_captions=int(num_captions),
    )
    table = pd.DataFrame(rows)
    return best_caption, table, warning


with gr.Blocks(title="Image Caption Enhancer") as demo:
    gr.Markdown("# Image Caption Enhancer")
    gr.Markdown("Загрузка изображения → генерация нескольких BLIP-подписей → выбор лучшей подписи скорером.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Изображение")
        with gr.Column():
            method_input = gr.Dropdown(
                choices=["cross_encoder", "bi_encoder", "consensus", "first_blip"],
                value="cross_encoder",
                label="Метод выбора",
            )
            num_captions_input = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=5,
                label="Количество BLIP-подписей",
            )
            run_button = gr.Button("Выбрать подпись")

    best_output = gr.Textbox(label="Итоговая подпись")
    table_output = gr.Dataframe(label="Все кандидаты и скоры")
    warning_output = gr.Textbox(label="Сообщение")

    run_button.click(
        fn=run_caption_selection,
        inputs=[image_input, method_input, num_captions_input],
        outputs=[best_output, table_output, warning_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
