import gradio as gr
from dotenv import load_dotenv

from src.predict import predict

load_dotenv()

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Enter a news headline or article", label="Text"),
    outputs=gr.Textbox(label="Predicted category"),
    title="AG News Classifier",
    description="Fine-tuned SciBERT model that classifies text into World, Sports, Business, or Sci/Tech.",
    examples=[
        "Wall Street stocks rallied after the Federal Reserve meeting.",
        "The team won the championship game last night.",
        "Scientists discover a new exoplanet orbiting a distant star.",
        "World leaders meet to discuss climate change policy.",
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)
