import gradio as gr
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import torch.nn.functional as F

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("./bert-news-model/final")
model.eval()

def classify(headline):
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    return {LABELS[i]: round(probs[i], 4) for i in range(4)}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="News Headline"),
    outputs=gr.Label(num_top_classes=4),
    title="News Topic Classifier",
    description="Paste a news headline — BERT predicts its category."
)

demo.launch()