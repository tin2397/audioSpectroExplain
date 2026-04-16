import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import librosa

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
model = ASTForAudioClassification.from_pretrained(
    model_name,
    attn_implementation="eager"
)

y, sr = librosa.load(r"C:\Users\tin23\OneDrive\Desktop\HONS WORKING\playWithSHAP\TinySOL\TinySOL2020\Strings\Violin\ordinario\Vn-ord-A#3-ff-4c-T15u.wav", sr=16000)
inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attentions = outputs.attentions
logits = outputs.logits

# Get prediction

predicted_class_id = torch.argmax(logits, dim=-1).item()
label = model.config.id2label[predicted_class_id]
print(f"Predicted label: {label}")
print(attentions[0].shape)  # Should be (batch_size, num_heads, seq_len, seq_len)