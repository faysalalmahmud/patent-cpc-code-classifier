#import necessery libraries
import gradio as gr
import onnxruntime as rt
from transformers import AutoTokenizer
import torch, json

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

with open("encode_revised_cpc_codes.json", "r") as fp:
  encode_cpc_types = json.load(fp)

cpc_code = list(encode_cpc_types.keys())

inf_session = rt.InferenceSession('distilroberta-base-patent-cpc-classifier-quantized.onnx')
input_name = inf_session.get_inputs()[0].name
output_name = inf_session.get_outputs()[0].name

def classify_cpc_code(abstract):
  input_ids = tokenizer(abstract)['input_ids'][:512]
  logits = inf_session.run([output_name], {input_name: [input_ids]})[0]
  logits = torch.FloatTensor(logits)
  probs = torch.sigmoid(logits)[0]
  return dict(zip(cpc_code, map(float, probs))) 

label = gr.Label(num_top_classes=5)
iface = gr.Interface(fn=classify_cpc_code, inputs="text", outputs=label)
iface.launch(inline=False)