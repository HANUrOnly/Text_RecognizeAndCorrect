from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
pretrained ="T5Corrector-base-v2"
tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = T5ForConditionalGeneration.from_pretrained(pretrained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def correct(text, max_length):
    model_inputs = tokenizer(text,
                                max_length=max_length,
                                truncation=True,
                                return_tensors="pt").to(device)
    output = model.generate(**model_inputs,
                              num_beams=5,
                              no_repeat_ngram_size=4,
                              do_sample=True,
                              early_stopping=True,
                              max_length=max_length,
                              return_dict_in_generate=True,
                              output_scores=True)
    pred_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
    return pred_output
print("")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

image_path = '5.jpg'
img = cv2.imread(image_path)
recognized_texts = []
# 进行文字识别
result = ocr.ocr(image_path, cls=True)

for line in result:
    for word_info in line:
        text = word_info[1][0]
        recognized_texts.append(text)

output_text = ''.join(recognized_texts)
print(f"识别到的文字: {output_text}")

correction = correct(output_text, max_length=32)
print("纠正后的文字: "+correction)