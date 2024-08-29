from paddleocr import PaddleOCR
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 初始化PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 加载文本纠错模型
pretrained = "T5Corrector-base-v2"
tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = T5ForConditionalGeneration.from_pretrained(pretrained)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义文本纠错函数
def correct(text, max_length=32, num_beams=5, num_iterations=2):
    for _ in range(num_iterations):
        model_inputs = tokenizer(text,
                                 max_length=max_length,
                                 truncation=True,
                                 return_tensors="pt").to(device)
        output = model.generate(**model_inputs,
                                num_beams=num_beams,
                                no_repeat_ngram_size=4,
                                do_sample=True,
                                early_stopping=True,
                                max_length=max_length,
                                return_dict_in_generate=True,
                                output_scores=True)
        text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
    return text

# 定义OCR识别和强化纠错流程
def ocr_and_correct(image_path, max_length=32, num_iterations=2):
    # 进行OCR识别
    result = ocr.ocr(image_path, cls=True)
    recognized_sentences = []

    # 提取识别的句子并进行初步预处理
    for line in result:
        sentence = ''.join([word_info[1][0] for word_info in line])
        recognized_sentences.append(sentence.strip())

    corrected_sentences = []

    # 多阶段纠错
    for sentence in recognized_sentences:
        corrected = correct(sentence, max_length, num_iterations=num_iterations)
        corrected_sentences.append(corrected)

    # 返回原始和纠正后的结果
    return recognized_sentences, corrected_sentences

# 添加标记错别字的步骤
def mark_corrections(original_text, corrected_text):
    marked_text = []
    for orig_char, corr_char in zip(original_text, corrected_text):
        if orig_char != corr_char:
            marked_text.append(f"[{corr_char}]")
        else:
            marked_text.append(orig_char)
    return ''.join(marked_text)

# 测试代码
image_path = '6.jpg'  # 替换为你的图片路径
recognized_sentences, corrected_texts = ocr_and_correct(image_path, num_iterations=3)

# 处理标记后的输出
for original, corrected in zip(recognized_sentences, corrected_texts):
    marked_correction = mark_corrections(original, corrected)
    print(f"原始: {original}\n纠正: {corrected}\n标记: {marked_correction}\n")

    # 保存标记后的结果到文件
    with open("correction_results.txt", "a", encoding="utf-8") as f:
        f.write(f"原始: {original}\n纠正: {corrected}\n标记: {marked_correction}\n\n")
