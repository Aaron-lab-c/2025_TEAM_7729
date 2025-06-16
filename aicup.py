import torch
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from collections import defaultdict
import re
from typing import Optional, Tuple
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
  processor: Any
  decoder_start_token_id: int

  def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    input_features = [{"input_features": feature["input_features"]} for feature in features]
    batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

    label_features = [{"input_ids": feature["labels"]} for feature in features]

    labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
      labels = labels[:, 1:]

    batch["labels"] = labels

    return batch

def transcribe_with_timestamps(audio_data,model,processor):
  audio_array = audio_data['audio']['array']
  sr = audio_data['audio']['sampling_rate']
  duration = len(audio_array) / sr

  input_features = processor.feature_extractor(
      audio_array, sampling_rate=sr, return_tensors="pt"
  ).input_features.to(model.device)

  with torch.no_grad():
    generated_ids = model.generate(input_features)

  transcription = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

  tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids[0])
  num_tokens = len(tokens)
  timestamps = [(i * duration / num_tokens) for i in range(num_tokens)]

  result = {
      "text": transcription,
      "segments": []
  }

  for i, (token, timestamp) in enumerate(zip(tokens, timestamps)):
      result["segments"].append({
          "word": token,
          "start": round(timestamp, 2),
          "end": round(timestamps[i + 1] if i + 1 < len(timestamps) else duration, 2)
      })

  return result

def collate_batch_with_prompt_template(batch, tokenizer, template = "<|endoftext|> __CONTENT__\n\n####\n\n__LABEL__ <|END|>", IGNORED_PAD_IDX = -100):
  """ template: __CONTENT__ and __LABEL__ will be replaced with the content and the corresponding labels."""
  # default template: {bos} {data['content']} {sep}

  texts = [template.replace("__LABEL__", data['label']).replace("__CONTENT__", data['content']) for data in list(batch)]
  encoded_seq = tokenizer(texts, padding=True)

  indexed_tks = torch.tensor(encoded_seq['input_ids'])
  attention_mask = torch.tensor(encoded_seq['attention_mask'])
  encoded_label = torch.tensor(encoded_seq['input_ids'])
  encoded_label[encoded_label == tokenizer.pad_token_id] = IGNORED_PAD_IDX

  return indexed_tks, encoded_label, attention_mask

def read_file(path):
  with open(path , 'r' , encoding = 'utf-8-sig') as fr:
    return fr.readlines()

def process_annotation_file(lines):
  '''
  deal /w anwser.txt anno file

  output:annotation dicitonary
  '''
  print("process annotation file...")
  entity_dict = {}
  dataa = read_file(lines)
  for line in dataa:
    items = line.strip('\n').split('\t')
    if len(items) == 5:
      item_dict = {
          'phi' : items[1],
          'st_time' : float(items[2]),
          'ed_time' : float(items[3]),
          'entity' : items[4],
    }
    elif len(items) == 6:
      item_dict = {
          'phi' : items[1],
          'st_time' : float(items[2]),
          'ed_time' : float(items[3]),
          'entity' : items[4],
          'normalize_time' : items[5],
      }
    if items[0] not in entity_dict:
        entity_dict[items[0]] = [item_dict]
    else:
        entity_dict[items[0]].append(item_dict)
  print("annotation file done")
  return entity_dict

def process_transcribe(audio_transcribe):
  sents = read_file(audio_transcribe)

  transcribe = {}
  for s_idx, sentence in enumerate(sents):
    items = sentence.strip('\n').split('\t')
    transcribe[items[0]] = items[1]
  return transcribe

def process_medical_report(annos_dict,trans_lines):
  temp_seq , seq_pairs = "" , []
  new_line_idx = 0
  for pid,content in trans_lines.items():
    pid_annos_dict = annos_dict[pid] if pid in annos_dict else [{"phi":"PHI","entity":"Null"}]
    temp_seq = "\\n".join(f"{phi_entity['phi']}:{phi_entity['entity']}" for phi_entity in pid_annos_dict)
    seq_pair = f"{pid}\t{content}\t{temp_seq}\n"
    seq_pairs.append(seq_pair)
  return seq_pairs

def generate_annotated_audio_transcribe_parallel(anno_file_path, transcribe_report_folder , tsv_output_path , num_processes=4):
  annos_dict = process_annotation_file(anno_file_path)
  trans_lines = process_transcribe(transcribe_report_folder)

  print("processing each medical file")

  processed_data = process_medical_report(annos_dict, trans_lines)
  print(processed_data[10])
  print(len(processed_data))
  print("All medical file done")
  print("write out to tsv format...")
  with open(tsv_output_path , 'w' , encoding = 'utf-8') as fw:
      for seq_pair in processed_data:
          fw.write(seq_pair)
  print("tsv format dataset done")
  # return all_seq_pairs

def gpt_batch_decode(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Tokenize
    input_ids = tokenizer.encode(text, return_tensors="pt",
          truncation=True, max_length=max_input_tokens)

    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(input_ids=input_ids.to(device),
        pad_token_id = tokenizer.eos_token_id, max_new_tokens=max_output_tokens)

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer
def is_all_english(text: str) -> bool:
    """判斷字串是否全為英文字母（含空白）。"""
    return all(ch.isalpha() or ch.isspace() for ch in text)

def normalize_time(text: str) -> str:
    """
    您自己的時間正規化邏輯。
    下面僅示意：去除尾端句點並補零對齊。
    """
    def repl(match):
        h = match.group(1)
        m = match.group(2)
        # 小时和分钟都补齐为两位
        h = h.zfill(2)
        m = m.zfill(2)
        return f"{h}:{m}"

    return re.sub(r"\b(\d{1,2})-(\d{1,2})\b", repl, text)
BANNED_PERSONALNAME = (
    '拜拜', 'parents', 'dr.', 'women', '星期', 'first', 'god', 'me',
    'girl ', 'mom', 'dad', 'his', 'him', 'her', 'he', 'she', 'father',
    'you', 'brother', 'night', 'now', 'friend', 'babysitter',
    'week', 'backyard', 'last time', 'hospital', 'yesterday',
    'today', 'sir', 'sister', 'last', 'family', 'call', 'friday'
)
def clean_kv(key: str, value: str) -> Optional[Tuple[str, str]]:
    """
    依照一系列規則清洗 (key, value)。
    - 傳回 None 表示這組資料應該被忽略。
    - 否則傳回 (key, value)。
    """
    # key, value = key.strip(), value.strip()

    # ---------- 規則一：值的直接改寫 ----------
    if key == "DOCTOR" and re.fullmatch(r"[A-Z]\.[A-Z]", value):
        value += "."

    if key == "LAB_NUMBER":
        key = "ID_NUMBER"

    if key == "TIME":
        value = normalize_time(value).strip()

    if key == "DOCTOR":
        value = value.replace("Dr.", "").strip()

    if key == "ZIP_CODE":
        key = "ZIP"

    # ---------- 規則二：需要「跳過」的情況 ----------
    # AGE 若值是全英文，丟棄
    if key == "AGE" and is_all_english(value):
        return None, None

    # PERSONALNAME 過長或含禁用詞，丟棄
    if key == "PERSONALNAME":
        tmp = value.lower()
        if len(tmp) > 34 or any(b in tmp for b in BANNED_PERSONALNAME):
          return None, None

    # TIME 中含 "driving"
    if key == "TIME" and "driving" in value:
        return None, None

    # DOCTOR 含中英文雜訊
    if key == "DOCTOR" and any(bad in value for bad in ("嘉宾", "谢谢", "健保卡")):
        return None, None

    # DATE 中的模糊詞
    if key == "DATE" and any(word in value for word in ("earlier", "girls", "morning", "evening")):
        return None, None

    # DURATION 或 DATE 同時含 "millimeter"
    if key in ("DURATION", "DATE") and "millimeter" in value:
        return None, None

    # PERSONALNAME / DATE 只有單一英文字母
    if key in ("PERSONALNAME", "DATE") and re.fullmatch(r"[A-Za-z]", value):
        return None, None

    # FAMILYNAME 值為「老闆」
    if key == "FAMILYNAME" and value == "老闆":
        return None, None

    # ---------- 全部檢查通過 ----------
    return key, value
class OpenDeidBatchSampler():    
    def __init__(self, data, batch_size):
        self.pooled_indices = []
        self.data = data
        self.batch_size = batch_size
        self.len = len(list(data))  
    def __iter__(self):
        self.pooled_indices = []
        indices = [(index, len(data["content"])) for index, data in enumerate(self.data)]
        random.shuffle(indices)
        # create pool of indices with similar lengths 
        for i in range(0, len(indices), self.batch_size * 100):
            self.pooled_indices.extend(sorted(indices[i:i + self.batch_size * 100], key=lambda x: x[1], reverse=True))
        self.pooled_indices = [x[0] for x in self.pooled_indices]

        # yield indices for current batch
        for i in range(0, len(self.pooled_indices), self.batch_size):
            yield self.pooled_indices[i:i + self.batch_size]
    def __len__(self):
        return (self.len + self.batch_size - 1) // self.batch_size   