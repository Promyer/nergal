import json
from string import punctuation
from tqdm.contrib.telegram import tqdm
from torch.utils.data import Dataset
import torch
from transformers import MBartTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

class SummarizationDataset(Dataset):

    def __init__(self, data_path, paraphrase_summary=False):
        self.data = []
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
        self.tokenizer.tgt_lang = "ru_RU"
        self.tokenizer.src_lang = "ru_RU"

        # parafrase block
        if paraphrase_summary:
            MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
            p_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
            p_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
            p_model = p_model.to(torch.device("cuda:"))
            def paraphrase(batch, beams=5, grams=4, do_sample=False):
                x = p_tokenizer([x["summary"] for x in batch], return_tensors='pt', padding=True).to(p_model.device)
                max_size = int(x.input_ids.shape[1] * 1.5 + 10)
                out = p_model.generate(**x, encoder_no_repeat_ngram_size=grams, num_beams=beams, max_length=max_size, do_sample=do_sample)
                paraphrased_batch = []
                for tknzd, sample in zip(out, batch):
                    paraphrase_text = p_tokenizer.decode(tknzd, skip_special_tokens=True).lower()
                    for letter in punctuation + "«»—":
                        paraphrase_text = paraphrase_text.replace(letter, "")
                    sample["summary"] = paraphrase_text
                    paraphrased_batch.append(sample)
                return paraphrased_batch

        data = []
        for line in open(data_path, "r"):
            data.append(line)

        paraphrase_bufer = []
        for line in tqdm(
            data,
            token="",
            chat_id=""
        ):
            news = json.loads(line)
            if paraphrase_summary:
                paraphrase_bufer.append({
                        "text": news["text"],
                        "summary": news["summary"]
                    })

            news["text"] = news["text"].lower()
            news["summary"] = news["summary"].lower()
            for letter in punctuation + "«»—":
                news["text"] = news["text"].replace(letter, "")
                news["summary"] = news["summary"].replace(letter, "")

            self.data.append({
                    "text": news["text"],
                    "summary": news["summary"]
                })
            if len(paraphrase_bufer) == 64:
                self.data += paraphrase(paraphrase_bufer)
                paraphrase_bufer = []

        if paraphrase_bufer:
            self.data += paraphrase(paraphrase_bufer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        model_inputs = self.tokenizer(
            self.data[idx]["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=600,
            )
        with self.tokenizer.as_target_tokenizer():
            model_result = self.tokenizer(
                self.data[idx]["summary"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=600,
                )
        model_inputs["labels"] = model_result["input_ids"]


        for key in model_inputs:
            model_inputs[key] = model_inputs[key][0]
        return model_inputs
