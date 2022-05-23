from rouge import Rouge
import numpy as np
from tqdm.contrib.telegram import tqdm
#from tqdm.notebook import tqdm
#from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import MBartForConditionalGeneration
from transformers import MBartTokenizer


class Trainer:

    def __init__(
    self, train_dl, val_dl, test_dl, device, num_epoch, optimizer,
    optimizer_config={}, run_name="", checkpoint_path=""
    ):
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25").to(device)
        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        freeze = False
        if freeze:
            for param in self.model.model.parameters():
                param.requires_grad = False
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.device = device
        self.num_epoch = num_epoch
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
        self.optimizer = optimizer(
            self.model.parameters(), **optimizer_config
        )
        self.run_name = run_name
        self.rouge = Rouge()

    def train_batch(self, batch, i):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        res = self.model(**batch)
        loss = res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def evaluate_batch(self, batch):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        res = self.model(**batch)
        loss = res["loss"]
        return loss.item()

    def train_epoch(self):
        self.model = self.model.train()
        train_losses = []
        for i, d in enumerate(self.train_dl):
            train_losses.append(self.train_batch(d, i))

            if i % 800 == 799:
                self.model = self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for d in self.val_dl:
                        val_losses.append(self.evaluate_batch(d))

                    val_mean = np.mean(val_losses)
                    if not self.best_loss or val_mean <= self.best_loss:
                        self.best_loss = val_mean
                        torch.save(self.model.state_dict(), f"best_model_{self.run_name}.pt")

                    train_mean = np.mean(train_losses)
                    self.writer.add_scalar("loss/train", train_mean, self.i)
                    self.writer.add_scalar("loss/validation", val_mean, self.i)

                    if i % 4000 == 3999:
                        rouge = self.test_epoch()
                        for key, score in rouge.items():
                            self.writer.add_scalar(f"metric/rouge/{key}", score, self.i)

                        rouge_mean = np.mean(list(rouge.values()))
                        if not self.best_rouge or rouge_mean <= self.best_rouge:
                            self.best_rouge = rouge_mean
                            torch.save(self.model.state_dict(), f"best_rouge_model_{self.run_name}.pt")
                        torch.save(self.model.state_dict(), f"last_model_{self.run_name}.pt")
                    self.i += 1

                train_losses = []


    def test_epoch(self):
        model_results = []
        summarizations = []
        with torch.no_grad():
            self.model = self.model.eval()
            for d in self.test_dl:
                summarizations_model = self.model.generate(
                    input_ids=d["input_ids"].to(self.device),
                    attention_mask=d["attention_mask"].to(self.device),
                    num_beams=5,
                    length_penalty=1.0,
                    max_length=160,
                    min_length=5,
                    no_repeat_ngram_size=0,
                    early_stopping=True
                )
                for s in summarizations_model:
                    p = self.tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    model_results.append(p)
                for s in d["labels"]:
                    p = self.tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    summarizations.append(p)
            try:
                score = self.rouge.get_scores(model_results, summarizations, avg=True)
            except:
                return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
        return {key: val["f"] for key, val in score.items()}

    def train(self):
        self.writer = SummaryWriter(log_dir=f"summarization_logs/mbart/{self.run_name}")
        self.best_loss = None
        self.best_rouge = None
        self.i = 0
        for epoch_num in tqdm(
            range(self.num_epoch),
            token="",
            chat_id=""
        ):
            self.train_epoch()
        self.writer.close()
