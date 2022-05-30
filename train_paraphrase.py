import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from nergal.dataset import SummarizationDataset
from nergal.trainer import Trainer


device = torch.device("cuda:")

train_dl = DataLoader(
    SummarizationDataset("gazeta_train.jsonl", paraphrase_summary=True),
    **{"batch_size": 4,
    "shuffle": True}
)
val_dl = DataLoader(
    SummarizationDataset("gazeta_val.jsonl"),
    **{"batch_size": 8,
    "shuffle": False}
)
test_dl = DataLoader(
    SummarizationDataset("gazeta_test.jsonl"),
    **{"batch_size": 8,
    "shuffle": False}
)

trainer = Trainer(train_dl, val_dl, test_dl, device, 20, AdamW, {"lr": 0.00001}, run_name="train_paraphrase")
trainer.train()
