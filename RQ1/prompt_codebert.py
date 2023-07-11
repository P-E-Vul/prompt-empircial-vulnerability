import json
import torch
import torch.nn as nn
import random
import os
import numpy as np
from openprompt.data_utils import InputExample
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def read_answers(filename):
    answers = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            # code = js['func']
            # target = js['target']
            example = InputExample(guid=js['target'], text_a=js['func'])
            answers.append(example)
    return answers


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


train_dataset = read_answers('train.jsonl')
valid_dataset = read_answers('valid.jsonl')
test_dataset = read_answers('test.jsonl')

classes = ['negative', 'positive']
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "microsoft/codebert-base")
from openprompt.prompts import ManualTemplate, SoftTemplate, MixedTemplate

promptTemplate = MixedTemplate(
    model=plm,
    text='The code {"placeholder":"text_a"} is {"mask"}.',
    tokenizer=tokenizer,
)
from openprompt.prompts import ManualVerbalizer

promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words={
        "negative": ["clean", "good"],
        "positive": ["defective", "bad"],
    },
    tokenizer=tokenizer,
)
from openprompt import PromptForClassification

promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)

from openprompt import PromptDataLoader

train_data_loader = PromptDataLoader(
    dataset=train_dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=16,
)
valid_data_loader = PromptDataLoader(
    dataset=valid_dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=16
)
test_data_loader = PromptDataLoader(
    dataset=test_dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=16
)
# from openprompt import ClassificationRunner
# cls_runner = ClassificationRunner(promptModel, train_dataloader=train_data_loader, valid_dataloader=valid_data_loader, test_dataloader=test_data_loader, config=model_config)
# quit()
promptModel = promptModel.cuda()

def test(model, test_data_loader):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    model.eval()
    device = torch.device("cuda")
    with torch.no_grad():
        for batch in test_data_loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=-1)
            tp += torch.logical_and(torch.eq(batch['guid'], 1), torch.eq(preds.cpu(), 1)).sum()
            fp += torch.logical_and(torch.eq(batch['guid'], 0), torch.eq(preds.cpu(), 1)).sum()
            tn += torch.logical_and(torch.eq(batch['guid'], 0), torch.eq(preds.cpu(), 0)).sum()
            fn += torch.logical_and(torch.eq(batch['guid'], 1), torch.eq(preds.cpu(), 0)).sum()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1_score)


def train(model, train_data_loader):
    model = model.cuda()
    set_seed()
    # ---------------
    max_epochs = 20
    max_steps = max_epochs * len(train_data_loader)
    warm_up_steps = len(train_data_loader)
    output_dir = './saved_models'
    gradient_accumulation_steps = 1
    lr = 2e-5
    adam_epsilon = 1e-5
    device = torch.device("cuda")
    max_grad_norm = 1.0
    # ----------------
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)
    checkpoint_last = os.path.join(output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))

    logger.info("  Total optimization steps = %d", max_steps)
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    total_loss = 0.0
    sum_loss = 0.0
    for idx in range(0, max_epochs):
        total_loss = 0.0
        sum_loss = 0.0
        logger.info("******* Epoch %d *****", idx)
        for batch_idx, batch in enumerate(train_data_loader):

            batch.to(device)
            labels = batch['guid'].to(device)
            model.train()
            logits = model(batch)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)

            sum_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if global_step % 50 == 0:
                    print('train/loss', sum_loss, global_step)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += sum_loss
                sum_loss = 0.
                global_step += 1
        logger.info(f"Training epoch {idx}, num_steps {global_step},  total_loss: {total_loss:.4f}")
        test(model, test_data_loader)


train(promptModel, train_data_loader)
