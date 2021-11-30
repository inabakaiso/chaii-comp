# %%

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc

gc.enable()
import math
import json
import time
import random
import multiprocessing
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn import model_selection
from string import punctuation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
from torch.utils.data.distributed import DistributedSampler

try:
    from apex import amp

    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)

logging.set_verbosity_warning()
logging.set_verbosity_error()


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else num_cpus - 1
    return optimal_value


print(f"Apex AMP Installed :: {APEX_INSTALLED}")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# %%

class Config:
    # model
    model_type = 'xlm_roberta'
    model_name_or_path = "../input//xlm-roberta-large-squad2"
    config_name = "../input/xlm-roberta-large-squad2"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    # tokenizer
    tokenizer_name = "../input/xlm-roberta-large-squad2"
    max_seq_length = 400
    doc_stride = 135

    # train
    epochs = 1
    train_batch_size = 4
    eval_batch_size = 128

    # optimzer
    optimizer_type = 'AdamW'
    learning_rate = 1e-5
    weight_decay = 1e-2
    epsilon = 1e-8
    max_grad_norm = 1.0

    # scheduler
    decay_name = 'linear-warmup'
    warmup_ratio = 0.1

    # logging
    logging_steps = 10

    # evaluate
    output_dir = 'output'
    seed = 3


# %%

class DatasetRetriever(Dataset):
    def __init__(self, features, mode='train'):
        super(DatasetRetriever, self).__init__()
        self.features = features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        if self.mode == 'train':
            return {
                'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                'offset_mapping': torch.tensor(feature['offset_mapping'], dtype=torch.long),
                'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
                'end_position': torch.tensor(feature['end_position'], dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                'offset_mapping': feature['offset_mapping'],
                'sequence_ids': feature['sequence_ids'],
                'id': feature['example_id'],
                'context': feature['context'],
                'question': feature['question']
            }




from collections import OrderedDict, defaultdict


class CNNModel(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(CNNModel, self).__init__()
        self.config = config
        config.update({
            "hidden_dropout_prob": 0.0,
            "layer_norm_eps": 1e-7,
            "output_hidden_states": True
        })
        self.h_size = config.hidden_size
        self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path, config=config)

        self.cnn = nn.Conv1d(self.config.hidden_size * 3, self.config.hidden_size, kernel_size=3, padding=1)
        self.gelu = nn.GELU()

        self.whole_head = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(0.1)),
            ('l1', nn.Linear(self.config.hidden_size * 3, 256)),
            ('act1', nn.GELU()),
            ('dropout', nn.Dropout(0.1)),
            ('l2', nn.Linear(256, 2))
        ]))

        self.se_head = nn.Linear(self.config.hidden_size, 2)
        self.inst_head = nn.Linear(self.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            # token_type_ids=None
    ):
        outputs = self.xlm_roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        hs = outputs.hidden_states
        LAST_HIDDEN_LAYERS = 3  # 24
        # out = outputs.hidden_states
        seq_output = torch.cat([hs[-1], hs[-2], hs[-3]], dim=-1)
        avg_output = torch.sum(seq_output * attention_mask.unsqueeze(-1), dim=1, keepdim=False)
        avg_output = avg_output / torch.sum(attention_mask, dim=-1, keepdim=True)
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0, 2, 1)).permute(0, 2, 1))

        se_out = self.se_head(self.dropout(seq_output))  # ()
        inst_out = self.inst_head(self.dropout(seq_output))

        return se_out[:, :, 0], se_out[:, :, 1]


# %%

# class Model(nn.Module):
#     def __init__(self, modelname_or_path, config):
#         super(Model, self).__init__()
#         self.config = config
#         self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path, config=config)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self._init_weights(self.qa_outputs)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()

#     def forward(
#         self,
#         input_ids,
#         attention_mask=None,
#     ):
#         outputs = self.xlm_roberta(
#             input_ids,
#             attention_mask=attention_mask,
#         )

#         sequence_output = outputs[0]
#         pooled_output = outputs[1]

#         # sequence_output = self.dropout(sequence_output)
#         qa_logits = self.qa_outputs(sequence_output)

#         start_logits, end_logits = qa_logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)

#         return start_logits, end_logits

# %%

def make_model(args, dev):
    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = CNNModel(args.model_name_or_path, config=config)
    return config, tokenizer, model


# %%

def prepare_test_features(args, example, tokenizer):
    example["question"] = example["question"].lstrip()

    tokenized_example = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    features = []
    for i in range(len(tokenized_example["input_ids"])):
        feature = {}
        feature["example_id"] = example['id']
        feature['context'] = example['context']
        feature['question'] = example['question']
        feature['input_ids'] = tokenized_example['input_ids'][i]
        feature['attention_mask'] = tokenized_example['attention_mask'][i]
        feature['offset_mapping'] = tokenized_example['offset_mapping'][i]
        feature['sequence_ids'] = [0 if i is None else i for i in tokenized_example.sequence_ids(i)]
        features.append(feature)
    return features


# %%

import collections


def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    print(all_start_logits)

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []

        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            features[feature_index]["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions




# %%

def get_predictions(checkpoint_path, dev):
    config, tokenizer, model = make_model(Config(), dev)
    model.cuda();
    model.load_state_dict(
        torch.load(checkpoint_path)
    );

    start_logits = []
    end_logits = []
    for batch in test_dataloader:
        with torch.no_grad():
            outputs_start, outputs_end = model(batch['input_ids'].cuda(), batch['attention_mask'].cuda())
            start_logits.append(outputs_start.cpu().numpy().tolist())
            end_logits.append(outputs_end.cpu().numpy().tolist())
            del outputs_start, outputs_end
    del model, tokenizer, config
    gc.collect()
    return np.vstack(start_logits), np.vstack(end_logits)


#

if __name__ == '__main__':

    # %%

    test = pd.read_csv('../input/test.csv')

    test['context'] = test['context'].apply(lambda x: ' '.join(x.split()))
    test['question'] = test['question'].apply(lambda x: ' '.join(x.split()))

    # base_model_path = '../input/chaii-qa-5-fold-xlmroberta-torch-fit'

    tokenizer = AutoTokenizer.from_pretrained(Config().tokenizer_name)

    test_features = []
    for i, row in test.iterrows():
        test_features += prepare_test_features(Config(), row, tokenizer)

    args = Config()
    test_dataset = DatasetRetriever(test_features, mode='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        sampler=SequentialSampler(test_dataset),
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=True,
        drop_last=False
    )

    print('start get prediction')
    start_logits1, end_logits1 = get_predictions('../output_external/checkpoint-fold-0-epoch-0/pytorch_model.bin',
                                                 'cnn')
    start_logits2, end_logits2 = get_predictions('../output_external/checkpoint-fold-1-epoch-0/pytorch_model.bin',
                                                 'cnn')
    start_logits3, end_logits3 = get_predictions('../output_external/checkpoint-fold-2-epoch-0/pytorch_model.bin',
                                                 'cnn')
    start_logits4, end_logits4 = get_predictions('../output_external/checkpoint-fold-3-epoch-0/pytorch_model.bin',
                                                 'cnn')
    start_logits5, end_logits5 = get_predictions('../output_external/checkpoint-fold-4-epoch-0/pytorch_model.bin',
                                                 'cnn')

    start_logits_1 = (start_logits1 + start_logits2 + start_logits3 + start_logits4 + start_logits5) / 5
    end_logits_1 = (end_logits1 + end_logits2 + end_logits3 + end_logits4 + end_logits5) / 5

    start_logits1, end_logits1 = get_predictions('../output_external/checkpoint-fold-0-epoch-1/pytorch_model.bin', 'cnn')
    start_logits2, end_logits2 = get_predictions('../output_external/checkpoint-fold-1-epoch-1/pytorch_model.bin', 'cnn')
    start_logits3, end_logits3 = get_predictions('../output_external/checkpoint-fold-2-epoch-1/pytorch_model.bin', 'cnn')
    start_logits4, end_logits4 = get_predictions('../output_external/checkpoint-fold-3-epoch-1/pytorch_model.bin', 'cnn')
    start_logits5, end_logits5 = get_predictions('../output_external/checkpoint-fold-4-epoch-1/pytorch_model.bin', 'cnn')

    start_logits_2 = (start_logits1 + start_logits2 + start_logits3 + start_logits4 + start_logits5 )/5
    end_logits_2 = (end_logits1 + end_logits2 + end_logits3 + end_logits4 + end_logits5)/5

    start_logits = (start_logits_1 + start_logits_2) / 2
    end_logits = (end_logits_1 + end_logits_2) / 2

    fin_preds = postprocess_qa_predictions(test, test_features, (start_logits, end_logits))

    submission = []
    for p1, p2 in fin_preds.items():
        p2 = " ".join(p2.split())
        p2 = p2.strip(punctuation)
        submission.append((p1, p2))

    sample = pd.DataFrame(submission, columns=["id", "PredictionString"])

    test_data = pd.merge(left=test, right=sample, on='id')

    bad_starts = [".", ",", "(", ")", "-", "–", ",", ";"]
    bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

    tamil_ad = "கி.பி"
    tamil_bc = "கி.மு"
    tamil_km = "கி.மீ"
    hindi_ad = "ई"
    hindi_bc = "ई.पू"

    cleaned_preds = []
    for pred, context in test_data[["PredictionString", "context"]].to_numpy():
        if pred == "":
            cleaned_preds.append(pred)
            continue
        while any([pred.startswith(y) for y in bad_starts]):
            pred = pred[1:]
        while any([pred.endswith(y) for y in bad_endings]):
            if pred.endswith("..."):
                pred = pred[:-3]
            else:
                pred = pred[:-1]
        if pred.endswith("..."):
            pred = pred[:-3]

        if any([pred.endswith(tamil_ad), pred.endswith(tamil_bc), pred.endswith(tamil_km), pred.endswith(hindi_ad),
                pred.endswith(hindi_bc)]) and pred + "." in context:
            pred = pred + "."

        cleaned_preds.append(pred)

    test_data["PredictionString"] = cleaned_preds
    test_data[['id', 'PredictionString']].to_csv('submission.csv', index=False)

    # %%

    print(test_data[['id', 'PredictionString']])