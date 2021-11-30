from config import *
from preprocess import *
import collections


def postprocess_qa_predictions(examples, features1, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30):
    features = features1
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]
        # print(example['id'],example_index,feature_indices)
        min_null_score = None
        valid_answers = []

        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            offset_mapping = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]

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

"""
最初の文字の予測と最後の文字の予測を比較してlossを計算する
"""

def loss_fn(preds, labels):
    start_preds, end_preds = preds
    start_labels, end_labels = labels

    start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
    end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
    total_loss = (start_loss + end_loss) / 2
    return total_loss

"""
それぞれのlayerのparam管理
"""
def get_optimizer_grouped_parameters(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    group1=['layer.0.','layer.1.','layer.2.','layer.3.']
    group2=['layer.4.','layer.5.','layer.6.','layer.7.']
    group3=['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.xlm_roberta.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.xlm_roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': args.weight_decay, 'lr': args.learning_rate/2.6},
        {'params': [p for n, p in model.xlm_roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in model.xlm_roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': args.weight_decay, 'lr': args.learning_rate*2.6},
        {'params': [p for n, p in model.xlm_roberta.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
        {'params': [p for n, p in model.xlm_roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': args.learning_rate/2.6},
        {'params': [p for n, p in model.xlm_roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': args.learning_rate},
        {'params': [p for n, p in model.xlm_roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': args.learning_rate*2.6},
        {'params': [p for n, p in model.named_parameters() if args.model_type not in n], 'lr':args.learning_rate*20, "weight_decay": 0.0},
    ]
    return optimizer_grouped_parameters

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def init_logger(log_file='../log/train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()

#     LOGGER.info(f"========== fold: {fold} training ==========")

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def convert_answers(row):
    return {'answer_start': [row[0]], 'text': [row[1]]}

# class EarlyStopping(Callback):
#     def __init__(
#         self,
#         monitor,
#         model_path,
#         valid_dataframe,
#         valid_data_loader,
#         tokenizer,
#         pad_on_right,
#         max_length,
#         doc_stride,
#         patience=3,
#         mode="min",
#         delta=0.001,
#         save_weights_only=False,
#     ):
#         self.monitor = monitor
#         self.patience = patience
#         self.counter = 0
#         self.mode = mode
#         self.best_score = None
#         self.early_stop = False
#         self.delta = delta
#         self.save_weights_only = save_weights_only
#         self.model_path = model_path
#         if self.mode == "min":
#             self.val_score = np.Inf
#         else:
#             self.val_score = -np.Inf
#
#         if self.monitor.startswith("train_"):
#             self.model_state = "train"
#             self.monitor_value = self.monitor[len("train_") :]
#         elif self.monitor.startswith("valid_"):
#             self.model_state = "valid"
#             self.monitor_value = self.monitor[len("valid_") :]
#         else:
#             raise Exception("monitor must start with train_ or valid_")
#
#         self.valid_targets = valid_dataframe.answer_text.values
#         self.valid_data_loader = valid_data_loader
#         self.tokenizer = tokenizer
#         valid_dataframe = valid_dataframe.drop(["answer_text", "answer_start"], axis=1)
#         self.valid_dataset = Dataset.from_pandas(valid_dataframe)
#         self.valid_features = self.valid_dataset.map(
#             partial(
#                 prepare_validation_features,
#                 tokenizer=self.tokenizer,
#                 pad_on_right=pad_on_right,
#                 max_length=max_length,
#                 doc_stride=doc_stride,
#             ),
#             batched=True,
#             remove_columns=self.valid_dataset.column_names,
#         )
#
#     def on_epoch_end(self, model):
#         model.eval()
#         tk0 = tqdm(self.valid_data_loader, total=len(self.valid_data_loader))
#         start_logits = []
#         end_logits = []
#
#         for _, data in enumerate(tk0):
#             with torch.no_grad():
#                 for key, value in data.items():
#                     data[key] = value.to("cuda")
#                 output, _, _ = model(**data)
#                 start = output[0].detach().cpu().numpy()
#                 end = output[1].detach().cpu().numpy()
#                 start_logits.append(start)
#                 end_logits.append(end)
#
#         start_logits = np.vstack(start_logits)
#         end_logits = np.vstack(end_logits)
#
#         valid_preds = postprocess_qa_predictions(
#             self.valid_dataset, self.tokenizer, self.valid_features, (start_logits, end_logits)
#         )
#         epoch_score = np.mean([jaccard(x, y) for x, y in zip(self.valid_targets, valid_preds.values())])
#         print(f"Jaccard Score = {epoch_score}")
#         model.train()
#         if self.mode == "min":
#             score = -1.0 * epoch_score
#         else:
#             score = np.copy(epoch_score)
#
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(epoch_score, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
#             if self.counter >= self.patience:
#                 model.model_state = enums.ModelState.END
#         else:
#             self.best_score = score
#             self.save_checkpoint(epoch_score, model)
#             self.counter = 0
#
#     def save_checkpoint(self, epoch_score, model):
#         if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
#             print("Validation score improved ({} --> {}). Saving model!".format(self.val_score, epoch_score))
#             model.save(self.model_path, weights_only=self.save_weights_only)
#         self.val_score = epoch_score


def convert_answers(r):
    start = r[0]
    text = r[1]
    return {"answer_start": [start], "text": [text]}