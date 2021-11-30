from config import *
from dataset import *
from xlm_model_2 import *
#from xlm_model import *
# from xlm_model_3 import *
from utils import *
from preprocess import *
from train_fn import *
# from pooler_roberta import *
#from cnn_roberta import *


def make_model(args):
    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = Model(args.model_name_or_path, config=config,layer_start=12,layer_weights=None)
    #model = Model(args.model_name_or_path, config=config)
    # model = RobertaForSentimentExtraction(args.model_name_or_path, config=config)
    return config, tokenizer, model

def _make_optimizer(args, model):
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model)
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": args.weight_decay,
#         },
#         {
#             "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]
    if args.optimizer_type == "AdamW":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.epsilon,
            correct_bias=True
        )
        return optimizer
def __make_optimizer(args, model):
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    if args.optimizer_type == "AdamW":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.epsilon,
            correct_bias=True
        )
        return optimizer

def make_optimizer(args, model):
    named_parameters = list(model.named_parameters())

    roberta_parameters = named_parameters[:389]
    pooler_parameters = named_parameters[389:391]
    qa_parameters = named_parameters[391:]

    parameters = []

    # increase lr every k layer
    increase_lr_every_k_layer = 1
    lrs = np.linspace(1, 5, 24 // increase_lr_every_k_layer)
    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01
        splitted_name = name.split('.')
        lr = args.learning_rate  # Config.lr
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[3]):
            layer_num = int(splitted_name[3])
            lr = lrs[layer_num // increase_lr_every_k_layer] * lr

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    default_lr = 1e-3  # default LR for AdamW
    for layer_num, (name, params) in enumerate(qa_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01
        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": default_lr})

    for layer_num, (name, params) in enumerate(pooler_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01
        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": default_lr})

    return AdamW(parameters)


def make_scheduler(
        args, optimizer,
        num_warmup_steps,
        num_training_steps
):
    if args.decay_name == "cosine-warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    return scheduler


def make_loader(
        args, data,
        tokenizer, fold
):
    train_set, valid_set = data[data['kfold'] != fold], data[data['kfold'] == fold].reset_index(drop=True)

    train_features, valid_features = [[] for _ in range(2)]
    for i, row in train_set.iterrows():
        train_features += prepare_train_features(args, row, tokenizer)
    for i, row in valid_set.iterrows():
        valid_features += prepare_train_features(args, row, tokenizer)

    ## Weighted sampler
    hindi_tamil_count = []
    for i, f in enumerate(train_features):
        hindi_tamil_count.append(train_features[i]['hindi_tamil'])
    class_sample_count = pd.Series(hindi_tamil_count).value_counts().values
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in hindi_tamil_count])
    samples_weight = torch.from_numpy(samples_weight)
    wsampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                              len(samples_weight))

    train_dataset = DatasetRetriever(train_features, mode="train")
    valid_dataset = DatasetRetriever(valid_features, mode="valid")
    print(f"Num examples Train= {len(train_dataset)}, Num examples Valid={len(valid_dataset)}")

    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=True,
        drop_last=False
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        sampler=valid_sampler,
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, valid_dataloader, valid_features, valid_set


def init_training(args, data, fold):
    fix_all_seeds(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # model
    model_config, tokenizer, model = make_model(args)
    if torch.cuda.device_count() >= 1:
        print('Model pushed to {} GPU(s), type {}.'.format(
            torch.cuda.device_count(),
            torch.cuda.get_device_name(0))
        )
        model = model.cuda()
    else:
        raise ValueError('CPU training is not supported')

    # data loaders
    train_dataloader, valid_dataloader, valid_features, valid_set = make_loader(args, data, tokenizer, fold)

    # optimizer
    optimizer = make_optimizer(args, model)

    # scheduler
    num_training_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.epochs
    if args.warmup_ratio > 0:
        num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    else:
        num_warmup_steps = 0
    print(f"Total Training Steps: {num_training_steps}, Total Warmup Steps: {num_warmup_steps}")
    scheduler = make_scheduler(args, optimizer, num_warmup_steps, num_training_steps)

    # mixed precision training with NVIDIA Apex
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    result_dict = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': np.inf
    }

    return (
        model, model_config, tokenizer, optimizer, scheduler,
        train_dataloader, valid_dataloader, result_dict, valid_features, valid_set
    )


def run(data, fold):
    args = Config()

    all_jacard_scores = []

    model, model_config, tokenizer, optimizer, scheduler, train_dataloader, \
    valid_dataloader, result_dict, valid_features, valid_set = init_training(args, data, fold)

    trainer = Trainer(model, tokenizer, optimizer, scheduler)
    evaluator = Evaluator(model)

    train_time_list = []
    valid_time_list = []

    for epoch in range(args.epochs):
        result_dict['epoch'].append(epoch)

        # Train
        torch.cuda.synchronize()
        tic1 = time.time()
        result_dict = trainer.train(
            args, train_dataloader,
            epoch, result_dict
        )
        torch.cuda.synchronize()
        tic2 = time.time()
        train_time_list.append(tic2 - tic1)
        # Evaluate
        torch.cuda.synchronize()
        tic3 = time.time()
        result_dict, all_outputs_start, all_outputs_end = evaluator.evaluate(
            valid_dataloader, epoch, result_dict
        )
        torch.cuda.synchronize()
        #         # Get valid jaccard score
        valid_features1 = valid_features.copy()
        valid_preds = postprocess_qa_predictions(valid_set, valid_features1, (all_outputs_start, all_outputs_end),
                                                 tokenizer)
        valid_set['PredictionString'] = valid_set['id'].map(valid_preds)
        valid_set['jaccard'] = valid_set[['answer_text', 'PredictionString']].apply(lambda x: jaccard(x[0], x[1]), axis=1)
        print("valid jaccard: ", np.mean(valid_set.jaccard))
        all_jacard_scores.append(np.mean(valid_set.jaccard))

        tic4 = time.time()
        valid_time_list.append(tic4 - tic3)

        output_dir = os.path.join(args.output_dir, f"checkpoint-fold-{fold}-epoch-{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        if result_dict['val_loss'][-1] < result_dict['best_val_loss']:
            print("{} Epoch, Best epoch was updated! Valid Loss: {: >4.5f}".format(epoch, result_dict['val_loss'][-1]))
            result_dict["best_val_loss"] = result_dict['val_loss'][-1]

            #             os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
        model_config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Saving model checkpoint to {output_dir}.")

        print()

    evaluator.save(result_dict, output_dir)

    print(
        f"Total Training Time: {np.sum(train_time_list)}secs, Average Training Time per Epoch: {np.mean(train_time_list)}secs.")
    print(
        f"Total Validation Time: {np.sum(valid_time_list)}secs, Average Validation Time per Epoch: {np.mean(valid_time_list)}secs.")

    # del trainer, evaluator
    # del model, model_config, tokenizer
    # del optimizer, scheduler
    # del train_dataloader, valid_dataloader, result_dict

if __name__ == "__main__":
    data = pd.read_csv('../input/preprocess_folded_inputs.csv')
    data['answers'] = data[['answer_start', 'answer_text']].apply(convert_answers, axis=1)
    # data = preprocess(data, 'dev')
    for fold in range(5):
        print()
        print()
        run(data, fold)
