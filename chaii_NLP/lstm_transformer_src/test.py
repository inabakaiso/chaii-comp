from config import *
from utils import *
from xlm_model import *
from dataset import *


def test(data, checkpoint_path, inference=False):
    if inference:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

        test_features = []
        for i, row in data.iterrows():
            test_features += prepare_test_features(args, row, tokenizer)


        test_dataset = DatasetRetriever(test_features, mode='test')
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            sampler=SequentialSampler(test_dataset),
            num_workers=optimal_num_of_loader_workers(),
            pin_memory=True,
            drop_last=False
        )

        #config, tokenizer, model = make_model(Config())
        config = AutoConfig.from_pretrained(args.config_name)
        model = Model(args.model_name_or_path, config=config)

        model.cuda();
        model.load_state_dict(
            torch.load('../output/' + checkpoint_path)
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

        return np.vstack(start_logits), np.vstack(end_logits), test_features

if __name__ == "__main__":
    data = pd.read_csv('../input/test.csv')
    start_logits1, end_logits1, test_features1 = test(data, '../output/checkpoint-fold-0/pytorch_model.bin', args.inference)
    start_logits2, end_logits2, test_features2 = test(data, '../output/checkpoint-fold-1/pytorch_model.bin', args.inference)
    start_logits3, end_logits3, test_features3 = test(data, '../output/checkpoint-fold-2/pytorch_model.bin', args.inference)
    start_logits4, end_logits4, test_features4 = test(data, '../output/checkpoint-fold-3/pytorch_model.bin', args.inference)
    start_logits5, end_logits5, test_features5 = test(data, '../output/checkpoint-fold-4/pytorch_model.bin', args.inference)

    start_logits = (start_logits1 + start_logits2 + start_logits3 + start_logits4 + start_logits5) / 5
    end_logits = (end_logits1 + end_logits2 + end_logits3 + end_logits4 + end_logits5) / 5
    test_features = set(list(test_features1 + test_features2 + test_features3 + test_features4 + test_features5))

    predictions = postprocess_qa_predictions(test, test_features, (start_logits, end_logits))

    test['PredictionString'] = test['id'].map(predictions)
    test[['id', 'PredictionString']].to_csv('../submission/submission.csv', index=False)

    print(test[['id', 'PredictionString']])