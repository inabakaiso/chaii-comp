from config import *


def prepare_train_features(args, example, tokenizer):
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

    sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_example.pop("offset_mapping")

    features = []
    for i, offsets in enumerate(offset_mapping):
        feature = {}

        input_ids = tokenized_example["input_ids"][i]
        attention_mask = tokenized_example["attention_mask"][i]

        feature['input_ids'] = input_ids
        feature['attention_mask'] = attention_mask
        feature['offset_mapping'] = offsets

        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_example.sequence_ids(i)

        ## for validation
        feature["example_id"] = example['id']
        feature['sequence_ids'] = [0 if i is None else i for i in tokenized_example.sequence_ids(i)]
        feature['context'] = example["context"]
        feature['question'] = example["question"]
        feature['hindi_tamil'] = 0 if example["language"] == 'hindi' else 1
        ##

        sample_index = sample_mapping[i]
        answers = example["answers"]

        if len(answers["answer_start"]) == 0:
            feature["start_position"] = cls_index
            feature["end_position"] = cls_index
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                feature["start_position"] = cls_index
                feature["end_position"] = cls_index
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                feature["start_position"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                feature["end_position"] = token_end_index + 1

        features.append(feature)
    return features


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

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def convert_answers(row):
    return {'answer_start': [row['answer_start']], 'text': [row['answer_text']]}

def preprocess(dataset, tier):
    num_exs = 0
    examples = []
    import pdb
    pdb.set_trace()
    for articles_id in tqdm(range(len(dataset)), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset.loc[dataset.id == articles_id, ()]
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            context = context.apply(lambda x: x.replace("''", '" '))
            context = context.apply(lambda x: x.replace("``", '" '))
            qas = article_paragraphs[pid]['qas']
            for qn in qas:
                question = qn['question']
                ans_text = qn['answers'][0]['text']
                ans_start_charloc = qn['answers'][0]['answer_start']
                ans_end_charloc = ans_start_charloc + len(ans_text)
                examples.append(
                    {
                        # 'id':articles_id,
                        'context':context,
                        'question':question,
                        'answer_text':ans_text,
                        'answer_start':ans_start_charloc,
                        # 'answer_end':ans_end_charloc
                    }
                )

                num_exs += 1
    print(num_exs)
    return examples