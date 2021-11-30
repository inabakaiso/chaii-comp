from config import *

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
            if self.mode == 'valid':
                return {
                    'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
                    'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                    'offset_mapping': torch.tensor(feature['offset_mapping'], dtype=torch.long),
                    'sequence_ids': feature['sequence_ids'],
                    'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
                    'end_position': torch.tensor(feature['end_position'], dtype=torch.long),
                    'example_id': feature['example_id'],
                    'context': feature['context'],
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