from config import *
from collections import OrderedDict, defaultdict

class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(Model, self).__init__()
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
        seq_output = torch.cat([hs[-1],hs[-2],hs[-3]], dim=-1)
        avg_output = torch.sum(seq_output * attention_mask.unsqueeze(-1), dim=1, keepdim=False)
        avg_output = avg_output / torch.sum(attention_mask, dim=-1, keepdim=True)
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0, 2, 1)).permute(0, 2, 1))

        se_out = self.se_head(self.dropout(seq_output))  # ()
        inst_out = self.inst_head(self.dropout(seq_output))

        return se_out[:, :, 0], se_out[:, :, 1]