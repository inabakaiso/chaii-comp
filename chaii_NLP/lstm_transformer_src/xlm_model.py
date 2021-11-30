from config import *

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

        self.dropout = torch.nn.Dropout(0.3)
        self.qa_output = torch.nn.Linear(self.h_size * 2, 2)

        torch.nn.init.normal_(self.qa_output.weight, std=0.02)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            # token_type_ids=None
    ):
        out = self.xlm_roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        LAST_HIDDEN_LAYERS = 3  # 24

        out = out.hidden_states
        out = torch.stack(tuple(out[-i - 1] for i in range(LAST_HIDDEN_LAYERS)), dim=0)

        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        out = torch.cat((out_mean, out_max), dim=-1)

        logits = torch.mean(torch.stack([
            self.qa_output(self.dropout(out))
            for _ in range(5)
        ], dim=0), dim=0)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits