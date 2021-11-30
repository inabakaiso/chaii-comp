from config import *

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


class Model(nn.Module):
    def __init__(self, modelname_or_path, config, layer_start, layer_weights=None):
        super(Model, self).__init__()
        self.config = config
        config.update({
            "hidden_dropout_prob": 0.0,
            "layer_norm_eps": 1e-7,
            "output_hidden_states": True
        })

        self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path, config=config)
        self.layer_start = layer_start
        self.pooling = WeightedLayerPooling(config.num_hidden_layers,
                                            layer_start=layer_start,
                                            layer_weights=None)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = torch.nn.Dropout(0.3)
        self.qa_output = torch.nn.Linear(config.hidden_size, 2)
        torch.nn.init.normal_(self.qa_output.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.xlm_roberta(input_ids, attention_mask=attention_mask)
        all_hidden_states = torch.stack(outputs.hidden_states)
        weighted_pooling_embeddings = self.layer_norm(self.pooling(all_hidden_states))
        #weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]

        norm_embeddings = self.dropout(weighted_pooling_embeddings)
        logits = self.qa_output(norm_embeddings)
        start_logits, end_logits = logits.split(1, dim=-1)

        # 4 400 1
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits