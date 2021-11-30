from config import *


class PoolerStartLogits(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps if hasattr(config, "layer_norm_eps") else 1e-5)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None):
        assert (
                start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class RobertaForSentimentExtraction(nn.Module):
    base_model_prefix = "roberta"

    def __init__(self, model_name_or_path, config):
        super().__init__()

        self.config = config
        self.xlm_roberta = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)

    #        self.transformer = self.roberta

    def freeze(self):
        for child in self.roberta.children():
            for param in child.parameters():
                param.requires_grad = False

    def unfreeze(self):
        for child in self.roberta.children():
            for param in child.parameters():
                param.requires_grad = True

    def forward(
            self, beam_size=1, cls_ids=None,
            input_ids=None, attention_mask=None, token_type_ids=None, input_mask=None, position_ids=None,
            head_mask=None, inputs_embeds=None, start_positions=None, end_positions=None, p_mask=None
    ):
        outputs = self.xlm_roberta(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.hidden_states
        #        hidden_states = torch.mean(torch.stack((outputs[2][-1],outputs[2][-2], outputs[2][-3], outputs[2][-4])),0)
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        outputs = outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        # if start_positions is not None and end_positions is not None:
        #     # If we are on multi-GPU, let's remove the dimension added by batch splitting
        #     for x in (start_positions, end_positions):
        #         if x is not None and x.dim() > 1:
        #             x.squeeze_(-1)
        #
        #     # during training, compute the end logits based on the ground truth of the start position
        #     end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)
        #
        #     start_loss = loss_fct(start_logits, start_positions)
        #     end_loss = loss_fct(end_logits, end_positions)
        #     total_loss = (start_loss + end_loss) / 2
        #     outputs = total_loss
        #
        # else:
        # during inference, compute the end logits based on beam search
        bsz, slen, hsz = hidden_states.size()
        start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)
        #            start_log_probs = F.sigmoid(start_logits)

        start_top_log_probs, start_top_index = torch.topk(
            start_log_probs, beam_size, dim=-1
        )  # shape (bsz, start_n_top)
        start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
        start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
        start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

        hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
            start_states
        )  # shape (bsz, slen, start_n_top, hsz)
        p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
        end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
        end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)
        #            end_log_probs = F.sigmoid(end_logits)

        end_top_log_probs, end_top_index = torch.topk(
            end_log_probs, beam_size, dim=1
        )  # shape (bsz, end_n_top, start_n_top)
        end_top_log_probs = end_top_log_probs.view(-1, beam_size * beam_size)
        end_top_index = end_top_index.view(-1, beam_size * beam_size)

        outputs = start_top_log_probs, start_top_index, end_top_log_probs, end_top_index

        return outputs
