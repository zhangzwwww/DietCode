from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torch

from transformer import (SequenceClassifierOutput, T5PreTrainedModel, T5EncoderModel, T5Model)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        classifier_dropout = (
            0.1
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CodeT5ForSequenceClassification(nn.Module):
    def __init__(self, encoder, config = None):
        super().__init__()
        self.num_labels = 2
        self.config = config

        self.encoder = encoder
        self.classifier = RobertaClassificationHead()

        # Initialize weights and apply final processing
        # self.post_init()
        # self.init_weights()
        # self._backward_compatibility_gradient_checkpointing()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = True

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
