from transformers import Trainer
import torch
import torch.nn as nn


class WeightedLossTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        weights = torch.tensor(
            [1.1, 1.0, 1.0],  # Neg, Neutral, Positive
            device=logits.device
        )

        loss_fn = nn.CrossEntropyLoss(
            weight=weights
        )

        loss = loss_fn(
            logits,
            labels
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )