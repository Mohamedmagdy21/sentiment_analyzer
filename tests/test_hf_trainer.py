from unittest.mock import AsyncMock, MagicMock


import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd


class TestHuggingFaceTrainer(unittest.TestCase):

    @patch[MagicMock | AsyncMock]("training.trainers.hf_trainer.AutoTokenizer")
    @patch("training.trainers.hf_trainer.AutoModelForSequenceClassification")
    @patch("training.trainers.hf_trainer.get_peft_model")
    def test_build_model_applies_lora(
        self,
        mock_get_peft,
        mock_auto_model,
        mock_auto_tokenizer,
    ):
        from training.trainers.hf_trainer import HuggingFaceTrainer

        mock_base = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_base
        mock_peft = MagicMock()
        mock_get_peft.return_value = mock_peft

        trainer = HuggingFaceTrainer(
            name="test",
            pretrained_name="roberta-base",
            tokenizer_name="roberta-base",
            num_labels=3,
            max_length=64,
            artifact_dir="/tmp/test_model",
            labels={"negative": 0, "neutral": 1, "positive": 2},
        )

        result = trainer._build_model()

        mock_auto_model.from_pretrained.assert_called_once_with(
            "roberta-base", num_labels=3, ignore_mismatched_sizes=True
        )
        mock_get_peft.assert_called_once()
        args, kwargs = mock_get_peft.call_args
        base_model_arg, lora_config_arg = args
        self.assertIs(base_model_arg, mock_base)
        self.assertEqual(lora_config_arg.r, 8)
        self.assertEqual(lora_config_arg.lora_alpha, 16)
        self.assertEqual(lora_config_arg.lora_dropout, 0.1)
        self.assertEqual(
            lora_config_arg.target_modules,
            ["query", "key", "value", "dense"],
        )
        self.assertIs(result, mock_peft)
        self.assertIs(trainer.model, mock_peft)

    def test_train_with_tiny_csv(self):
        from training.trainers.hf_trainer import HuggingFaceTrainer

        with tempfile.TemporaryDirectory() as tmp:

            train_csv = f"{tmp}/train.csv"
            val_csv = f"{tmp}/val.csv"

            pd.DataFrame({
                "text": ["good movie", "bad movie", "ok film"],
                "label": [2, 0, 1],
            }).to_csv(train_csv, index=False)

            pd.DataFrame({
                "text": ["great", "terrible"],
                "label": [2, 0],
            }).to_csv(val_csv, index=False)

            dataset_cfg = MagicMock()
            dataset_cfg.processed_train_path = train_csv
            dataset_cfg.processed_val_path = val_csv

            with (
                patch(
                    "training.trainers.hf_trainer.AutoTokenizer"
                ) as mock_tok,
                patch(
                    "training.trainers.hf_trainer.AutoModelForSequenceClassification"
                ) as mock_model,
                patch(
                    "training.trainers.hf_trainer.get_peft_model"
                ) as mock_peft,
                patch(
                    "training.trainers.hf_trainer.HuggingFaceTrainer._build_trainer"
                ) as mock_build,
            ):

                mock_tok_instance = MagicMock()

                def fake_tokenize(examples, **kw):
                    bs = len(examples["text"])
                    return {
                        "input_ids": [[0] * 64 for _ in range(bs)],
                        "attention_mask": [[1] * 64 for _ in range(bs)],
                    }

                mock_tok_instance.side_effect = (
                    lambda *a, **kw: mock_tok_instance
                )
                mock_tok_instance.return_value = mock_tok_instance

                mock_tok.from_pretrained.return_value = mock_tok_instance
                mock_tok_instance.__call__ = MagicMock(
                    side_effect=fake_tokenize
                )

                mock_base = MagicMock()
                mock_model.from_pretrained.return_value = mock_base
                mock_peft_model = MagicMock()
                mock_peft.return_value = mock_peft_model

                mock_trainer_obj = MagicMock()
                mock_build.return_value = mock_trainer_obj

                trainer = HuggingFaceTrainer(
                    name="test",
                    pretrained_name="roberta-base",
                    tokenizer_name="roberta-base",
                    num_labels=3,
                    max_length=64,
                    artifact_dir=f"{tmp}/model",
                    labels={"neg": 0, "neu": 1, "pos": 2},
                )

                result = trainer.train(dataset_cfg)

                self.assertIs(result, mock_trainer_obj)
                mock_build.assert_called_once()
                mock_trainer_obj.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
