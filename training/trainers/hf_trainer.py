# training/trainers/hf_trainer.py
# training/trainers/hf_trainer.py

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

from training.trainers.base_trainer import BaseTrainer
import os
import pandas as pd
from datasets import Dataset

from training.trainers.custom_loss import WeightedLossTrainer


import numpy as np



class HuggingFaceTrainer(BaseTrainer):

    def __init__(
        self,
        name,
        pretrained_name: str,
        tokenizer_name: str,
        num_labels: int,
        max_length: int,
        artifact_dir: str,
        labels: dict,
        **kwargs
    ):

        self.name=name
        self.pretrained_name = pretrained_name
        self.tokenizer_name = tokenizer_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.artifact_dir=artifact_dir
        self.labels=labels

        self.model = None
        self.tokenizer = None

    def _build_tokenizer(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name
        )

        return self.tokenizer

    def _build_model(self):

        base_model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True
            )
        )

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense"]
        )

        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()

        return self.model

    def _build_trainer(
        self,
        train_dataset=None,
        eval_dataset=None
    ):

        training_args = TrainingArguments(
         output_dir="artifacts/checkpoints",
         #max_steps=10,
         num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        #eval_strategy="epoch",
        #eval_steps=10,
        #save_strategy="epoch",
        #save_steps=10,
        save_strategy="steps",
        save_steps=100,
        logging_dir="artifacts/logs",
        report_to="none",
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        include_num_input_tokens_seen=False,
        )
        trainer = WeightedLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        return trainer

    def train(self, dataset_cfg):

        print(
            f"Loading tokenizer: {self.tokenizer_name}"
        )

        self._build_tokenizer()

        print(
            f"Loading model: {self.pretrained_name}"
        )

        self._build_model()

        print(
            "Loading processed datasets..."
        )

        train_df = pd.read_csv(
            dataset_cfg.processed_train_path
        )

        val_df = pd.read_csv(
            dataset_cfg.processed_val_path
        )

        print(
            f"Train samples: {len(train_df)}"
        )

        print(
            f"Validation samples: {len(val_df)}"
        )

        print(
            f"Training {self.pretrained_name}"
        )

        
        # Convert train_df and val_df
        # into HuggingFace Dataset objects

        train_dataset = Dataset.from_pandas( train_df, preserve_index=False ) 
        val_dataset = Dataset.from_pandas( val_df, preserve_index=False )

        
        # Tokenize text column
        def tokenize_function(examples): 
            return self.tokenizer( examples["text"], truncation=True, padding="max_length", max_length=self.max_length )

        
        # Build Trainer using tokenized datasets
        train_dataset = train_dataset.map( tokenize_function, batched=True )

        val_dataset = val_dataset.map( tokenize_function, batched=True )

        train_dataset = train_dataset.rename_column( "label", "labels" )
        val_dataset = val_dataset.rename_column( "label", "labels" )

        train_dataset.set_format( type="torch", columns=[ "input_ids", "attention_mask", "labels" ] )
        val_dataset.set_format( type="torch", columns=[ "input_ids", "attention_mask", "labels" ] )


        trainer = self._build_trainer( train_dataset=train_dataset, eval_dataset=val_dataset )

        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        print("[TRAIN_CALL] Starting trainer.train()")
        trainer.train(resume_from_checkpoint=False)
        print("[TRAIN_CALL] trainer.train() completed")
        os.makedirs(
            self.artifact_dir,
            exist_ok=True
        )

        trainer.save_model(
            self.artifact_dir
        )

        self.tokenizer.save_pretrained(
            self.artifact_dir
        )

        return trainer

