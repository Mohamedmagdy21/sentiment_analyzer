
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from preprocessing.processors.BaseProcessor import BaseProcessor


class AmazonPreprocessor(BaseProcessor):

    def __init__(
        self,
        test_validation_split: float,
        random_state: int,
        stratify: bool
    ):
        self.test_validation_split = test_validation_split
        self.random_state = random_state
        self.stratify = stratify

    def run(self, dataset_cfg):

        df = self.load_data(dataset_cfg)

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        ) = self.preprocess_data(df,dataset_cfg)

        self.save_splits(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            dataset_cfg
        )

    def load_data(self, dataset_cfg):

        df = pd.read_csv(
            dataset_cfg.raw_path,
            encoding=dataset_cfg.encoding
        )

        df = df[
            [
                dataset_cfg.text_column,
                dataset_cfg.label_column
            ]
        ]

        df = df.dropna()

        return df

    def preprocess_data(self, data,dataset_cfg):

        text_col = dataset_cfg.text_column
        label_col = dataset_cfg.label_column

        #data = data[
        #    data[label_col] != 3
        #].copy()

        

       # data[label_col] = (
       #     data[label_col] >= 4
       # ).astype(int)

        data[label_col] = data[label_col].replace({
         1: 0,
         2: 0
        })

        data[label_col]=data[label_col].replace({3:1})

        data[label_col] = data[label_col].replace({
         4: 2,
         5: 2
        })





        stratify_col = (
            data[label_col]
            if self.stratify
            else None
        )
        

        X_train, X_temp, y_train, y_temp = train_test_split(
            data[text_col],
            data[label_col],
            test_size=self.test_validation_split,
            random_state=self.random_state,
            stratify=stratify_col
        )

        stratify_temp = (
            y_temp
            if self.stratify
            else None
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=self.random_state,
            stratify=stratify_temp
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

    def save_splits(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        dataset_cfg
    ):

        train_df = pd.DataFrame(
            {
                "text": X_train,
                "label": y_train
            }
        )

        val_df = pd.DataFrame(
            {
                "text": X_val,
                "label": y_val
            }
        )

        test_df = pd.DataFrame(
            {
                "text": X_test,
                "label": y_test
            }
        )

        os.makedirs(
            os.path.dirname(
                dataset_cfg.processed_train_path
            ),
            exist_ok=True
        )

        train_df.to_csv(
            dataset_cfg.processed_train_path,
            index=False
        )

        val_df.to_csv(
            dataset_cfg.processed_val_path,
            index=False
        )

        test_df.to_csv(
            dataset_cfg.processed_test_path,
            index=False
        )

        print(
            f"Train saved to: {dataset_cfg.processed_train_path}"
        )

        print(
            f"Validation saved to: {dataset_cfg.processed_val_path}"
        )

        print(
            f"Test saved to: {dataset_cfg.processed_test_path}"
        )

        print(
            f"Train samples: {len(train_df)}"
        )

        print(
            f"Validation samples: {len(val_df)}"
        )

        print(
            f"Test samples: {len(test_df)}"
        )

