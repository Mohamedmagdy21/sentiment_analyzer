from sklearn.model_selection import train_test_split
import pandas as pd
import os

from preprocessing.processors.BaseProcessor import BaseProcessor


class TwitterPreprocessor(BaseProcessor):

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

        df_twitter = self.load_data(dataset_cfg)
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(
            df_twitter
        )
        self.save_splits(X_train, X_val, X_test, y_train, y_val, y_test, dataset_cfg)

    def load_data(self, dataset_cfg):

        df = pd.read_csv(
            dataset_cfg.raw_path,
            encoding=dataset_cfg.encoding,
            header=None
        )

        df.columns = [
            "target",
            "ids",
            "date",
            "flag",
            "user",
            "text"
        ]

        df = df.rename(
            columns={"target": "sentiment"}
        )

        # Sentiment140:
        # 0 = Negative
        # 4 = Positive

       # df["sentiment"] = df["sentiment"].replace(
        #    {
        #        0: 0,
         #       4: 1
         #   }
        #)

        df["sentiment"] = df["sentiment"].replace({
         0: 0,  # negative
         2: 1,  # neutral
         4: 2,  # positive
        })

        df_org = df

        return df_org

    def preprocess_data(self, data):

        stratify_col = (
            data["sentiment"]
            if self.stratify
            else None
        )

        X_train, X_temp, y_train, y_temp = train_test_split(
            data["text"],
            data["sentiment"],
            test_size=self.test_validation_split,
            random_state=self.random_state,
            stratify=stratify_col
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
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
