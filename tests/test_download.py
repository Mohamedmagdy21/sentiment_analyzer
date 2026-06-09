# tests/test_download.py

from backend.kaggle_client import KaggleClient


def main():

    client = KaggleClient()

    client.download_output(
        kernel_ref="mohamedmagdyw/twitter-training",
        output_dir="downloads/twitter"
    )

    print("Download completed")


if __name__ == "__main__":
    main()