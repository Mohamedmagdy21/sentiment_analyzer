from backend.kaggle_client import KaggleClient


def main():

    print("Starting test...")

    client = KaggleClient()

    print("Client created")

    client.push_kernel(
        "kaggle/twitter_training"
    )

    print("Push finished")


if __name__ == "__main__":
    main()