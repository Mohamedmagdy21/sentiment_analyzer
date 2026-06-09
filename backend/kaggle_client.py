import json
import os
import subprocess
import time

from dotenv import load_dotenv


class KaggleClient:

    def __init__(self):

        load_dotenv()

        self.api_token = os.getenv(
            "KAGGLE_API_TOKEN"
        )
        self._kernel_id = None

        if not self.api_token:
            raise ValueError(
                "KAGGLE_API_TOKEN not found in .env"
            )

        os.environ["KAGGLE_API_TOKEN"] = (
            self.api_token
        )

    def push_kernel(
        self,
        kernel_dir: str
    ):

        metadata_path = os.path.join(
            kernel_dir,
            "kernel-metadata.json"
        )

        if not os.path.exists(
            kernel_dir
        ):
            raise FileNotFoundError(
                f"Kernel directory not found: {kernel_dir}"
            )

        with open(
            metadata_path
        ) as f:
            metadata = json.load(f)

        self._kernel_id = metadata["id"]

        result = subprocess.run(
            [
                "kaggle",
                "kernels",
                "push"
            ],
            cwd=kernel_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:

            raise RuntimeError(
                f"""
                Kaggle push failed.

                STDOUT:
                {result.stdout}

                STDERR:
                {result.stderr}
                """
            )

        print(
            "Kernel pushed successfully."
        )

        print(
            result.stdout
        )

        return result.stdout

    def get_status(
        self
    ):

        if not self._kernel_id:
            raise ValueError(
                "No kernel pushed yet."
            )

        result = subprocess.run(
            [
                "kaggle",
                "kernels",
                "status",
                self._kernel_id
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"""
                Failed to get kernel status.

                STDERR:
                {result.stderr}
                """
            )

        return result.stdout.strip()

    def wait_for_completion(
        self,
        interval: int = 30
    ):

        if not self._kernel_id:
            raise ValueError(
                "No kernel pushed yet."
            )

        while True:
            status = self.get_status()
            print(
                f"Status: {status}"
            )

            if status == "complete":
                print(
                    "Kernel completed successfully."
                )
                return status

            if status in (
                "error",
                "failed"
            ):
                raise RuntimeError(
                    f"Kernel {self._kernel_id} "
                    f"ended with status: {status}"
                )

            time.sleep(interval)

    def download_output(
        self,
        kernel_ref: str,
        output_dir: str
    ):

        result = subprocess.run(
            [
                "kaggle",
                "kernels",
                "output",
                kernel_ref,
                "-p",
                output_dir,
                "-o"
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:

            raise RuntimeError(
                f"""
                Failed to download kernel output.

                STDOUT:
                {result.stdout}

                STDERR:
                {result.stderr}
                """
            )

        print(result.stdout)

        return output_dir