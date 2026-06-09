import json
import os
import re
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
        self._kernel_ref = None

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
        self._kernel_ref = metadata["id"]

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

        if "error" in result.stdout.lower():
            # Push reported an error even with rc=0
            raise RuntimeError(
                f"""
                Kaggle push reported error.

                STDOUT:
                {result.stdout}
                """
            )

        # Parse version number: "Kernel version N successfully pushed."
        m = re.search(
            r"Kernel version\s+(\d+)",
            result.stdout
        )
        if m:
            version = m.group(1)
            self._kernel_ref = f"{metadata['id']}/{version}"
            print(
                f"Pushed version {version}, "
                f"tracking: {self._kernel_ref}"
            )
        else:
            print(
                "Kernel pushed (version unknown)"
            )

        print(
            result.stdout
        )

        return result.stdout

    def get_status(
        self
    ):

        if not self._kernel_ref:
            raise ValueError(
                "No kernel pushed yet."
            )

        result = subprocess.run(
            [
                "kaggle",
                "kernels",
                "status",
                self._kernel_ref
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

        raw = result.stdout.strip()
        # Parse: "owner/slug has status \"KernelWorkerStatus.VALUE\""
        if '"' in raw:
            status = raw.split('"')[1]
            if "." in status:
                status = status.split(".")[1]
        else:
            status = raw
        return status

    def wait_for_completion(
        self,
        interval: int = 60,
        timeout: int = 28800,
    ):

        if not self._kernel_ref:
            raise ValueError(
                "No kernel pushed yet."
            )

        deadline = time.time() + timeout

        while True:
            if time.time() > deadline:
                raise TimeoutError(
                    f"Kernel {self._kernel_ref} "
                    f"did not complete within {timeout}s"
                )

            status = self.get_status()
            print(
                f"Status: {status}"
            )

            if status.upper() == "COMPLETE":
                print(
                    "Kernel completed successfully."
                )
                return status

            terminal = (
                "ERROR",
                "FAILED",
                "CANCEL_ACKNOWLEDGED",
                "CANCELLED",
                "CANCELED",
                "KILLED",
                "TIMED_OUT",
            )
            if status.upper() in terminal:
                raise RuntimeError(
                    f"Kernel {self._kernel_ref} "
                    f"ended with terminal status: {status}"
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

    def download_log(
        self,
        kernel_ref: str,
        output_dir: str
    ):

        os.makedirs(output_dir, exist_ok=True)

        log_path = os.path.join(
            output_dir,
            f"{kernel_ref.replace('/', '-')}.log"
        )

        result = subprocess.run(
            [
                "kaggle",
                "kernels",
                "logs",
                kernel_ref,
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:

            raise RuntimeError(
                f"""
                Failed to download kernel logs.

                STDOUT:
                {result.stdout}

                STDERR:
                {result.stderr}
                """
            )

        with open(log_path, "w") as f:
            f.write(result.stdout)

        print(
            f"Kernel log saved to {log_path}"
        )

        return log_path

    def upload_dataset(
        self,
        dataset_dir: str,
        version_notes: str = "Automated update",
    ):
        metadata_path = os.path.join(
            dataset_dir, "dataset-metadata.json"
        )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"dataset-metadata.json not found in {dataset_dir}"
            )

        with open(metadata_path) as f:
            metadata = json.load(f)
        dataset_id = metadata["id"]

        print(f"Uploading dataset {dataset_id} from {dataset_dir}")

        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", dataset_dir],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"Dataset {dataset_id} created successfully.")
            print(result.stdout)
            return

        print(f"Dataset may already exist; attempting version update...")
        print(result.stdout)
        print(result.stderr)

        result = subprocess.run(
            [
                "kaggle", "datasets", "version",
                "-p", dataset_dir,
                "-m", version_notes,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to upload dataset {dataset_id}.\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        print(f"Dataset {dataset_id} versioned successfully.")
        print(result.stdout)