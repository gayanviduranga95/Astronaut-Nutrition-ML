
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import time
import socket

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

RAW.mkdir(parents=True, exist_ok=True)


# ============================================================
# NHANES 2017-2018 DATASETS
# ============================================================

FILES = {
    "BIOPRO_J": "Laboratory",
    "GHB_J": "Laboratory",
    "FERTIN_J": "Laboratory",
    "HDL_J": "Laboratory",
    "TCHOL_J": "Laboratory",
    "TRIGLY_J": "Laboratory",
    "ALB_CR_J": "Laboratory",
    "BPX_J": "Examination",
}

BASE_URL = (
    "https://wwwn.cdc.gov/Nchs/Data/"
    "Nhanes/Public/2017/DataFiles"
)

MAX_RETRIES = 5
TIMEOUT_SECONDS = 90


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_file(url, destination):
    """
    Download a dataset with automatic retries.

    Existing valid files are reused.
    Incomplete downloads are not saved as final files.
    """

    if destination.exists():
        print("File already exists. Checking local copy.")
        return

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/octet-stream,*/*",
        },
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"Download attempt "
                f"{attempt}/{MAX_RETRIES}"
            )

            with urlopen(
                request,
                timeout=TIMEOUT_SECONDS,
            ) as response:
                data = response.read()

            # SAS XPORT files begin with a HEADER RECORD.
            if not data.startswith(b"HEADER RECORD"):
                raise ValueError(
                    "Downloaded content is not "
                    "a valid SAS XPORT header"
                )

            temporary = destination.with_suffix(".part")
            temporary.write_bytes(data)

            temporary.replace(destination)

            print(
                "Download successful:",
                round(len(data) / (1024 * 1024), 2),
                "MB",
            )

            return

        except HTTPError as error:
            print(
                f"HTTP error {error.code}: "
                f"{error.reason}"
            )

            # A missing file or forbidden resource is
            # unlikely to improve with repeated requests.
            if error.code in (400, 401, 403, 404):
                raise

        except (
            URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            OSError,
            ValueError,
        ) as error:
            print("Attempt failed:", error)

        if attempt < MAX_RETRIES:
            wait_seconds = 5 * attempt

            print(
                f"Waiting {wait_seconds} seconds "
                "before retry..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Download failed after {MAX_RETRIES} attempts"
    )


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_dataset(destination):
    """
    Check that the downloaded SAS XPORT file
    can be read and contains valid participant IDs.
    """

    df = pd.read_sas(
        destination,
        format="xport",
    )

    if "SEQN" not in df.columns:
        raise ValueError(
            "Dataset does not contain SEQN"
        )

    if df["SEQN"].isna().any():
        raise ValueError(
            "Dataset contains missing SEQN values"
        )

    duplicates = int(
        df["SEQN"].duplicated().sum()
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Unique SEQN:", df["SEQN"].nunique())
    print("Duplicates:", duplicates)

    if duplicates > 0:
        print(
            "WARNING: Duplicate SEQN detected. "
            "Do not perform a one-to-one merge "
            "without investigating."
        )

    return df


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    print("=" * 65)
    print("ASTRONAUT NUTRITION ML")
    print("STEP 36: ADDITIONAL LABORATORY DATASETS")
    print("=" * 65)

    successful = []
    failed = []

    for name, category in FILES.items():
        print("\n" + "=" * 65)
        print("Processing:", name)
        print("Category:", category)
        print("=" * 65)

        destination = RAW / f"{name}.xpt"
        url = f"{BASE_URL}/{name}.xpt"

        try:
            download_file(url, destination)

            try:
                verify_dataset(destination)

            except Exception:
                # A bad local copy should not be silently
                # accepted on the next run.
                if destination.exists():
                    invalid_path = destination.with_suffix(
                        ".invalid"
                    )
                    destination.replace(invalid_path)

                    print(
                        "Invalid file moved to:",
                        invalid_path,
                    )

                raise

            successful.append(name)

            print("Status: VERIFIED")

        except Exception as error:
            failed.append(name)

            print("Status: FAILED")
            print("Reason:", error)

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)

    print("Successful:", len(successful))
    print("Failed:", len(failed))

    if successful:
        print("\nVerified datasets:")
        for name in successful:
            print("  [OK]", name)

    if failed:
        print("\nFailed datasets:")
        for name in failed:
            print("  [FAILED]", name)

    print("\nRaw dataset directory:", RAW)

    if failed:
        print(
            "\nSome downloads failed. "
            "Check the network connection or "
            "download those files manually."
        )
    else:
        print(
            "\nAll additional datasets "
            "downloaded and verified successfully!"
        )


if __name__ == "__main__":
    main()
