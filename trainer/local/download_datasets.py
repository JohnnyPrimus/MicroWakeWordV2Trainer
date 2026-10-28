"""Step: download background/negative audio data for augmentation and training.

Ports the Colab notebook's dataset-download cells (which existed twice,
near-identically, as cell-4 and cell-7 - this keeps the later/corrected
version) to plain local functions, with two deliberate departures from the
notebook and from microWakeWord's own upstream notebook:

1. It drops the notebook's bespoke FileMapper/dataclass resume-tracking
   machinery in favor of a simple "does the destination directory already
   have files in it" skip check, which is all a one-workspace-per-wake-word
   pipeline needs.

2. It does NOT use the `datasets` library's automatic audio decoding. All
   three source datasets have been restructured since the notebooks were
   written (Audioset now ships as parquet with a `List` feature type; FMA's
   `mchl914/fma_xsmall` dropped its `16k/*.wav` files in favor of a zip), and
   pulling in a `datasets` version new enough to read them requires the
   `torchcodec` package for audio decode/encode, which in turn requires a
   system FFmpeg install that isn't available by default on Windows. Since
   none of that machinery is actually needed here, this instead downloads
   files directly via `huggingface_hub` and decodes audio with `soundfile`
   (bundles libsndfile, no FFmpeg needed).

IMPORTANT: the RIR/Audioset/FMA data downloaded here carries a mixture of
licenses and usage restrictions (see microWakeWord's documentation/data_sources.md).
Models trained with it should be treated as personal-use only unless you've
checked the license of each source for commercial use.
"""

from __future__ import annotations

import io
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile
import scipy.signal
import soundfile as sf

from trainer.local.config import TrainerConfig

NEGATIVE_DATASET_FILES = ["dinner_party.zip", "dinner_party_eval.zip", "no_speech.zip", "speech.zip"]

RIR_REPO = "davidscripka/MIT_environmental_impulse_responses"
FMA_REPO = "mchl914/fma_xsmall"
AUDIOSET_REPO = "agkphysics/Audioset"
NEGATIVE_DATASET_REPO = "kahrendt/microwakeword"
DEFAULT_DOWNLOAD_RETRIES = 10
DEFAULT_DOWNLOAD_TIMEOUT = 120
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _hf_token(config: TrainerConfig) -> Optional[str]:
    return os.environ.get(config.datasets.hf_token_env) or None


def _hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _negative_dataset_url(filename: str) -> str:
    return f"{_hf_endpoint()}/datasets/{NEGATIVE_DATASET_REPO}/resolve/main/{filename}"


def _dir_has_files(path: Path) -> bool:
    return path.exists() and any(path.glob("*.wav"))


def _download_resumable(
    url: str,
    destination: Path,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
) -> None:
    """Download a file, resuming from a partial destination after interruption."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        offset = destination.stat().st_size if destination.exists() else 0
        headers = {"User-Agent": "microWakeWordV2Trainer/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            print(f"Resuming download at {offset} bytes (attempt {attempt}/{retries})")
        elif attempt > 1:
            print(f"Retrying download (attempt {attempt}/{retries})")

        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                # A mirror or proxy may ignore Range and return the whole file.
                # In that case truncate and write from the beginning.
                append = offset > 0 and response.status == 206
                if offset and not append:
                    print("Server did not honor the resume request; restarting this file")
                    offset = 0

                expected_size = offset
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    expected_size += int(content_length)

                with destination.open("ab" if append else "wb") as output:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())

            actual_size = destination.stat().st_size
            if content_length is not None and actual_size != expected_size:
                raise OSError(
                    f"incomplete response: got {actual_size} bytes, expected {expected_size}"
                )
            return
        except Exception as exc:
            current_size = destination.stat().st_size if destination.exists() else 0
            if attempt == retries:
                raise RuntimeError(
                    f"download failed after {retries} attempts at {current_size} bytes: {url}"
                ) from exc
            delay = min(2 ** (attempt - 1), 30)
            print(f"Download interrupted at {current_size} bytes: {exc}; retrying in {delay}s")
            time.sleep(delay)


def _write_wav_16k(array: np.ndarray, orig_sr: int, dest: Path) -> None:
    if array.ndim > 1:
        array = array.mean(axis=1)  # downmix to mono
    if orig_sr != 16000:
        gcd = np.gcd(orig_sr, 16000)
        array = scipy.signal.resample_poly(array, 16000 // gcd, orig_sr // gcd)
    array = np.clip(array, -1.0, 1.0)
    scipy.io.wavfile.write(str(dest), 16000, (array * 32767).astype(np.int16))


def download_mit_rirs(config: TrainerConfig) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    print("Downloading MIT environmental impulse response dataset...")
    token = _hf_token(config)
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(RIR_REPO, repo_type="dataset") if f.startswith("16khz/") and f.endswith(".wav")]

    config.rir_dir.mkdir(parents=True, exist_ok=True)
    missing = [filename for filename in files if not (config.rir_dir / Path(filename).name).exists()]
    if not missing:
        print(f"{config.rir_dir} already populated, skipping MIT RIR download")
        return

    for i, filename in enumerate(files, 1):
        destination = config.rir_dir / Path(filename).name
        if destination.exists():
            continue
        local_path = hf_hub_download(repo_id=RIR_REPO, repo_type="dataset", filename=filename, token=token)
        shutil.copyfile(local_path, destination)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} RIR clips")
    print(f"Wrote {len(files)} RIR clips to {config.rir_dir}")


def download_fma(config: TrainerConfig) -> None:
    if _dir_has_files(config.fma_dir):
        print(f"{config.fma_dir} already populated, skipping FMA download")
        return

    from huggingface_hub import hf_hub_download

    print("Downloading Free Music Archive (xsmall) dataset...")
    token = _hf_token(config)
    zip_path = hf_hub_download(repo_id=FMA_REPO, repo_type="dataset", filename="fma_xs.zip", token=token)

    extract_dir = config.workspace / "_fma_extract_tmp"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    config.fma_dir.mkdir(parents=True, exist_ok=True)
    audio_files = [p for p in extract_dir.rglob("*") if p.suffix.lower() in (".mp3", ".wav", ".flac")]
    written = 0
    for i, path in enumerate(audio_files, 1):
        try:
            array, sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            print(f"  skipping {path.name}: {e}")
            continue
        _write_wav_16k(array, sr, config.fma_dir / f"{path.stem}.wav")
        written += 1
        if i % 500 == 0 or i == len(audio_files):
            print(f"  {i}/{len(audio_files)} FMA clips processed")

    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"Wrote {written} FMA clips to {config.fma_dir}")


def download_audioset(config: TrainerConfig) -> None:
    if _dir_has_files(config.audioset_dir):
        print(f"{config.audioset_dir} already populated, skipping Audioset download")
        return

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    split = config.datasets.audioset_split
    if not split:
        raise ValueError(
            "datasets.audioset_split is null (full dataset). This pipeline only "
            "downloads one split at a time - set it to e.g. 'bal_train'."
        )

    token = _hf_token(config)
    api = HfApi(token=token)
    all_files = api.list_repo_files(AUDIOSET_REPO, repo_type="dataset")
    shard_files = sorted(f for f in all_files if f.startswith(f"data/{split}/") and f.endswith(".parquet"))
    if not shard_files:
        raise ValueError(f"No parquet shards found under data/{split}/ in {AUDIOSET_REPO}")

    print(f"Downloading Audioset ({split}, {len(shard_files)} shards)...")
    config.audioset_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for shard_i, shard in enumerate(shard_files, 1):
        local_path = hf_hub_download(repo_id=AUDIOSET_REPO, repo_type="dataset", filename=shard, token=token)
        parquet_file = pq.ParquetFile(local_path)
        for batch in parquet_file.iter_batches(batch_size=256, columns=["audio"]):
            for value in batch.column("audio").to_pylist():
                if not value or not value.get("bytes"):
                    continue
                try:
                    array, sr = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
                except Exception:
                    continue
                count += 1
                _write_wav_16k(array, sr, config.audioset_dir / f"{count}.wav")
        print(f"  shard {shard_i}/{len(shard_files)} done, {count} clips so far")

    print(f"Wrote {count} Audioset clips to {config.audioset_dir}")


def download_negative_features(config: TrainerConfig) -> None:
    config.negative_datasets_dir.mkdir(parents=True, exist_ok=True)

    for filename in NEGATIVE_DATASET_FILES:
        subdir_name = filename[: -len(".zip")]
        dest_dir = config.negative_datasets_dir / subdir_name
        completion_marker = config.negative_datasets_dir / f".{subdir_name}.complete"
        zip_path = config.negative_datasets_dir / filename

        if dest_dir.exists():
            # Older versions created no completion marker. If their archive is
            # already gone, treat the directory as a completed legacy extract.
            if completion_marker.exists() or not zip_path.exists():
                print(f"{dest_dir} already exists, skipping {filename}")
                continue
            print(f"{dest_dir} was only partially extracted; extracting it again")
            shutil.rmtree(dest_dir)

        url = _negative_dataset_url(filename)
        print(f"Downloading {url} -> {zip_path}")
        archive_ready = False
        if zip_path.exists():
            try:
                with zipfile.ZipFile(zip_path):
                    archive_ready = True
            except zipfile.BadZipFile:
                print(
                    f"Found incomplete archive ({zip_path.stat().st_size} bytes); "
                    "attempting to resume it"
                )
        if archive_ready:
            print(f"{zip_path} is already a complete archive; skipping download")
        else:
            _download_resumable(
                url,
                zip_path,
                retries=max(1, config.datasets.download_retries),
            )

        print(f"Extracting {zip_path}")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(config.negative_datasets_dir)
        except zipfile.BadZipFile:
            print("Downloaded archive is invalid; removing it and retrying from zero")
            zip_path.unlink(missing_ok=True)
            _download_resumable(
                url,
                zip_path,
                retries=max(1, config.datasets.download_retries),
            )
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(config.negative_datasets_dir)

        completion_marker.touch()
        zip_path.unlink(missing_ok=True)


def run(config: TrainerConfig) -> None:
    proxy_configured = bool(
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
    )
    print(f"Using Hugging Face endpoint: {_hf_endpoint()}")
    print(f"Network proxy: {'configured' if proxy_configured else 'not configured'}")
    download_mit_rirs(config)
    download_audioset(config)
    download_fma(config)
    download_negative_features(config)
