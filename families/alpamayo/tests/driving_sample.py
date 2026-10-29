# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned public driving example, streamed once and reusable offline."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image


def load_sample(directory, specification):
    """Validate raw inputs; never accept generated model output as ground truth."""
    directory = Path(directory)
    provenance = json.loads((directory / "sample-manifest.json").read_text())
    for key in ("clip_id", "t0_us", "dataset", "dataset_revision", "provider_revision"):
        assert provenance[key] == specification[key], f"sample provenance differs: {key}"
    assert specification["frame_count"] == len(specification["camera_indices"]) * 4
    for i in range(specification["frame_count"]):
        with Image.open(directory / f"frame-{i:02d}.png") as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape == (1080, 1920, 3)
        assert list(pixels.shape) == specification["frame_shape"]
        assert pixels.dtype == np.uint8
    with np.load(directory / "state-and-ground-truth.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert data["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert data["ego_history_rot"].shape == (1, 1, 16, 3, 3)
    assert data["ego_future_xyz"].shape == (1, 1, 64, 3)
    assert data["camera_indices"].tolist() == specification["camera_indices"]
    assert all(np.isfinite(value).all() for value in data.values())
    assert set(data) == set(specification["state_arrays"])
    for key, layout in specification["state_arrays"].items():
        assert list(data[key].shape) == layout["shape"], key
        assert str(data[key].dtype) == layout["dtype"], key
    return data


def prepare_sample(directory, specification):
    """Use the independent provider loader and bounded, revision-pinned reads."""
    import physical_ai_av
    import torch
    from huggingface_hub.utils import get_session, hf_raise_for_status

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "sample-manifest.json").exists():
        return load_sample(directory, specification)
    revision = specification["provider_revision"]
    url = (
        f"https://raw.githubusercontent.com/{specification['provider_repository']}/"
        f"{revision}/src/alpamayo_r1/load_physical_aiavdataset.py"
    )
    with urlopen(url, timeout=60) as response:
        source = response.read(64 * 1024)
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    path = directory / "provider_sample_loader.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("alpamayo_public_sample_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class PinnedDataset(physical_ai_av.PhysicalAIAVDatasetInterface):
        def __init__(self):
            self.metadata_bytes = 0
            self.range_bytes = 0
            self.features_opened = 0
            super().__init__(
                revision=specification["dataset_revision"],
                cache_dir=directory / "cache",
                local_dir=directory / "metadata",
                confirm_download_threshold_gb=0.125,
            )

        def download_file(self, filename, **kwargs):
            assert filename in {
                "features.csv",
                "clip_index.parquet",
                "metadata/feature_presence.parquet",
                "metadata/sensor_presence.parquet",
                "metadata/data_collection.parquet",
            }
            item = self.api.get_paths_info(paths=[filename], **self.repo_snapshot_info)[0]
            self.metadata_bytes += item.size
            assert item.size <= 64 * 1024**2 and self.metadata_bytes <= 128 * 1024**2
            return super().download_file(filename, **kwargs)

        @contextmanager
        def open_file(self, filename, mode="rb", maybe_stream=False):
            assert mode == "rb" and maybe_stream and filename.endswith(".zip")
            self.features_opened += 1
            assert self.features_opened <= 5
            with self.fs.open(
                f"datasets/{self.repo_id}/{filename}",
                mode,
                revision=self.revision,
                block_size=1024**2,
            ) as stream:
                assert stream.resolved_path.revision == specification["dataset_revision"]

                def fetch_range(start, end):
                    assert 0 <= start < end and end - start <= 128 * 1024**2
                    assert self.range_bytes + end - start <= 512 * 1024**2
                    headers = {
                        "Range": f"bytes={start}-{end - 1}",
                        **self.fs._api._build_hf_headers(),
                    }
                    chunks, received = [], 0
                    with get_session().stream(
                        "GET", stream.url(), headers=headers, follow_redirects=True, timeout=60
                    ) as response:
                        hf_raise_for_status(response)
                        assert response.status_code == 206, "refuse full-archive downloads"
                        assert response.headers.get("content-range", "").startswith(
                            f"bytes {start}-"
                        )
                        for chunk in response.iter_bytes(chunk_size=1024**2):
                            received += len(chunk)
                            assert received <= end - start
                            chunks.append(chunk)
                    assert received == end - start
                    self.range_bytes += received
                    return b"".join(chunks)

                stream._fetch_range = fetch_range
                stream.cache.fetcher = fetch_range
                yield stream

    values = module.load_physical_aiavdataset(
        specification["clip_id"],
        t0_us=specification["t0_us"],
        avdi=PinnedDataset(),
        maybe_stream=True,
    )
    frames = values["image_frames"]
    assert frames.shape == (4, 4, 3, 1080, 1920)
    for i, frame in enumerate(frames.flatten(0, 1)):
        Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()).save(directory / f"frame-{i:02d}.png")
    arrays = {
        key: value.cpu().numpy()
        for key, value in values.items()
        if isinstance(value, torch.Tensor) and key != "image_frames"
    }
    np.savez(directory / "state-and-ground-truth.npz", **arrays)
    (directory / "sample-manifest.json").write_text(json.dumps(specification, indent=2) + "\n")
    return load_sample(directory, specification)
