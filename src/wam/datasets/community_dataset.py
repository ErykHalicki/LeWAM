from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


class CommunityDataset:
    """All subdatasets in a community repo, loaded from a pre-downloaded local directory.

    Download first with CommunityDataset.download(), then build the per-camera datasets:

        CommunityDataset.download(repo_id="ehalicki/LeWAM_community_dataset", cache_root="/data/lewam")
        cd = CommunityDataset(repo_id="ehalicki/LeWAM_community_dataset", cache_root="/data/lewam")
        cd.prefetch_metadata()
        # cd.datasets: dict[int, MultiLeRobotDataset] keyed by camera count (1, 2, 3)
    """

    def __init__(
        self,
        repo_id: str,
        cache_root: str | Path | None = None,
        revision: str | None = None,
    ):
        self.repo_id = repo_id
        self.cache_root = cache_root
        self.revision = revision if revision else "main"

        self.download(repo_id, cache_root, self.revision)
        self.subpaths = self._discover_subpaths_local(repo_id, cache_root)
        self.datasets: dict[int, MultiLeRobotDataset] = {}

    @classmethod
    def download(
        cls,
        repo_id: str,
        cache_root: str | Path | None = None,
        revision: str = "main",
    ):
        """Download the full dataset to local storage. Skips if data already present."""
        from huggingface_hub import snapshot_download
        base = Path(cache_root) if cache_root else HF_LEROBOT_HOME
        local_dir = base / repo_id
        if local_dir.exists() and any(local_dir.rglob("*.parquet")):
            print(f"Dataset already present at {local_dir}, skipping download.")
            return
        local_dir.mkdir(exist_ok=True, parents=True)
        print(f"Downloading {repo_id} to {local_dir} ...")
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
        )
        print("Download complete.")

    @staticmethod
    def _discover_subpaths_local(repo_id: str, cache_root: str | Path | None = None) -> list[str]:
        base = Path(cache_root) if cache_root else HF_LEROBOT_HOME
        local_dir = base / repo_id
        if not local_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {local_dir}. Run CommunityDataset.download() first."
            )
        return [
            meta_dir.parent.relative_to(local_dir).as_posix()
            for meta_dir in sorted(local_dir.glob("*/*/meta"))
            if meta_dir.is_dir()
        ]

    @staticmethod
    def _expand_camera_timestamps(delta_timestamps: dict, n_cams: int) -> dict:
        cam_keys = ["observation.images.image"] + [f"observation.images.image{i}" for i in range(2, n_cams + 1)]
        base_ts = next((v for k, v in delta_timestamps.items() if k.startswith("observation.images")), None)
        non_cam = {k: v for k, v in delta_timestamps.items() if not k.startswith("observation.images")}
        if base_ts is None:
            return delta_timestamps
        return {**non_cam, **{k: base_ts for k in cam_keys}}

    def _local_root(self) -> Path:
        base = Path(self.cache_root) if self.cache_root else HF_LEROBOT_HOME
        return base / self.repo_id

    def prefetch_metadata(self, **dataset_kwargs) -> dict[int, MultiLeRobotDataset]:
        """Load metadata, filter to action_dim=6, and build one MultiLeRobotDataset per camera count."""
        metas = {}
        for subpath in self.subpaths:
            meta = LeRobotDatasetMetadata(self.repo_id, self._local_root() / subpath, self.revision)
            metas[subpath] = meta
        print(f"loaded metadata for {len(metas)}/{len(self.subpaths)} datasets")

        metas = {sp: m for sp, m in metas.items() if m.shapes["action"][0] == 6}
        print(f"Kept {len(metas)}/{len(self.subpaths)} single-arm datasets (action_dim=6)")
        metas = {sp: m for sp, m in metas.items() if m.fps == 30}
        print(f"Kept {len(metas)}/{len(self.subpaths)} datasets with fps=30")
        metas = {sp: m for sp, m in metas.items() if "observation.images.image" in list(m.camera_keys)}
        print(f"Kept {len(metas)}/{len(self.subpaths)} datasets with observation.images.image")
        self.metas = metas

        buckets: dict[int, list[str]] = {}
        for subpath, meta in metas.items():
            n = len(list(meta.camera_keys))
            buckets.setdefault(n, []).append(subpath)

        root = self._local_root()
        for n_cams, subpaths in buckets.items():
            print(f"Building MultiLeRobotDataset for {n_cams} camera(s) with {len(subpaths)} subdatasets")
            kw = dict(dataset_kwargs)
            if "delta_timestamps" in kw and kw["delta_timestamps"]:
                kw["delta_timestamps"] = self._expand_camera_timestamps(kw["delta_timestamps"], n_cams)
            self.datasets[n_cams] = MultiLeRobotDataset(
                repo_ids=subpaths,
                root=root,
                **kw,
            )

        return self.datasets
