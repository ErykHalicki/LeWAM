import random
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
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

    def load_metadata(self):
        """Load and filter metadata (cheap, no dataset construction). Populates self.metas and self.buckets."""
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

        self.buckets: dict[int, list[str]] = {}
        for subpath, meta in metas.items():
            n = len(list(meta.camera_keys))
            self.buckets.setdefault(n, []).append(subpath)

    def split_episodes(
        self,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Deterministic per-sub-dataset episode split. Requires load_metadata() first."""
        train_eps: dict[str, list[int]] = {}
        val_eps: dict[str, list[int]] = {}
        for subpath, meta in self.metas.items():
            all_ep = list(range(meta.total_episodes))
            if len(all_ep) < 2:
                print(f"  {subpath}: only {len(all_ep)} episode(s), all go to train")
                train_eps[subpath] = all_ep
                val_eps[subpath] = []
                continue
            rng = random.Random(seed)
            rng.shuffle(all_ep)
            n_val = max(1, round(len(all_ep) * val_fraction))
            train_eps[subpath] = sorted(all_ep[:-n_val])
            val_eps[subpath] = sorted(all_ep[-n_val:])
        return train_eps, val_eps

    def prefetch_metadata(self, episodes: dict[str, list[int]] | None = None, **dataset_kwargs) -> dict[int, MultiLeRobotDataset]:
        """Build one MultiLeRobotDataset per camera count. Calls load_metadata() if not already done."""
        if not hasattr(self, "metas") or not self.metas:
            self.load_metadata()

        root = self._local_root()
        for n_cams, subpaths in self.buckets.items():
            kw = dict(dataset_kwargs)
            if "delta_timestamps" in kw and kw["delta_timestamps"]:
                kw["delta_timestamps"] = self._expand_camera_timestamps(kw["delta_timestamps"], n_cams)
            if episodes is not None:
                ep_dict = {sp: episodes[sp] for sp in subpaths if sp in episodes and episodes[sp]}
                if not ep_dict:
                    continue
                kw["episodes"] = ep_dict
                subpaths = list(ep_dict.keys())
            print(f"Building MultiLeRobotDataset for {n_cams} camera(s) with {len(subpaths)} subdatasets")
            self.datasets[n_cams] = MultiLeRobotDataset(
                repo_ids=subpaths,
                root=root,
                **kw,
            )

        return self.datasets

    def build_val_dataset(
        self,
        val_episodes: dict[str, list[int]],
        target_num_cameras: int = 2,
        **dataset_kwargs,
    ) -> MultiLeRobotDataset | None:
        """Build a single MultiLeRobotDataset for validation from one camera-count bucket."""
        if target_num_cameras not in self.buckets:
            print(f"No {target_num_cameras}-camera sub-datasets found, skipping val dataset")
            return None

        subpaths = self.buckets[target_num_cameras]
        ep_dict = {sp: val_episodes[sp] for sp in subpaths if sp in val_episodes and val_episodes[sp]}
        if not ep_dict:
            print(f"No val episodes for {target_num_cameras}-camera sub-datasets")
            return None

        kw = dict(dataset_kwargs)
        if "delta_timestamps" in kw and kw["delta_timestamps"]:
            kw["delta_timestamps"] = self._expand_camera_timestamps(kw["delta_timestamps"], target_num_cameras)
        kw["episodes"] = ep_dict

        val_subpaths = list(ep_dict.keys())
        n_eps = sum(len(v) for v in ep_dict.values())
        print(f"Building val MultiLeRobotDataset: {len(val_subpaths)} subdatasets, {n_eps} episodes ({target_num_cameras} cameras)")
        return MultiLeRobotDataset(
            repo_ids=val_subpaths,
            root=self._local_root(),
            **kw,
        )
