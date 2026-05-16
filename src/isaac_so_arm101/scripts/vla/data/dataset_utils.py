from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import math
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - allow import in workspaces without torch
    class Dataset:  # type: ignore
        pass


IGNORE_INDEX = -100


@dataclass
class JsonlExample:
    image: str
    instruction: str
    action: List[float]


def _lower(s: str) -> str:
    return s.strip().lower()


def build_openvla_prompt(instruction: str, *, vla_path: Union[str, Path]) -> str:
    """Builds a text prompt consistent with OpenVLA's public deploy examples."""
    instruction = _lower(instruction)
    vla_path = str(vla_path)

    if "v01" in vla_path:
        system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        return (
            f"{system_prompt} USER: What action should the robot take to {instruction}? "
            "ASSISTANT:"
        )

    return f"In: What action should the robot take to {instruction}?\nOut:"


class DiscreteActionTokenizer:
    """Discretizes continuous actions into OpenVLA-style action tokens.

    OpenVLA maps each action dimension to one token ID in the *tail* of the
    tokenizer vocabulary, using uniform binning into `bins` buckets.

    This is designed to match OpenVLA's default `ActionTokenizer` behavior.
    """

    def __init__(
        self,
        tokenizer,
        *,
        bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ) -> None:
        self._tokenizer = tokenizer
        self._bins = int(bins)
        self._min_action = float(min_action)
        self._max_action = float(max_action)
        self._edges = np.linspace(self._min_action, self._max_action, self._bins)

        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            raise ValueError("Tokenizer must expose a positive `vocab_size`.")
        self.action_token_begin_idx = int(vocab_size - (self._bins + 1))
        self._vocab_size = vocab_size

    def encode_to_token_ids(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float32), self._min_action, self._max_action)
        bucket_idx = np.digitize(action, self._edges)
        bucket_idx = np.clip(bucket_idx, 0, self._bins - 1)
        return (self.action_token_begin_idx + bucket_idx).astype(np.int64)

    def __call__(self, action: np.ndarray) -> str:
        token_ids = self.encode_to_token_ids(action)
        return self._tokenizer.decode(token_ids.tolist())


class JsonlVlaDataset(Dataset):
    def __init__(
        self,
        *,
        jsonl_path: Path,
        image_root: Optional[Path],
        tokenizer,
        image_transform,
        vla_path: Union[str, Path],
        action_tokenizer: DiscreteActionTokenizer,
        action_dim: int = 7,
        predict_stop_token: bool = True,
    ) -> None:
        self._jsonl_path = Path(jsonl_path)
        self._tokenizer = tokenizer
        self._image_transform = image_transform
        self._vla_path = str(vla_path)
        self._action_tokenizer = action_tokenizer
        self._action_dim = int(action_dim)
        self._predict_stop_token = bool(predict_stop_token)

        if image_root is None:
            self._image_root = self._jsonl_path.parent
        else:
            self._image_root = Path(image_root)

        self._examples: List[JsonlExample] = []
        with self._jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {self._jsonl_path}") from exc

                missing = [k for k in ("image", "instruction", "action") if k not in raw]
                if missing:
                    raise ValueError(f"Missing keys {missing} on line {line_no} of {self._jsonl_path}")

                ex = JsonlExample(
                    image=str(raw["image"]),
                    instruction=str(raw["instruction"]),
                    action=list(raw["action"]),
                )
                self._examples.append(ex)

        if not self._examples:
            raise ValueError(f"No usable examples found in {self._jsonl_path}")

        if self._tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define `eos_token_id`.")

        if self._tokenizer.pad_token_id is None:
            # OpenVLA tokenizers typically define this, but make a safe fallback.
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def __len__(self) -> int:  # noqa: D401
        return len(self._examples)

    def _resolve_image_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self._image_root / p

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self._examples[idx]
        image_path = self._resolve_image_path(ex.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load image referenced by JSONL dataset. "
                f"image={image_path} jsonl={self._jsonl_path} idx={idx}"
            ) from exc

        action = np.asarray(ex.action, dtype=np.float32)
        if action.shape != (self._action_dim,):
            raise ValueError(
                f"Expected action shape ({self._action_dim},) but got {action.shape} for {image_path}"
            )

        if np.any(action < -1.001) or np.any(action > 1.001):
            raise ValueError(
                "Actions must be normalized to [-1, 1] per-dimension. "
                f"Found out-of-range values for {image_path}."
            )

        action_text = self._action_tokenizer(action)
        prompt = build_openvla_prompt(ex.instruction, vla_path=self._vla_path)

        # Tokenize prompt + action tokens
        base_text = f"{prompt}{action_text}"
        tokenized = self._tokenizer(base_text, add_special_tokens=True, return_attention_mask=False)
        input_ids: List[int] = list(tokenized["input_ids"])

        # Ensure exactly one EOS at the end.
        if input_ids[-1] != self._tokenizer.eos_token_id:
            input_ids.append(self._tokenizer.eos_token_id)

        labels = list(input_ids)

        # Keep loss on: action tokens + EOS (stop) token.
        keep = self._action_dim + 1
        if len(labels) < keep:
            raise ValueError(
                "Tokenization produced a sequence shorter than expected. "
                "Check that action tokenization yields one token per action dimension."
            )

        for i in range(0, len(labels) - keep):
            labels[i] = IGNORE_INDEX

        if not self._predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # Pixel values
        pixel_values = self._image_transform(image)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }


class LeRobotVlaDataset(Dataset):
    """Loads one or more LeRobot v3 datasets and presents them as a combined VLA training set.

    Actions are normalized per-channel to [-1, 1] using the global min/max across all
    provided datasets. Norm stats are exposed via `get_norm_stats()` for saving.
    """

    def __init__(
        self,
        *,
        repo_ids: List[str],
        tokenizer,
        image_transform,
        vla_path: Union[str, Path],
        action_tokenizer: DiscreteActionTokenizer,
        action_dim: int = 7,
        predict_stop_token: bool = True,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "lerobot and pandas are required for --lerobot_repo_ids. "
                "Install with: pip install lerobot pandas"
            ) from exc

        self._tokenizer = tokenizer
        self._image_transform = image_transform
        self._vla_path = str(vla_path)
        self._action_tokenizer = action_tokenizer
        self._action_dim = int(action_dim)
        self._predict_stop_token = bool(predict_stop_token)

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._lerobot_datasets: List[Any] = []
        self._index: List[Tuple[int, int]] = []  # (dataset_idx, frame_idx)
        all_actions: List[np.ndarray] = []

        for ds_idx, repo_id in enumerate(repo_ids):
            print(f"[INFO] Loading LeRobot dataset: {repo_id}")
            ds = _LeRobotDataset(repo_id, video_backend="pyav")
            self._lerobot_datasets.append(ds)

            # Scan actions from Parquet (no video decoding needed for stats)
            data_dir = Path(ds.root) / "data"
            for pf in sorted(data_dir.glob("**/*.parquet")):
                df = pd.read_parquet(pf, columns=["action"])
                all_actions.extend(df["action"].tolist())

            for i in range(len(ds)):
                self._index.append((ds_idx, i))
            print(f"  -> {len(ds)} frames")

        if not self._index:
            raise ValueError("No frames found in the provided LeRobot datasets.")

        actions_arr = np.array(all_actions, dtype=np.float32)  # (N, action_dim)
        self.action_norm_min: np.ndarray = actions_arr.min(axis=0)
        self.action_norm_max: np.ndarray = actions_arr.max(axis=0)
        print(f"[INFO] Action norm min: {self.action_norm_min.tolist()}")
        print(f"[INFO] Action norm max: {self.action_norm_max.tolist()}")

    def get_norm_stats(self) -> Dict[str, Any]:
        return {
            "action_norm_min": self.action_norm_min.tolist(),
            "action_norm_max": self.action_norm_max.tolist(),
        }

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        rng = self.action_norm_max - self.action_norm_min
        rng = np.where(rng < 1e-8, 1.0, rng)
        return np.clip(2.0 * (action - self.action_norm_min) / rng - 1.0, -1.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, frame_idx = self._index[idx]
        frame = self._lerobot_datasets[ds_idx][frame_idx]

        # (C, H, W) uint8 tensor -> PIL RGB
        img_t = frame["observation.images.wrist_camera"]
        img_np = img_t.permute(1, 2, 0).numpy().astype(np.uint8)
        image = Image.fromarray(img_np)

        action = frame["action"].numpy().astype(np.float32)
        action_norm = self._normalize_action(action)

        if action_norm.shape != (self._action_dim,):
            raise ValueError(
                f"Expected action shape ({self._action_dim},) but got {action_norm.shape}"
            )

        instruction = str(frame["task"])

        action_text = self._action_tokenizer(action_norm)
        prompt = build_openvla_prompt(instruction, vla_path=self._vla_path)
        base_text = f"{prompt}{action_text}"
        tokenized = self._tokenizer(base_text, add_special_tokens=True, return_attention_mask=False)
        input_ids: List[int] = list(tokenized["input_ids"])

        if input_ids[-1] != self._tokenizer.eos_token_id:
            input_ids.append(self._tokenizer.eos_token_id)

        labels = list(input_ids)
        keep = self._action_dim + 1
        if len(labels) < keep:
            raise ValueError("Tokenization produced a sequence shorter than expected.")
        for i in range(len(labels) - keep):
            labels[i] = IGNORE_INDEX

        if not self._predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values = self._image_transform(image)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }


class PaddedCollatorForActionPrediction:
    def __init__(
        self,
        *,
        model_max_length: int,
        pad_token_id: int,
        padding_side: str = "right",
    ) -> None:
        if padding_side != "right":
            raise ValueError("Only right padding is supported.")
        self._model_max_length = int(model_max_length)
        self._pad_token_id = int(pad_token_id)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [ex["input_ids"] for ex in instances]
        labels = [ex["labels"] for ex in instances]
        pixel_values = [ex["pixel_values"] for ex in instances]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self._pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, : self._model_max_length]
        labels = labels[:, : self._model_max_length]

        attention_mask = input_ids.ne(self._pad_token_id)

        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values_out: Union[torch.Tensor, Dict[str, torch.Tensor]] = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values_out = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}
        else:
            raise TypeError(f"Unsupported pixel_values type: {type(pixel_values[0])}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values_out,
        }
