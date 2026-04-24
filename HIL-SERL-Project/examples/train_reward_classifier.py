# python train_reward_classifier.py \
#   --exp_name=galaxea_usb_insertion_single \
#   --classifier_keys=left_wrist_rgb

# 或者三路：

# python train_reward_classifier.py \
#   --exp_name=galaxea_usb_insertion_single \
#   --classifier_keys=head_rgb,left_wrist_rgb,right_wrist_rgb

import os
import sys
import glob
import json
import pickle as pkl
import functools
from typing import Dict, List, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERL_LAUNCHER_ROOT = os.path.join(PROJECT_ROOT, "serl_launcher")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SERL_LAUNCHER_ROOT not in sys.path:
    sys.path.insert(0, SERL_LAUNCHER_ROOT)

import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
from flax import jax_utils
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags
from gymnasium import spaces

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from examples.galaxea_task.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name",
    None,
    "Name of experiment corresponding to folder.",
)

flags.DEFINE_integer(
    "num_epochs",
    150,
    "Number of training epochs.",
)

flags.DEFINE_integer(
    "batch_size",
    256,
    "Global batch size across all GPUs.",
)

flags.DEFINE_string(
    "classifier_keys",
    None,
    (
        "Comma-separated image keys for reward classifier, e.g. "
        "'head_rgb,right_wrist_rgb'. "
        "If None, use config.classifier_keys."
    ),
)

flags.DEFINE_string(
    "data_dir",
    "classifier_data_single",
    "Directory containing success/failure classifier pkl files.",
)

flags.DEFINE_string(
    "ckpt_dir",
    "classifier_ckpt_single",
    "Directory to save classifier checkpoint.",
)

flags.DEFINE_boolean(
    "filter_by_exp_name",
    True,
    "Only load classifier data files whose filename starts with exp_name.",
)


def parse_classifier_keys(config) -> List[str]:
    if FLAGS.classifier_keys is None or FLAGS.classifier_keys.strip() == "":
        keys = list(config.classifier_keys)
    else:
        raw = FLAGS.classifier_keys.replace(" ", "")
        keys = [k for k in raw.split(",") if k]

    if len(keys) == 0:
        raise ValueError("classifier_keys 不能为空")

    return keys


def infer_space_from_value(x):
    """递归从样本值推断 gymnasium space。"""
    if isinstance(x, dict):
        return spaces.Dict({k: infer_space_from_value(v) for k, v in x.items()})

    arr = np.asarray(x)

    if arr.dtype == np.uint8:
        return spaces.Box(low=0, high=255, shape=arr.shape, dtype=np.uint8)
    elif np.issubdtype(arr.dtype, np.bool_):
        return spaces.Box(low=0, high=1, shape=arr.shape, dtype=np.bool_)
    elif np.issubdtype(arr.dtype, np.integer):
        return spaces.Box(
            low=np.iinfo(arr.dtype).min,
            high=np.iinfo(arr.dtype).max,
            shape=arr.shape,
            dtype=arr.dtype,
        )
    else:
        return spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=arr.dtype)


def get_obs_image(obs: Dict[str, Any], key: str):
    """
    支持两种格式：
    1. obs[key]
    2. obs["images"][key]
    """
    if key in obs:
        return obs[key]

    if "images" in obs and isinstance(obs["images"], dict) and key in obs["images"]:
        return obs["images"][key]

    raise KeyError(
        f"observation 中找不到 classifier key='{key}'。"
        f"obs keys={list(obs.keys())}"
    )


def prune_observation_for_classifier(obs: Dict[str, Any], classifier_keys: List[str]):
    """
    只保留 classifier 需要的相机 key。
    这样可以做到：
    - 原始数据录了三路
    - classifier 训练只用其中一路/两路/三路
    """
    if not isinstance(obs, dict):
        raise TypeError(f"observation 应该是 dict，但得到 {type(obs)}")

    new_obs = {}
    for key in classifier_keys:
        new_obs[key] = get_obs_image(obs, key)

    return new_obs


def prune_transition_for_classifier(trans: Dict[str, Any], classifier_keys: List[str]):
    trans = dict(trans)

    if "observations" not in trans:
        raise KeyError(f"transition 缺少 observations，keys={list(trans.keys())}")

    trans["observations"] = prune_observation_for_classifier(
        trans["observations"],
        classifier_keys,
    )

    if "next_observations" in trans and isinstance(trans["next_observations"], dict):
        try:
            trans["next_observations"] = prune_observation_for_classifier(
                trans["next_observations"],
                classifier_keys,
            )
        except Exception:
            # classifier 训练只需要 observations，next_observations 不强制要求
            trans.pop("next_observations", None)

    return trans


def find_classifier_data_files():
    data_root = os.path.join(os.getcwd(), FLAGS.data_dir)

    if FLAGS.filter_by_exp_name:
        success_paths = glob.glob(
            os.path.join(data_root, f"{FLAGS.exp_name}*success*.pkl")
        )
        failure_paths = glob.glob(
            os.path.join(data_root, f"{FLAGS.exp_name}*failure*.pkl")
        )
    else:
        success_paths = glob.glob(os.path.join(data_root, "*success*.pkl"))
        failure_paths = glob.glob(os.path.join(data_root, "*failure*.pkl"))

    success_paths = sorted(success_paths)
    failure_paths = sorted(failure_paths)

    if len(success_paths) == 0:
        raise FileNotFoundError(
            f"没有找到 success pkl。data_root={data_root}, "
            f"filter_by_exp_name={FLAGS.filter_by_exp_name}, exp_name={FLAGS.exp_name}"
        )

    if len(failure_paths) == 0:
        raise FileNotFoundError(
            f"没有找到 failure pkl。data_root={data_root}, "
            f"filter_by_exp_name={FLAGS.filter_by_exp_name}, exp_name={FLAGS.exp_name}"
        )

    return success_paths, failure_paths


def load_pkl_list(path: str):
    with open(path, "rb") as f:
        data = pkl.load(f)

    if not isinstance(data, list):
        raise TypeError(f"{path} 不是 list 格式，实际类型={type(data)}")

    return data


def find_first_valid_transition(paths: List[str], classifier_keys: List[str]):
    for path in paths:
        data = load_pkl_list(path)
        for trans in data:
            if not isinstance(trans, dict):
                continue
            if "observations" not in trans or "actions" not in trans:
                continue

            try:
                pruned = prune_transition_for_classifier(trans, classifier_keys)
                return pruned
            except Exception:
                continue

    raise ValueError(
        f"无法从数据里找到包含 classifier_keys={classifier_keys} 的 transition"
    )


def build_spaces_from_dataset(
    success_paths: List[str],
    failure_paths: List[str],
    classifier_keys: List[str],
):
    all_paths = list(success_paths) + list(failure_paths)

    sample_transition = find_first_valid_transition(all_paths, classifier_keys)

    observation_space = infer_space_from_value(sample_transition["observations"])
    action_space = infer_space_from_value(sample_transition["actions"])

    return observation_space, action_space, sample_transition


def shard_batch(batch, num_devices: int):
    """把全局 batch reshape 成 [n_devices, per_device_batch, ...]。"""
    def _shard(x):
        x = np.asarray(x)
        if x.shape[0] % num_devices != 0:
            raise ValueError(
                f"Batch 第一维 {x.shape[0]} 不能被设备数 {num_devices} 整除"
            )
        return x.reshape((num_devices, x.shape[0] // num_devices) + x.shape[1:])

    return jax.tree_util.tree_map(_shard, batch)


def tree_to_host_numpy(tree):
    """把 pytree 里的 jax.Array 全部转成 host numpy，便于保存 checkpoint。"""
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x))
        if isinstance(x, (jax.Array, jnp.ndarray))
        else x,
        tree,
    )


def insert_classifier_data(
    buffer: ReplayBuffer,
    paths: List[str],
    label: int,
    classifier_keys: List[str],
):
    inserted = 0
    skipped = 0

    for path in paths:
        data = load_pkl_list(path)

        for trans in data:
            try:
                trans = prune_transition_for_classifier(trans, classifier_keys)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"⚠️ 跳过 transition: path={path}, err={repr(e)}")
                continue

            trans = dict(trans)
            trans["labels"] = label

            if "actions" not in trans:
                trans["actions"] = np.zeros((1,), dtype=np.float32)
            else:
                trans["actions"] = np.zeros_like(
                    np.asarray(trans["actions"]),
                    dtype=np.float32,
                )

            buffer.insert(trans)
            inserted += 1

    return inserted, skipped


def batched_random_crop_auto(img, crop_key, padding=4):
    """
    兼容两种图像 batch：
    1. [B, H, W, C]       -> num_batch_dims=1
    2. [B, 1, H, W, C]    -> num_batch_dims=2

    pmap 后每张卡上的 img 通常是：
    - sample 图像 shape=(1,H,W,C) 时：img=[local_B,1,H,W,C]
    - sample 图像 shape=(H,W,C) 时：img=[local_B,H,W,C]
    """
    ndim = len(img.shape)

    if ndim == 5:
        num_batch_dims = 2
    elif ndim == 4:
        num_batch_dims = 1
    else:
        raise ValueError(
            f"图像维度异常，key 对应 img.shape={img.shape}，"
            f"期望 [B,H,W,C] 或 [B,1,H,W,C]"
        )

    return batched_random_crop(
        img,
        crop_key,
        padding=padding,
        num_batch_dims=num_batch_dims,
    )


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    classifier_keys = parse_classifier_keys(config)

    num_devices = jax.local_device_count()
    devices = jax.local_devices()

    if FLAGS.batch_size % num_devices != 0:
        raise ValueError(
            f"FLAGS.batch_size={FLAGS.batch_size} 必须能被设备数 num_devices={num_devices} 整除"
        )

    print("=" * 100)
    print("Reward Classifier Training")
    print("=" * 100)
    print(f"exp_name             : {FLAGS.exp_name}")
    print(f"data_dir             : {os.path.abspath(FLAGS.data_dir)}")
    print(f"ckpt_dir             : {os.path.abspath(FLAGS.ckpt_dir)}")
    print(f"classifier_keys      : {classifier_keys}")
    print(f"config.image_keys    : {getattr(config, 'image_keys', None)}")
    print(f"config.classifier_keys(default): {getattr(config, 'classifier_keys', None)}")
    print(f"Using {num_devices} local devices: {devices}")
    print(f"Global batch size    : {FLAGS.batch_size}")
    print(f"Per-device batch size: {FLAGS.batch_size // num_devices}")
    print("=" * 100)

    success_paths, failure_paths = find_classifier_data_files()

    print("\nSuccess files:")
    for p in success_paths:
        print(f"  {p}")

    print("\nFailure files:")
    for p in failure_paths:
        print(f"  {p}")

    observation_space, action_space, sample_transition = build_spaces_from_dataset(
        success_paths,
        failure_paths,
        classifier_keys,
    )

    print("\nObservation space used by classifier:")
    print(observation_space)
    print("\nAction space:")
    print(action_space)

    # -----------------------------
    # Positive buffer
    # -----------------------------
    pos_buffer = ReplayBuffer(
        observation_space,
        action_space,
        capacity=20000,
        include_label=True,
    )

    pos_inserted, pos_skipped = insert_classifier_data(
        pos_buffer,
        success_paths,
        label=1,
        classifier_keys=classifier_keys,
    )

    pos_iterator = pos_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2},
    )

    # -----------------------------
    # Negative buffer
    # -----------------------------
    neg_buffer = ReplayBuffer(
        observation_space,
        action_space,
        capacity=50000,
        include_label=True,
    )

    neg_inserted, neg_skipped = insert_classifier_data(
        neg_buffer,
        failure_paths,
        label=0,
        classifier_keys=classifier_keys,
    )

    neg_iterator = neg_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2},
    )

    print("\nDataset summary:")
    print(f"success inserted: {pos_inserted}, skipped: {pos_skipped}")
    print(f"failure inserted: {neg_inserted}, skipped: {neg_skipped}")
    print(f"success buffer size: {len(pos_buffer)}")
    print(f"failed buffer size : {len(neg_buffer)}")

    if len(pos_buffer) == 0:
        raise ValueError("success buffer 为空，无法训练 classifier")
    if len(neg_buffer) == 0:
        raise ValueError("failure buffer 为空，无法训练 classifier")

    rng = jax.random.PRNGKey(0)
    rng, init_key = jax.random.split(rng)

    # 用一组未分片的 sample 初始化 classifier
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    print("\nClassifier sample observation keys:")
    print(list(sample["observations"].keys()))
    for k in classifier_keys:
        print(f"  {k}: shape={np.asarray(sample['observations'][k]).shape}")

    classifier = create_classifier(
        init_key,
        sample["observations"],
        classifier_keys,
    )

    # 复制到多 GPU
    classifier = jax_utils.replicate(classifier)

    def data_augmentation_fn(rng, observations):
        for pixel_key in classifier_keys:
            rng, crop_key = jax.random.split(rng)

            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop_auto(
                        observations[pixel_key],
                        crop_key,
                        padding=4,
                    )
                }
            )

        return observations

    @functools.partial(jax.pmap, axis_name="devices")
    def train_step(state, batch, aug_key, dropout_key):
        obs = data_augmentation_fn(aug_key, batch["observations"])
        batch = batch.copy(add_or_replace={"observations": obs})

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                batch["observations"],
                rngs={"dropout": dropout_key},
                train=True,
            )

            loss = optax.sigmoid_binary_cross_entropy(
                logits,
                batch["labels"],
            ).mean()

            return loss, logits

        (loss, logits), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)

        grads = lax.pmean(grads, axis_name="devices")
        loss = lax.pmean(loss, axis_name="devices")

        preds = nn.sigmoid(logits) >= 0.5
        acc = jnp.mean(preds == batch["labels"])
        acc = lax.pmean(acc, axis_name="devices")

        new_state = state.apply_gradients(grads=grads)

        return new_state, loss, acc

    for epoch in tqdm(range(FLAGS.num_epochs)):
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        batch = concat_batches(pos_sample, neg_sample, axis=0)

        labels = np.asarray(batch["labels"], dtype=np.float32)[..., None]
        batch = batch.copy(add_or_replace={"labels": labels})

        batch = shard_batch(batch, num_devices)

        rng, aug_master_key = jax.random.split(rng)
        rng, dropout_master_key = jax.random.split(rng)

        aug_keys = jax.random.split(aug_master_key, num_devices)
        dropout_keys = jax.random.split(dropout_master_key, num_devices)

        classifier, train_loss, train_accuracy = train_step(
            classifier,
            batch,
            aug_keys,
            dropout_keys,
        )

        train_loss_scalar = float(jax.device_get(train_loss[0]))
        train_acc_scalar = float(jax.device_get(train_accuracy[0]))

        print(
            f"Epoch: {epoch + 1}, "
            f"Train Loss: {train_loss_scalar:.4f}, "
            f"Train Accuracy: {train_acc_scalar:.4f}"
        )

    # 保存前 unreplicate，再转 host numpy
    classifier_to_save = jax_utils.unreplicate(classifier)
    classifier_to_save = tree_to_host_numpy(classifier_to_save)

    save_dir = os.path.join(os.getcwd(), FLAGS.ckpt_dir)
    os.makedirs(save_dir, exist_ok=True)

    checkpoints.save_checkpoint(
        save_dir,
        classifier_to_save,
        step=FLAGS.num_epochs,
        overwrite=True,
    )

    metadata = {
        "exp_name": FLAGS.exp_name,
        "classifier_keys": classifier_keys,
        "num_epochs": FLAGS.num_epochs,
        "batch_size": FLAGS.batch_size,
        "success_files": success_paths,
        "failure_files": failure_paths,
        "success_inserted": pos_inserted,
        "failure_inserted": neg_inserted,
        "success_skipped": pos_skipped,
        "failure_skipped": neg_skipped,
    }

    metadata_path = os.path.join(save_dir, "classifier_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 100)
    print("✅ classifier checkpoint 已保存")
    print("=" * 100)
    print(f"checkpoint dir : {save_dir}")
    print(f"metadata path  : {metadata_path}")
    print(f"classifier_keys: {classifier_keys}")
    print("\nRLPD 中必须使用同一组 classifier_keys 加载这个 ckpt。")
    print("但 RLPD policy 的 image_keys 可以和 classifier_keys 不一样。")
    print("=" * 100)


if __name__ == "__main__":
    app.run(main)

# import os
# import sys
# import glob
# import pickle as pkl
# import functools

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# SERL_LAUNCHER_ROOT = os.path.join(PROJECT_ROOT, "serl_launcher")

# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
# if SERL_LAUNCHER_ROOT not in sys.path:
#     sys.path.insert(0, SERL_LAUNCHER_ROOT)

# import jax
# import jax.numpy as jnp
# import jax.lax as lax
# import flax.linen as nn
# from flax import jax_utils
# from flax.training import checkpoints
# import numpy as np
# import optax
# from tqdm import tqdm
# from absl import app, flags
# from gymnasium import spaces

# from serl_launcher.data.data_store import ReplayBuffer
# from serl_launcher.utils.train_utils import concat_batches
# from serl_launcher.vision.data_augmentations import batched_random_crop
# from serl_launcher.networks.reward_classifier import create_classifier

# from examples.galaxea_task.mappings import CONFIG_MAPPING


# FLAGS = flags.FLAGS
# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
# flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
# flags.DEFINE_integer("batch_size", 256, "Global batch size across all GPUs.")


# def infer_space_from_value(x):
#     """递归从样本值推断 gymnasium space。"""
#     if isinstance(x, dict):
#         return spaces.Dict({k: infer_space_from_value(v) for k, v in x.items()})

#     arr = np.asarray(x)

#     if arr.dtype == np.uint8:
#         return spaces.Box(low=0, high=255, shape=arr.shape, dtype=np.uint8)
#     elif np.issubdtype(arr.dtype, np.bool_):
#         return spaces.Box(low=0, high=1, shape=arr.shape, dtype=np.bool_)
#     elif np.issubdtype(arr.dtype, np.integer):
#         return spaces.Box(
#             low=np.iinfo(arr.dtype).min,
#             high=np.iinfo(arr.dtype).max,
#             shape=arr.shape,
#             dtype=arr.dtype,
#         )
#     else:
#         return spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=arr.dtype)


# def build_spaces_from_dataset(success_paths, failure_paths):
#     """从 success / failure pkl 中找一条样本，推断 observation_space 和 action_space。"""
#     all_paths = list(success_paths) + list(failure_paths)
#     if not all_paths:
#         raise ValueError("classifier_data_single 目录下没有找到 success/failure pkl 文件")

#     sample_transition = None
#     for path in all_paths:
#         with open(path, "rb") as f:
#             data = pkl.load(f)
#         if len(data) > 0:
#             sample_transition = data[0]
#             break

#     if sample_transition is None:
#         raise ValueError("所有 pkl 文件都是空的，无法推断 observation_space/action_space")

#     observation_space = infer_space_from_value(sample_transition["observations"])
#     action_space = infer_space_from_value(sample_transition["actions"])
#     return observation_space, action_space


# def shard_batch(batch, num_devices: int):
#     """把全局 batch reshape 成 [n_devices, per_device_batch, ...]。"""
#     def _shard(x):
#         x = np.asarray(x)
#         if x.shape[0] % num_devices != 0:
#             raise ValueError(
#                 f"Batch 第一维 {x.shape[0]} 不能被设备数 {num_devices} 整除"
#             )
#         return x.reshape((num_devices, x.shape[0] // num_devices) + x.shape[1:])

#     return jax.tree_util.tree_map(_shard, batch)


# def tree_to_host_numpy(tree):
#     """把 pytree 里的 jax.Array 全部转成 host numpy，便于 checkpoints.save_checkpoint。"""
#     return jax.tree_util.tree_map(
#         lambda x: np.asarray(jax.device_get(x)) if isinstance(x, (jax.Array, jnp.ndarray)) else x,
#         tree,
#     )


# def main(_):
#     assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
#     config = CONFIG_MAPPING[FLAGS.exp_name]()

#     num_devices = jax.local_device_count()
#     devices = jax.local_devices()

#     if FLAGS.batch_size % num_devices != 0:
#         raise ValueError(
#             f"FLAGS.batch_size={FLAGS.batch_size} 必须能被设备数 num_devices={num_devices} 整除"
#         )

#     print(f"Using {num_devices} local devices: {devices}")
#     print(f"Global batch size: {FLAGS.batch_size}")
#     print(f"Per-device batch size: {FLAGS.batch_size // num_devices}")

#     success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_single", "*success*.pkl"))
#     failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_single", "*failure*.pkl"))

#     observation_space, action_space = build_spaces_from_dataset(success_paths, failure_paths)

#     # -----------------------------
#     # Positive buffer
#     # -----------------------------
#     pos_buffer = ReplayBuffer(
#         observation_space,
#         action_space,
#         capacity=20000,
#         include_label=True,
#     )

#     for path in success_paths:
#         with open(path, "rb") as f:
#             success_data = pkl.load(f)

#         for trans in success_data:
#             if "images" in trans["observations"].keys():
#                 continue

#             trans = dict(trans)
#             trans["labels"] = 1
#             trans["actions"] = np.zeros_like(np.asarray(trans["actions"]), dtype=np.float32)
#             pos_buffer.insert(trans)

#     pos_iterator = pos_buffer.get_iterator(
#         sample_args={"batch_size": FLAGS.batch_size // 2},
#     )

#     # -----------------------------
#     # Negative buffer
#     # -----------------------------
#     neg_buffer = ReplayBuffer(
#         observation_space,
#         action_space,
#         capacity=50000,
#         include_label=True,
#     )

#     for path in failure_paths:
#         with open(path, "rb") as f:
#             failure_data = pkl.load(f)

#         for trans in failure_data:
#             if "images" in trans["observations"].keys():
#                 continue

#             trans = dict(trans)
#             trans["labels"] = 0
#             trans["actions"] = np.zeros_like(np.asarray(trans["actions"]), dtype=np.float32)
#             neg_buffer.insert(trans)

#     neg_iterator = neg_buffer.get_iterator(
#         sample_args={"batch_size": FLAGS.batch_size // 2},
#     )

#     print(f"failed buffer size: {len(neg_buffer)}")
#     print(f"success buffer size: {len(pos_buffer)}")

#     rng = jax.random.PRNGKey(0)
#     rng, init_key = jax.random.split(rng)

#     # 用一组未分片的 sample 初始化 classifier
#     pos_sample = next(pos_iterator)
#     neg_sample = next(neg_iterator)
#     sample = concat_batches(pos_sample, neg_sample, axis=0)

#     classifier = create_classifier(
#         init_key,
#         sample["observations"],
#         config.classifier_keys,
#     )

#     # 复制到多 GPU
#     classifier = jax_utils.replicate(classifier)

#     def data_augmentation_fn(rng, observations):
#         """
#         虽然已经 pmap 到单个 device，
#         但图像张量仍然是 [local_B, 1, H, W, C]，
#         所以这里仍然需要 num_batch_dims=2。
#         """
#         for pixel_key in config.classifier_keys:
#             rng, crop_key = jax.random.split(rng)
#             observations = observations.copy(
#                 add_or_replace={
#                     pixel_key: batched_random_crop(
#                         observations[pixel_key],
#                         crop_key,
#                         padding=4,
#                         num_batch_dims=2,
#                     )
#                 }
#             )
#         return observations

#     @functools.partial(jax.pmap, axis_name="devices")
#     def train_step(state, batch, aug_key, dropout_key):
#         obs = data_augmentation_fn(aug_key, batch["observations"])
#         batch = batch.copy(add_or_replace={"observations": obs})

#         def loss_fn(params):
#             logits = state.apply_fn(
#                 {"params": params},
#                 batch["observations"],
#                 rngs={"dropout": dropout_key},
#                 train=True,
#             )
#             loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
#             return loss, logits

#         (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

#         grads = lax.pmean(grads, axis_name="devices")
#         loss = lax.pmean(loss, axis_name="devices")

#         preds = (nn.sigmoid(logits) >= 0.5)
#         acc = jnp.mean(preds == batch["labels"])
#         acc = lax.pmean(acc, axis_name="devices")

#         new_state = state.apply_gradients(grads=grads)
#         return new_state, loss, acc

#     for epoch in tqdm(range(FLAGS.num_epochs)):
#         pos_sample = next(pos_iterator)
#         neg_sample = next(neg_iterator)
#         batch = concat_batches(pos_sample, neg_sample, axis=0)

#         # labels -> float32, shape [B, 1]
#         labels = np.asarray(batch["labels"], dtype=np.float32)[..., None]
#         batch = batch.copy(add_or_replace={"labels": labels})

#         # 分片到多 GPU
#         batch = shard_batch(batch, num_devices)

#         rng, aug_master_key = jax.random.split(rng)
#         rng, dropout_master_key = jax.random.split(rng)

#         aug_keys = jax.random.split(aug_master_key, num_devices)
#         dropout_keys = jax.random.split(dropout_master_key, num_devices)

#         classifier, train_loss, train_accuracy = train_step(
#             classifier,
#             batch,
#             aug_keys,
#             dropout_keys,
#         )

#         # pmap 返回每张卡一份相同标量，取第 0 张即可
#         train_loss_scalar = float(jax.device_get(train_loss[0]))
#         train_acc_scalar = float(jax.device_get(train_accuracy[0]))

#         print(
#             f"Epoch: {epoch + 1}, "
#             f"Train Loss: {train_loss_scalar:.4f}, "
#             f"Train Accuracy: {train_acc_scalar:.4f}"
#         )

#     # 关键：保存前先 unreplicate，再转成 host numpy
#     classifier_to_save = jax_utils.unreplicate(classifier)
#     classifier_to_save = tree_to_host_numpy(classifier_to_save)

#     save_dir = os.path.join(os.getcwd(), "classifier_ckpt_single")
#     os.makedirs(save_dir, exist_ok=True)

#     checkpoints.save_checkpoint(
#         save_dir,
#         classifier_to_save,
#         step=FLAGS.num_epochs,
#         overwrite=True,
#     )

#     print(f"✅ classifier checkpoint 已保存到: {save_dir}")


# if __name__ == "__main__":
#     app.run(main)


