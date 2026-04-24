#按照三路相机拍摄即可，反正训练的时候可以选择只训练哪些相机


import os
import sys
import time
import copy
import pickle as pkl
import datetime
import threading
from collections import deque

from tqdm import tqdm
import numpy as np
from absl import app, flags
from pynput import keyboard

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from examples.galaxea_task.usb_pick_insertion_single.config import env_config


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name",
    "galaxea_usb_insertion_single",
    "Name of experiment corresponding to folder.",
)

flags.DEFINE_integer(
    "successes_needed",
    200,
    "Number of successful transitions to collect.",
)

flags.DEFINE_integer(
    "pre_success_frames",
    3,
    "Number of frames before SPACE to also label as success.",
)

flags.DEFINE_integer(
    "post_success_frames",
    3,
    "Number of frames after SPACE to also label as success.",
)

flags.DEFINE_float(
    "failure_hz",
    10.0,
    "Sampling frequency for negative samples.",
)

flags.DEFINE_string(
    "classifier_image_keys",
    "__config__",
    (
        "Comma-separated image keys to save for reward classifier data. "
        "'__config__' means use env_config.classifier_keys. "
        "'all' means keep all observation image keys."
    ),
)

flags.DEFINE_boolean(
    "classifier_strict_obs_keys",
    True,
    "If True, raise error when requested classifier image key is missing.",
)


# ==========================================================
# 键盘状态
# ==========================================================
success_presses = 0
unlock_failure_presses = 0
key_lock = threading.Lock()

space_is_down = False
a_is_down = False


def on_press(key):
    global success_presses, unlock_failure_presses
    global space_is_down, a_is_down

    try:
        with key_lock:
            if key == keyboard.Key.space:
                if not space_is_down:
                    success_presses += 1
                    space_is_down = True
                    print(f"\n⌨️ 检测到 SPACE：待触发成功事件 +1（当前待消费: {success_presses}）")
                return

            if hasattr(key, "char") and key.char is not None:
                ch = key.char.lower()
                if ch == "a":
                    if not a_is_down:
                        unlock_failure_presses += 1
                        a_is_down = True
                        print(f"\n⌨️ 检测到 A：请求恢复失败采集（当前待消费: {unlock_failure_presses}）")
    except Exception as e:
        print(f"\n⚠️ 键盘监听异常(on_press): {e}")


def on_release(key):
    global space_is_down, a_is_down

    try:
        with key_lock:
            if key == keyboard.Key.space:
                space_is_down = False
                return

            if hasattr(key, "char") and key.char is not None:
                ch = key.char.lower()
                if ch == "a":
                    a_is_down = False
    except Exception as e:
        print(f"\n⚠️ 键盘监听异常(on_release): {e}")


# ==========================================================
# classifier 图像 key 解析 / obs 裁剪
# ==========================================================
def parse_key_list(raw_value, default_keys=None, allow_all=False):
    """
    解析逗号分隔 key 列表。

    - "__config__" -> default_keys
    - "all"        -> None，表示不裁剪
    - "none"       -> []
    """
    if raw_value is None:
        return list(default_keys or [])

    value = str(raw_value).strip()

    if value == "__config__":
        return list(default_keys or [])

    if allow_all and value.lower() == "all":
        return None

    if value.lower() in ("none", "null", ""):
        return []

    return [x.strip() for x in value.split(",") if x.strip()]


def get_classifier_storage_image_keys():
    return parse_key_list(
        FLAGS.classifier_image_keys,
        default_keys=getattr(env_config, "classifier_keys", []),
        allow_all=True,
    )


def _get_obs_value(obs, key):
    """
    支持两种 observation 结构：
    1. obs[key]
    2. obs["images"][key]
    """
    if key in obs:
        return obs[key]

    if "images" in obs and isinstance(obs["images"], dict) and key in obs["images"]:
        return obs["images"][key]

    raise KeyError(
        f"observation 中找不到 key='{key}'。当前 obs keys={list(obs.keys())}"
    )


def prune_obs_for_classifier_storage(obs, classifier_image_keys, strict=True):
    """
    只裁剪“保存到 reward classifier pkl 里的 obs”。

    classifier_image_keys=None 表示保留完整 obs。
    classifier_image_keys=[...] 表示只保留这些图像 key。
    """
    if obs is None:
        return obs

    if not isinstance(obs, dict):
        return obs

    if classifier_image_keys is None:
        return copy.deepcopy(obs)

    keep = {}

    for key in classifier_image_keys:
        try:
            keep[key] = _get_obs_value(obs, key)
        except KeyError:
            if strict:
                raise
            print(f"⚠️ classifier 数据保存时跳过缺失图像 key={key}")

    return keep


def prune_transition_for_classifier_storage(trans, classifier_image_keys, strict=True):
    """
    裁剪 transition 里的 observations / next_observations。
    """
    trans = copy.deepcopy(trans)

    if "observations" in trans:
        trans["observations"] = prune_obs_for_classifier_storage(
            trans["observations"],
            classifier_image_keys=classifier_image_keys,
            strict=strict,
        )

    if "next_observations" in trans:
        trans["next_observations"] = prune_obs_for_classifier_storage(
            trans["next_observations"],
            classifier_image_keys=classifier_image_keys,
            strict=strict,
        )

    return trans


def print_obs_keys_summary(transitions, name="classifier_data"):
    if not transitions:
        print(f"⚠️ {name}: empty")
        return

    obs = transitions[0].get("observations", {})
    next_obs = transitions[0].get("next_observations", {})

    print("\n===== reward classifier 数据 observation keys 检查 =====")
    print(f"{name} observations keys     : {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
    print(f"{name} next_observations keys: {list(next_obs.keys()) if isinstance(next_obs, dict) else type(next_obs)}")


# ==========================================================
# 工具函数
# ==========================================================
def make_transition(obs, actions, next_obs, rew, done, classifier_image_keys):
    """
    reward classifier 数据 transition。

    注意：
    - env.step() 内部可以仍然拿完整 obs。
    - 但保存到 pkl 之前，只保留 classifier_image_keys。
    """
    transition = dict(
        observations=prune_obs_for_classifier_storage(
            obs,
            classifier_image_keys=classifier_image_keys,
            strict=FLAGS.classifier_strict_obs_keys,
        ),
        actions=np.asarray(actions, dtype=np.float32),
        next_observations=prune_obs_for_classifier_storage(
            next_obs,
            classifier_image_keys=classifier_image_keys,
            strict=FLAGS.classifier_strict_obs_keys,
        ),
        rewards=rew,
        masks=1.0 - float(done),
        dones=bool(done),
    )

    transition = prune_transition_for_classifier_storage(
        transition,
        classifier_image_keys=classifier_image_keys,
        strict=FLAGS.classifier_strict_obs_keys,
    )

    return copy.deepcopy(transition)


def add_success_transition(transition, successes, pbar, success_needed):
    if len(successes) < success_needed:
        successes.append(copy.deepcopy(transition))
        pbar.update(1)
        return True
    return False


def flush_pending_failures(pending_failures, failures):
    while pending_failures:
        failures.append(pending_failures.popleft())


def sanitize_transition_list_for_classifier_storage(transitions, classifier_image_keys):
    return [
        prune_transition_for_classifier_storage(
            t,
            classifier_image_keys=classifier_image_keys,
            strict=FLAGS.classifier_strict_obs_keys,
        )
        for t in transitions
    ]


# ==========================================================
# 主函数
# ==========================================================
def main(_):
    global success_presses, unlock_failure_presses

    classifier_image_keys = get_classifier_storage_image_keys()

    print(f"🚀 开始采集视觉分类器数据：{FLAGS.exp_name}")
    print("💡 操作指南：")
    print("   1. 初次启动时会自动复位一次。")
    print("   2. 进入采集后，不再自动复位。")
    print("   3. 默认按固定频率持续收集失败样本。")
    print("   4. 当画面达到成功状态时，按一次【空格】。")
    print("      - 当前帧会记为成功")
    print("      - 空格前后若干帧也会一起记为成功")
    print("      - 同时锁定失败采集，避免成功后续画面被误标为失败")
    print("   5. 当你确认已经离开成功状态后，按一次【A】恢复失败采集。")
    print("   6. 长按 SPACE / A 都不会连发，必须松开后再次按下才会再触发。")
    print("   7. Mode 2 / 脚本控制阶段会暂停 step，不继续发动作。")
    print("   8. 保存 reward classifier 数据时，只保存 classifier_keys 对应图像。\n")

    print("===== 当前 reward classifier 数据保存配置 =====")
    print(f"env_config.ENV_IMAGE_KEYS      = {getattr(env_config, 'ENV_IMAGE_KEYS', None)}")
    print(f"env_config.image_keys          = {getattr(env_config, 'image_keys', None)}")
    print(f"env_config.classifier_keys     = {getattr(env_config, 'classifier_keys', None)}")
    print(f"classifier_image_keys          = {classifier_image_keys if classifier_image_keys is not None else 'all'}")
    print(f"classifier_strict_obs_keys     = {FLAGS.classifier_strict_obs_keys}")
    print("==============================================\n")

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    )
    listener.start()

    env = env_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )

    obs, _ = env.reset()
    print("✅ 环境已重置，开始高频采集...")

    successes = []
    failures = []

    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed, desc="✅ 已收集成功帧数")

    recent_transitions = deque(maxlen=max(0, FLAGS.pre_success_frames))
    pending_failures = deque(maxlen=max(1, FLAGS.pre_success_frames + 1))

    post_success_remaining = 0
    failure_locked = False

    last_failure_sample_time = 0.0
    failure_interval = 1.0 / FLAGS.failure_hz if FLAGS.failure_hz > 0 else 0.0

    try:
        while len(successes) < success_needed:
            base_env = env.unwrapped

            # Mode 2 / script_control_enabled=True 时，不继续 env.step()
            if getattr(base_env, "script_control_enabled", False):
                time.sleep(0.05)
                continue

            actions = np.zeros(env.action_space.shape, dtype=np.float32)
            next_obs, rew, done, truncated, info = env.step(actions)

            if "intervene_action" in info:
                actions = np.asarray(info["intervene_action"], dtype=np.float32)

            transition = make_transition(
                obs=obs,
                actions=actions,
                next_obs=next_obs,
                rew=rew,
                done=done,
                classifier_image_keys=classifier_image_keys,
            )

            # 注意：这里 obs 保持完整 next_obs，不要用裁剪后的 transition["next_observations"]。
            # 因为下一步 make_transition 仍然需要完整 obs，后续如果扩展也更安全。
            obs = next_obs

            trigger_success = False
            trigger_unlock_failure = False

            with key_lock:
                if success_presses > 0:
                    success_presses -= 1
                    trigger_success = True

                if unlock_failure_presses > 0:
                    unlock_failure_presses = 0
                    trigger_unlock_failure = True

            if trigger_unlock_failure:
                if failure_locked:
                    failure_locked = False
                    print("\n🔓 已收到 A 键：恢复失败采集。")
                else:
                    print("\nℹ️ 收到 A 键，但当前失败采集本来就是开启状态。")

            if trigger_success:
                if failure_locked or post_success_remaining > 0:
                    print("\n⚠️ 当前仍处于成功锁定/成功后续采集阶段，本次 SPACE 已忽略。")
                else:
                    pre_count = 0
                    for old_trans in recent_transitions:
                        if add_success_transition(old_trans, successes, pbar, success_needed):
                            pre_count += 1
                        if len(successes) >= success_needed:
                            break

                    curr_count = 0
                    if len(successes) < success_needed:
                        if add_success_transition(transition, successes, pbar, success_needed):
                            curr_count = 1

                    pending_failures.clear()
                    post_success_remaining = max(0, FLAGS.post_success_frames)
                    failure_locked = True

                    print(
                        f"\n✅ 成功事件触发：前帧 {pre_count} 张 + 当前帧 {curr_count} 张已记为成功；"
                        f"接下来还会自动记录后续 {post_success_remaining} 帧成功；"
                        f"失败采集已锁定，按 A 可恢复。"
                    )

            elif post_success_remaining > 0:
                if add_success_transition(transition, successes, pbar, success_needed):
                    print(
                        f"\n✅ 自动记录成功后续帧 1 张，剩余后续成功帧: {post_success_remaining - 1}"
                    )
                post_success_remaining -= 1

            else:
                now = time.time()
                should_sample_failure = (
                    (not failure_locked)
                    and (
                        failure_interval <= 0.0
                        or (now - last_failure_sample_time) >= failure_interval
                    )
                )

                if should_sample_failure:
                    pending_failures.append(copy.deepcopy(transition))
                    last_failure_sample_time = now

                    if len(pending_failures) > FLAGS.pre_success_frames:
                        failures.append(pending_failures.popleft())

            recent_transitions.append(copy.deepcopy(transition))

            status = "🔒失败锁定" if failure_locked else "🟢失败开启"
            sys.stdout.write(
                f"\r{status} | ✅ success: {len(successes)}/{success_needed} "
                f"| ❌ failures: {len(failures)} "
                f"| pending_failures: {len(pending_failures)} "
                f"| pending_post_success: {post_success_remaining}   "
            )
            sys.stdout.flush()

            # 保持原逻辑：只启动时 reset 一次，之后忽略 done/truncated
            if done or truncated:
                done = False
                truncated = False

        print("\n\n✅ 达到目标！停止录制。")

    finally:
        listener.stop()
        try:
            env.close()
        except Exception:
            pass

    if not failure_locked and post_success_remaining == 0:
        flush_pending_failures(pending_failures, failures)
    else:
        print(
            f"\nℹ️ 结束时仍处于成功锁定/成功后续阶段，"
            f"丢弃 {len(pending_failures)} 张挂起失败帧，避免标签污染。"
        )

    # 最终保存前再保险裁剪一次
    successes = sanitize_transition_list_for_classifier_storage(
        successes,
        classifier_image_keys=classifier_image_keys,
    )
    failures = sanitize_transition_list_for_classifier_storage(
        failures,
        classifier_image_keys=classifier_image_keys,
    )

    print_obs_keys_summary(successes, name="successes")
    print_obs_keys_summary(failures, name="failures")

    save_dir = os.path.join(os.path.dirname(__file__), "classifier_data_single")
    os.makedirs(save_dir, exist_ok=True)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    key_tag = "all" if classifier_image_keys is None else "-".join(classifier_image_keys)

    success_file = os.path.join(
        save_dir,
        f"{FLAGS.exp_name}_{len(successes)}_success_images_{key_tag}_{uuid}.pkl",
    )
    with open(success_file, "wb") as f:
        pkl.dump(successes, f)
        print(f"💾 成功！保存了 {len(successes)} 帧【成功画面】至 {success_file}")

    failure_file = os.path.join(
        save_dir,
        f"{FLAGS.exp_name}_{len(failures)}_failure_images_{key_tag}_{uuid}.pkl",
    )
    with open(failure_file, "wb") as f:
        pkl.dump(failures, f)
        print(f"💾 成功！保存了 {len(failures)} 帧【失败画面】至 {failure_file}")


if __name__ == "__main__":
    app.run(main)


    
# import os
# import sys
# import time
# import copy
# import pickle as pkl
# import datetime
# import threading
# from collections import deque

# from tqdm import tqdm
# import numpy as np
# from absl import app, flags
# from pynput import keyboard

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# # 单臂任务配置入口
# from examples.galaxea_task.usb_pick_insertion_single.config import env_config
# # 如果以后切回双臂，就换成：
# # from examples.galaxea_task.usb_pick_insertion.config import env_config

# FLAGS = flags.FLAGS
# flags.DEFINE_string(
#     "exp_name",
#     "galaxea_usb_insertion_single",
#     "Name of experiment corresponding to folder.",
# )
# flags.DEFINE_integer(
#     "successes_needed",
#     200,
#     "Number of successful transitions to collect.",
# )
# flags.DEFINE_integer(
#     "pre_success_frames",
#     3,
#     "Number of frames before SPACE to also label as success.",
# )
# flags.DEFINE_integer(
#     "post_success_frames",
#     3,
#     "Number of frames after SPACE to also label as success.",
# )
# flags.DEFINE_float(
#     "failure_hz",
#     10.0,
#     "Sampling frequency for negative samples.",
# )

# # ==========================================================
# # 键盘状态
# # ==========================================================
# success_presses = 0
# unlock_failure_presses = 0
# key_lock = threading.Lock()

# space_is_down = False
# a_is_down = False


# def on_press(key):
#     global success_presses, unlock_failure_presses
#     global space_is_down, a_is_down

#     try:
#         with key_lock:
#             if key == keyboard.Key.space:
#                 if not space_is_down:
#                     success_presses += 1
#                     space_is_down = True
#                     print(f"\n⌨️ 检测到 SPACE：待触发成功事件 +1（当前待消费: {success_presses}）")
#                 return

#             if hasattr(key, "char") and key.char is not None:
#                 ch = key.char.lower()
#                 if ch == "a":
#                     if not a_is_down:
#                         unlock_failure_presses += 1
#                         a_is_down = True
#                         print(f"\n⌨️ 检测到 A：请求恢复失败采集（当前待消费: {unlock_failure_presses}）")
#     except Exception as e:
#         print(f"\n⚠️ 键盘监听异常(on_press): {e}")


# def on_release(key):
#     global space_is_down, a_is_down
#     try:
#         with key_lock:
#             if key == keyboard.Key.space:
#                 space_is_down = False
#                 return

#             if hasattr(key, "char") and key.char is not None:
#                 ch = key.char.lower()
#                 if ch == "a":
#                     a_is_down = False
#     except Exception as e:
#         print(f"\n⚠️ 键盘监听异常(on_release): {e}")


# # ==========================================================
# # 工具函数
# # ==========================================================
# def make_transition(obs, actions, next_obs, rew, done):
#     return copy.deepcopy(
#         dict(
#             observations=obs,
#             actions=np.asarray(actions, dtype=np.float32),
#             next_observations=next_obs,
#             rewards=rew,
#             masks=1.0 - float(done),
#             dones=bool(done),
#         )
#     )


# def add_success_transition(transition, successes, pbar, success_needed):
#     """安全加入 success，避免超过目标数。"""
#     if len(successes) < success_needed:
#         successes.append(copy.deepcopy(transition))
#         pbar.update(1)
#         return True
#     return False


# def flush_pending_failures(pending_failures, failures):
#     """把挂起的失败样本正式写入 failures。"""
#     while pending_failures:
#         failures.append(pending_failures.popleft())


# def main(_):
#     global success_presses, unlock_failure_presses

#     print(f"🚀 开始采集视觉分类器数据：{FLAGS.exp_name}")
#     print("💡 操作指南：")
#     print("   1. 初次启动时会自动复位一次。")
#     print("   2. 进入采集后，不再自动复位。")
#     print("   3. 默认按固定频率持续收集失败样本。")
#     print("   4. 当画面达到成功状态时，按一次【空格】。")
#     print("      - 当前帧会记为成功")
#     print("      - 空格前后若干帧也会一起记为成功")
#     print("      - 同时锁定失败采集，避免成功后续画面被误标为失败")
#     print("   5. 当你确认已经离开成功状态后，按一次【A】恢复失败采集。")
#     print("   6. 长按 SPACE / A 都不会连发，必须松开后再次按下才会再触发。\n")

#     listener = keyboard.Listener(
#         on_press=on_press,
#         on_release=on_release,
#     )
#     listener.start()

#     env = env_config.get_environment(
#         fake_env=False,
#         save_video=False,
#         classifier=False,
#     )

#     obs, _ = env.reset()
#     print("✅ 环境已重置，开始高频采集...")

#     successes = []
#     failures = []

#     success_needed = FLAGS.successes_needed
#     pbar = tqdm(total=success_needed, desc="✅ 已收集成功帧数")

#     # 最近若干帧（用于成功前几帧一起打成 success）
#     recent_transitions = deque(maxlen=max(0, FLAGS.pre_success_frames))

#     # 最近挂起的 failure，先不立刻写死到 failures，避免马上又被改判成 success
#     pending_failures = deque(maxlen=max(1, FLAGS.pre_success_frames + 1))

#     # 成功后的后续 success 帧还要继续录几帧
#     post_success_remaining = 0

#     # 是否锁定失败采集
#     failure_locked = False

#     # 控制 failure 采样频率
#     last_failure_sample_time = 0.0
#     failure_interval = 1.0 / FLAGS.failure_hz if FLAGS.failure_hz > 0 else 0.0

#     try:
#         while len(successes) < success_needed:
#             actions = np.zeros(env.action_space.shape, dtype=np.float32)
#             next_obs, rew, done, truncated, info = env.step(actions)

#             if "intervene_action" in info:
#                 actions = info["intervene_action"]

#             transition = make_transition(
#                 obs=obs,
#                 actions=actions,
#                 next_obs=next_obs,
#                 rew=rew,
#                 done=done,
#             )
#             obs = next_obs

#             # --------------------------------------------------
#             # 消费按键事件
#             # --------------------------------------------------
#             trigger_success = False
#             trigger_unlock_failure = False

#             with key_lock:
#                 if success_presses > 0:
#                     success_presses -= 1
#                     trigger_success = True

#                 if unlock_failure_presses > 0:
#                     unlock_failure_presses = 0
#                     trigger_unlock_failure = True

#             # A 键：恢复失败采集
#             if trigger_unlock_failure:
#                 if failure_locked:
#                     failure_locked = False
#                     print("\n🔓 已收到 A 键：恢复失败采集。")
#                 else:
#                     print("\nℹ️ 收到 A 键，但当前失败采集本来就是开启状态。")

#             # --------------------------------------------------
#             # SPACE：触发一次成功事件
#             # --------------------------------------------------
#             if trigger_success:
#                 if failure_locked or post_success_remaining > 0:
#                     print("\n⚠️ 当前仍处于成功锁定/成功后续采集阶段，本次 SPACE 已忽略。")
#                 else:
#                     # 1) 最近若干帧也记成成功
#                     pre_count = 0
#                     for old_trans in recent_transitions:
#                         if add_success_transition(old_trans, successes, pbar, success_needed):
#                             pre_count += 1
#                         if len(successes) >= success_needed:
#                             break

#                     # 2) 当前帧记成成功
#                     curr_count = 0
#                     if len(successes) < success_needed:
#                         if add_success_transition(transition, successes, pbar, success_needed):
#                             curr_count = 1

#                     # 3) 清空 pending_failures，避免最近失败帧和成功前帧冲突
#                     pending_failures.clear()

#                     # 4) 开启成功后续帧采集，并锁定失败采集
#                     post_success_remaining = max(0, FLAGS.post_success_frames)
#                     failure_locked = True

#                     print(
#                         f"\n✅ 成功事件触发：前帧 {pre_count} 张 + 当前帧 {curr_count} 张已记为成功；"
#                         f"接下来还会自动记录后续 {post_success_remaining} 帧成功；"
#                         f"失败采集已锁定，按 A 可恢复。"
#                     )

#             # --------------------------------------------------
#             # 成功后续帧
#             # --------------------------------------------------
#             elif post_success_remaining > 0:
#                 if add_success_transition(transition, successes, pbar, success_needed):
#                     print(
#                         f"\n✅ 自动记录成功后续帧 1 张，剩余后续成功帧: {post_success_remaining - 1}"
#                     )
#                 post_success_remaining -= 1

#             # --------------------------------------------------
#             # 普通失败采样（仅在未锁定时）
#             # --------------------------------------------------
#             else:
#                 now = time.time()
#                 should_sample_failure = (
#                     (not failure_locked)
#                     and (
#                         failure_interval <= 0.0
#                         or (now - last_failure_sample_time) >= failure_interval
#                     )
#                 )

#                 if should_sample_failure:
#                     pending_failures.append(copy.deepcopy(transition))
#                     last_failure_sample_time = now

#                     # 为了避免最近几帧马上被写死成 failure，
#                     # 只把“更早的挂起失败帧”真正提交到 failures
#                     if len(pending_failures) > FLAGS.pre_success_frames:
#                         failures.append(pending_failures.popleft())

#             # 最近帧缓存永远保留，用于下一次成功事件
#             recent_transitions.append(copy.deepcopy(transition))

#             # 状态提示
#             status = "🔒失败锁定" if failure_locked else "🟢失败开启"
#             sys.stdout.write(
#                 f"\r{status} | ✅ success: {len(successes)}/{success_needed} "
#                 f"| ❌ failures: {len(failures)} "
#                 f"| pending_failures: {len(pending_failures)} "
#                 f"| pending_post_success: {post_success_remaining}   "
#             )
#             sys.stdout.flush()

#             # 保持你原来的逻辑：只启动时 reset 一次，之后忽略 done/truncated
#             if done or truncated:
#                 done = False
#                 truncated = False

#         print("\n\n✅ 达到目标！停止录制。")

#     finally:
#         listener.stop()
#         try:
#             env.close()
#         except Exception:
#             pass

#     # 结束前，把仍挂起的 failure 全部写入
#     # 只有在未锁定失败采集的情况下才落盘这些 pending failures；
#     # 如果最后停在 success 锁定状态，这些 pending 大概率靠近成功区，宁可不要。
#     if not failure_locked and post_success_remaining == 0:
#         flush_pending_failures(pending_failures, failures)
#     else:
#         print(
#             f"\nℹ️ 结束时仍处于成功锁定/成功后续阶段，"
#             f"丢弃 {len(pending_failures)} 张挂起失败帧，避免标签污染。"
#         )

#     save_dir = os.path.join(os.path.dirname(__file__), "classifier_data_single")
#     os.makedirs(save_dir, exist_ok=True)

#     uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#     success_file = os.path.join(
#         save_dir,
#         f"{FLAGS.exp_name}_{len(successes)}_success_images_{uuid}.pkl",
#     )
#     with open(success_file, "wb") as f:
#         pkl.dump(successes, f)
#         print(f"💾 成功！保存了 {len(successes)} 帧【成功画面】至 {success_file}")

#     failure_file = os.path.join(
#         save_dir,
#         f"{FLAGS.exp_name}_{len(failures)}_failure_images_{uuid}.pkl",
#     )
#     with open(failure_file, "wb") as f:
#         pkl.dump(failures, f)
#         print(f"💾 成功！保存了 {len(failures)} 帧【失败画面】至 {failure_file}")


# if __name__ == "__main__":
#     app.run(main)





