# python prune_demo_episodes.py \
#   --input /home/eren/HIL-SERL/HIL-SERL-Project/examples/demo_data_single/galaxea_usb_insertion_single_30_demos_abs2rel_feedback_2026-04-29_16-18-23.pkl \
#   --drop_zero_based 3 20

# prune_demo_episodes.py
import argparse
import os
import pickle
import copy
import numpy as np


def as_float(x, default=0.0):
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def as_bool(x):
    try:
        return bool(np.asarray(x).reshape(-1)[0])
    except Exception:
        return bool(x)


def is_episode_end(tr):
    done = tr.get("dones", tr.get("done", False))
    truncated = tr.get("truncated", tr.get("truncates", False))
    mask = tr.get("masks", tr.get("mask", 1.0))
    reward = tr.get("rewards", tr.get("reward", 0.0))

    return (
        as_bool(done)
        or as_bool(truncated)
        or as_float(mask, 1.0) == 0.0
        or as_float(reward, 0.0) > 0.0
    )


def extract_transitions(payload):
    if isinstance(payload, list):
        return payload, "list", None

    if isinstance(payload, dict):
        for key in ["transitions", "data", "demo_data"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key], "dict", key

    raise TypeError(f"Unsupported pkl payload type: {type(payload)}")


def split_episodes(transitions):
    episodes = []
    cur = []

    for tr in transitions:
        cur.append(tr)
        if is_episode_end(tr):
            episodes.append(cur)
            cur = []

    if cur:
        episodes.append(cur)

    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 30 demos pkl")
    parser.add_argument("--output", default=None, help="输出 pkl")
    parser.add_argument(
        "--drop_zero_based",
        nargs="+",
        type=int,
        default=[],
        help="按 0-based episode index 删除，例如 3 20",
    )
    parser.add_argument(
        "--drop_one_based",
        nargs="+",
        type=int,
        default=[],
        help="按 1-based 第几条删除，例如 3 20 会转成 2 19",
    )
    args = parser.parse_args()

    drop = set(args.drop_zero_based)
    drop.update(i - 1 for i in args.drop_one_based)

    with open(args.input, "rb") as f:
        payload = pickle.load(f)

    transitions, payload_type, dict_key = extract_transitions(payload)
    episodes = split_episodes(transitions)

    print(f"input transitions : {len(transitions)}")
    print(f"input episodes    : {len(episodes)}")
    print(f"drop zero-based   : {sorted(drop)}")

    keep_episodes = []
    for i, ep in enumerate(episodes):
        if i in drop:
            print(f"DROP episode {i}: length={len(ep)}")
        else:
            keep_episodes.append(ep)

    new_transitions = [tr for ep in keep_episodes for tr in ep]

    new_payload = copy.deepcopy(payload)
    if payload_type == "list":
        new_payload = new_transitions
    else:
        new_payload[dict_key] = new_transitions

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        suffix = "_pruned_drop_" + "_".join(map(str, sorted(drop)))
        args.output = base + suffix + ext

    with open(args.output, "wb") as f:
        pickle.dump(new_payload, f)

    print(f"output transitions: {len(new_transitions)}")
    print(f"output episodes   : {len(keep_episodes)}")
    print(f"saved             : {args.output}")


if __name__ == "__main__":
    main()