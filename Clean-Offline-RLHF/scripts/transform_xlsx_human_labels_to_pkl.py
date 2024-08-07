# -*- coding: UTF-8 -*-
"""
@Project ：open_source_dataset
@File    ：transform_xlsx_human_labels_to_pkl.py
@Author  ：Yifu Yuan
@Date    ：2024/2/20
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import uuid

import os
import json


VALID_INPUT_FORMATS = [
    '.csv',
    '.xlsx'
]


def load_df(path):
    out = None
    if path.lower().endswith('.csv'):
        out = pd.read_csv(path)
    elif path.lower().endswith('.xlsx'):
        out = pd.read_excel(path)
    else:
        raise ValueError('Input file has invalid format.')
    return out


def get_file_with_extensions(directory, extensions):
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    for file in files:
        if file.endswith(tuple(extensions)):
            return file
    raise ValueError('No file with desired extension found.')


def main(args):
    domain = args.domain
    env_name = args.env_name
    data_dir = args.data_dir
    save_dir = args.save_dir
    num_query = args.num_query
    len_query = args.len_query
    feedback_type = args.feedback_type
    use_metadata = args.use_metadata

    # if there is a metadata json file available, load its contents
    if use_metadata:
        with open(get_file_with_extensions(data_dir, '.json')) as f:
            metadata = json.load(f)
            domain = metadata['domain']
            env_name = metadata['environment_name']
            num_query = metadata['query_num']
            len_query = metadata['query_length']
            feedback_type = metadata['feedback_type']

    # if the input path is a directory, search for valid inputs files in it
    if os.path.isdir(data_dir):
        data_dir = get_file_with_extensions(data_dir, VALID_INPUT_FORMATS)

    df = load_df(data_dir)

    if 'video' not in df.columns:
        start_indices_1 = df['start_indices_1'].to_numpy()
        start_indices_2 = df['start_indices_2'].to_numpy()
        human_label = np.array([json.loads(entry.replace("'", '"'))['1']['0'] for entry in df['label']])
    else:
        # legacy code
        mask = df['video'].str.contains(env_name)
        mask = mask.fillna(False)
        filtered_df = df[mask]
        print(filtered_df)

        location = filtered_df["video"].values
        start_indices_1 = []
        start_indices_2 = []
        for loc in location:
            parts = loc.split('_')
            start_indices_1.append(int(parts[2]))
            start_indices_2.append(int(parts[4]))
        start_indices_1 = np.array(start_indices_1)
        start_indices_2 = np.array(start_indices_2)

        human_label = filtered_df["label"].astype(int).values

    '''
        label in xlsx：
        0：left better
        1：equal
        2：right better

        label in RM training:
        0：left better
        1：right better
        -1：equal

        transform human label
    '''
    # 1 -> -1
    human_label = np.where(human_label == 1, -1, human_label)
    # 2 -> 1
    human_label = np.where(human_label == 2, 1, human_label)

    count_1 = np.count_nonzero(human_label == 1)
    count_0 = np.size(human_label) - np.count_nonzero(human_label)
    count_minus_1 = np.count_nonzero(human_label == -1)

    print(f"domain_{domain}_env_{env_name}:")
    print("left better (0):", count_0)
    print("right better (1):", count_1)
    print("equal (-1):", count_minus_1)

    save_dir = os.path.join(save_dir, f"{env_name}_human_labels")
    identifier = str(uuid.uuid4().hex)
    if 'video' not in df.columns:
        save_dir = os.path.join(save_dir, feedback_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    suffix = f"_domain_{domain}_env_{env_name}_num_{num_query}_len_{len_query}_{identifier}"

    assert start_indices_1.shape[0] == start_indices_2.shape[0] == human_label.shape[0] == \
           num_query, f"{env_name}: {start_indices_1.shape[0]} / {human_label.shape[0]}"

    with open(os.path.join(save_dir, "indices_1" + suffix + ".pkl"), "wb") as f:
        pickle.dump(start_indices_1, f)
    with open(os.path.join(save_dir, "indices_2" + suffix + ".pkl"), "wb") as f:
        pickle.dump(start_indices_2, f)
    with open(os.path.join(save_dir, "human_label" + suffix + ".pkl"), "wb") as f:
        pickle.dump(human_label, f)
    print("save query indices and human labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--domain', type=str, default='atari')
    parser.add_argument('--env_name', type=str, default='enduro-medium-v0', help='Environment name.')
    # parser.add_argument('--data_dir', type=str, default='Enduro.xlsx', help='query path')
    parser.add_argument('--data_dir', type=str, help='query path')
    parser.add_argument('--save_dir', type=str, default='./crowdsource_human_labels/', help='query path')
    parser.add_argument('--num_query', type=int, default=2000, help='number of query.')
    parser.add_argument('--len_query', type=int, default=40, help='length of each query.')
    parser.add_argument('--feedback_type', type=str, default='comparative', help='feedback type.')
    parser.add_argument('--use_metadata', type=bool, default=True, help='use metadata json.')

    args = parser.parse_args()
    main(args)
