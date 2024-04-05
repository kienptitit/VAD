import numpy as np


def make_patch_label(num_frames, anomaly_frames, seg_size=32):
    frame_label = np.zeros(num_frames)
    for start, end in anomaly_frames:
        frame_label[start:end] = 1
    interval = np.linspace(0, num_frames - 1, seg_size + 1).astype(np.int32)
    label = np.zeros(seg_size)
    for i in range(seg_size):
        start = interval[i]
        end = interval[i + 1]
        if 1 in frame_label[start:end].tolist():
            label[i] = 1
    return label


def make_patch_label_MGFN(num_frames, anomaly_frames, seg_size=16):
    """
    :param num_frames:
    :param anomaly_frames:
    :param seg_size:
    :return:
    """
    frame_label = np.zeros(num_frames)
    for start, end in anomaly_frames:
        frame_label[start:end] = 1

    interval = np.linspace(0, num_frames - 1, num_frames // seg_size + 1, dtype=np.int32)
    label = np.zeros(num_frames // seg_size)
    for i in range(len(label)):
        start = interval[i]
        end = interval[i + 1]
        if 1 in frame_label[start:end].tolist():
            label[i] = 1
    return label


def get_all_label(label_path, mode='MGFN'):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            num_frames = int(line.split(" ")[1])
            anomaly_tuple = line.split(" ")[3:]
            anomaly_list = []
            for idx in range(len(anomaly_tuple) // 2):
                start = int(anomaly_tuple[2 * idx])
                end = int(anomaly_tuple[2 * idx + 1])
                if start > 0:
                    anomaly_list.append((start, end))
            if mode != 'MGFN':
                labels.append(make_patch_label(num_frames, anomaly_list))
            else:
                labels.append(make_patch_label_MGFN(num_frames, anomaly_list))
    return np.stack(labels) if mode != 'MGFN' else labels
