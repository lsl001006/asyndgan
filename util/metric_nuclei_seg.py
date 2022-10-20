import torch
import numpy as np


def accuracy(output, target, istrain):
    if istrain:
        with torch.no_grad():
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            pred = pred == 1
            target = target == 1
            pred = pred.view([pred.size(0), -1]).float()
            target = target.view([target.size(0), -1]).float()
            assert pred.size() == target.size()

            TP = torch.sum((pred == 1) * (target == 1), dim=1).float()
            FP = torch.sum((pred == 1) * (target == 0), dim=1).float()
            TN = torch.sum((pred == 0) * (target == 0), dim=1).float()
            FN = torch.sum((pred == 0) * (target == 1), dim=1).float()
            acc = torch.mean((TP + TN) / (TP + TN + FP + FN + 1e-8)).item()
    else:
        pred = output > 0
        target = target > 0
        pred = np.reshape(pred, [pred.shape[0], -1]).astype(np.float)
        target = np.reshape(target, [target.shape[0], -1]).astype(np.float)
        assert pred.shape == target.shape

        TP = np.sum((pred == 1) * (target == 1), axis=1).astype(np.float)
        FP = np.sum((pred == 1) * (target == 0), axis=1).astype(np.float)
        TN = np.sum((pred == 0) * (target == 0), axis=1).astype(np.float)
        FN = np.sum((pred == 0) * (target == 1), axis=1).astype(np.float)
        acc = np.mean((TP + TN) / (TP + TN + FP + FN + 1e-8))
    return acc


def dice(output, target, istrain):
    if istrain:
        with torch.no_grad():
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            pred = pred == 1
            target = target == 1
            pred = pred.view([pred.size(0), -1]).float()
            target = target.view([target.size(0), -1]).float()
            assert pred.size() == target.size()

            TP = torch.sum((pred == 1) * (target == 1), dim=1).float()
            FP = torch.sum((pred == 1) * (target == 0), dim=1).float()
            FN = torch.sum((pred == 0) * (target == 1), dim=1).float()
            dice = torch.mean((2*TP) / (2*TP + FP + FN + 1e-8)).item()
    else:
        pred = output > 0
        target = target > 0
        pred = np.reshape(pred, [pred.shape[0], -1]).astype(np.float)
        target = np.reshape(target, [target.shape[0], -1]).astype(np.float)
        assert pred.shape == target.shape

        TP = np.sum((pred == 1) * (target == 1), axis=1).astype(np.float)
        FP = np.sum((pred == 1) * (target == 0), axis=1).astype(np.float)
        FN = np.sum((pred == 0) * (target == 1), axis=1).astype(np.float)
        dice = np.mean((2 * TP) / (2 * TP + FP + FN + 1e-8))
    return dice


def aji(pred, target, istrain):
    """ target should be instance level label"""
    assert pred.shape == target.shape

    aji = 0.0
    for k in range(pred.shape[0]):
        aji += _AJI_fast(pred[k], target[k])
    aji /= pred.shape[0]
    return aji


def _AJI_fast(pred_arr, gt):
    gs, g_areas = np.unique(gt, return_counts=True)
    assert np.all(gs == np.arange(len(gs)))
    ss, s_areas = np.unique(pred_arr, return_counts=True)
    assert np.all(ss == np.arange(len(ss)))

    if len(ss) == 1:
        return 0

    i_idx, i_cnt = np.unique(np.concatenate([gt.reshape(1, -1), pred_arr.reshape(1, -1)]),
                             return_counts=True, axis=1)
    i_arr = np.zeros(shape=(len(gs), len(ss)), dtype=np.int)
    i_arr[i_idx[0], i_idx[1]] += i_cnt
    u_arr = g_areas.reshape(-1, 1) + s_areas.reshape(1, -1) - i_arr
    iou_arr = 1.0 * i_arr / u_arr

    i_arr = i_arr[1:, 1:]
    u_arr = u_arr[1:, 1:]
    iou_arr = iou_arr[1:, 1:]

    j = np.argmax(iou_arr, axis=1)

    c = np.sum(i_arr[np.arange(len(gs) - 1), j])
    u = np.sum(u_arr[np.arange(len(gs) - 1), j])
    used = np.zeros(shape=(len(ss) - 1), dtype=np.int)
    used[j] = 1
    u += (np.sum(s_areas[1:] * (1 - used)))
    return 1.0 * c / u


