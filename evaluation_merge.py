import os
import numpy as np
from shapely.geometry import Polygon
from math import sqrt
import matplotlib.pyplot as plt

test_path = 'DOTA_demo_view/detection/result_txt/result_before_merge/'
label_path = 'DOTA_demo_view/labelTxt/'
iou_ap_thresh = 0.5
iou_nms_thresh = 0.05


def get_iou(_x, _y, _ap):
    a = np.array(_x[0:8], dtype=np.float64).reshape(4, 2)
    poly1 = Polygon(a).convex_hull
    b = np.array(_y[0:8], dtype=np.float64).reshape(4, 2)
    poly2 = Polygon(b).convex_hull

    if not poly1.intersects(poly2):
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area
        if _ap:
            iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
        else:
            iou = float(inter_area) / min(poly1.area, poly2.area)
    # print(iou)
    return iou


def nms(obb):
    if len(obb) == 0:
        return []
    obb = np.array(obb, dtype=np.float64).copy()
    ind = np.argsort(obb[:, -1])[::-1]
    obb = obb[ind]
    tag = np.ones(obb.shape[0])
    start = 0
    for i in range(obb.shape[0]):
        # if obb[i][0] == 513.0 and obb[i][1] == 2743.0:
        #     start = 1
        #     print(i, tag[i], obb[i])
        if tag[i] == 0:
            continue
        for j in range(i+1, obb.shape[0]):
            if tag[j] and get_iou(obb[i], obb[j], 0) > iou_nms_thresh:
                tag[j] = 0
    obb = np.array([obb[x] for x in range(len(tag)) if tag[x]])
    # if start:
    #     print(obb)
    return obb


def get_conf(cord):
    # distance to the Border
    cord = np.array(cord, dtype=np.float64).copy()[:8]
    return min([min(x, 1024-x) for x in cord])


def dist(a, b):
    return sqrt(a*a+b*b)


def square(cord):
    cord = np.array(cord, dtype=np.float64).copy()[:8]
    w0 = dist(cord[1*2] - cord[0*2], cord[1*2+1] - cord[0*2+1])
    w1 = dist(cord[3*2] - cord[2*2], cord[3*2+1] - cord[2*2+1])
    h0 = dist(cord[2*2] - cord[1*2], cord[2*2+1] - cord[1*2+1])
    h1 = dist(cord[0*2] - cord[3*2], cord[0*2+1] - cord[3*2+1])
    # return fabs(w0-w1) < 35 and (h0-h1) < 35
    return 1


def bad_label(_name, _data):
    if (_name == '49' and _data[0] == 1977.0) or\
            (_name == '1296' and _data[0] == 954.0) or\
            (_name == '850'):
        # print(_name, dx, dy, _data)
        return 1
    return 0


if __name__ == "__main__":
    path_dir = os.listdir(test_path)
    gt = {}
    pred = {}
    gt_total = 0
    pred_total = 0
    for file in path_dir:
        f = open(test_path+file)
        st = f.readline().split(' ')
        name = file.split('.')[0]
        pred[name] = []
        while len(st)-(st[0] == ''):
            dx = float(st[0].split('_')[-4])
            dy = float(st[0].split('_')[-1])
            data = [float(st[x])+(dy if x & 1 else dx) for x in range(2, 10)]
            data.append(float(st[1]))
            pred[name].append(data)
            st = f.readline().split(' ')
        pred[name] = nms(pred[name])
        pred_total += len(pred[name])
    path_dir = os.listdir(label_path)
    flag = []
    for file in path_dir:
        # file = '7__1__512___0.txt'
        f = open(label_path+file)
        st = f.readline().split(' ')
        name = file.split('.')[0].split('_')
        dx = float(name[-4])
        dy = float(name[-1])
        name = name[0]
        if name not in gt:
            gt[name] = []
        while len(st)-(st[0] == ''):
            data = [float(st[x])+(dy if x & 1 else dx) for x in range(8)]
            if bad_label(name, data):
                st = f.readline().split(' ')
                continue
            data.append(gt_total)
            data.append(get_conf(st[:-2]))
            flag.append(0)
            gt_total += 1
            gt[name].append(data)
            st = f.readline().split(' ')
        if name == '1339':
            # add plane
            gt[name].append([370., 748., 348., 714., 387., 689., 409., 723., gt_total, 300])
            gt_total += 1
            flag.append(0)
    gt_total = 0
    for file in gt:
        if len(gt[file]) == 0:
            continue
        gt[file] = nms(gt[file])
        gt_total += len(gt[file])
    q = []
    for file in gt:
        if file not in pred or len(pred[file]) == 0:
            continue
        for x in pred[file]:
            fp = 0
            p = 0
            iou_max = 0
            for y in gt[file]:
                _iou = get_iou(x, y, 1)
                if _iou > iou_max:
                    iou_max = _iou
                    p = int(y[-2])
            if iou_max > iou_ap_thresh and not flag[p]:
                flag[p] = 1
                fp = 1
            if fp == 0:
                if get_conf(x) < 0:
                    continue
                if x[-1] > 0.6:
                    print(file, x)
            else:
                if x[-1] < 0.5:
                    print(file, x[-1])
            #         print(gt[file])
            #         print(pred[file])
            #         print(iou_max, x)
            q.append([x[-1], fp])
        for y in gt[file]:
            if flag[int(y[-2])] == 0:
                print(file, y)
    q = np.array(q, dtype=np.float64)
    idx = np.argsort(q[:, 0])[::-1]
    q = q[idx]
    pre = 0
    rec = 0
    fp = 0
    ap = 0
    last = 1
    conf_min = 1
    roc = []
    for x in q:
        if x[1]:
            pre += 1
            rec += 1
            # print('rec=%.5f prec=%.5f' % (rec/gt_total, pre/(pre+fp)))
            roc.append([rec/gt_total, pre/(pre+fp), x[0]])
            if rec <= gt_total:
                ap += (last+pre/(pre+fp))/2
                conf_min = x[0]
                # print(x[0])
        else:
            fp += 1
            last = pre/(pre+fp)
    if rec < gt_total:
        roc.append([1, 0, 0])
    print('rec=%.5f prec=%.5f' % (rec/gt_total, pre/(pre+fp)))
    ap /= gt_total
    print('AP-%d: %.5f' % (iou_ap_thresh*100, ap))
    print('Oriented Bounding Boxes: %d' % q.shape[0])
    print('Ground-Truth Total: %d' % gt_total)
    print('Lowest TP-Conf: %.5f' % conf_min)

    # plotting roc-curve
    roc = np.array(roc, dtype=np.float64)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(roc[:, 0], roc[:, 1], label='roc', color='r')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.9, 1)
    ax1.set_ylabel('Precision')
    ax1.set_xlabel('Recall')
    ax2.plot(roc[:, 0], roc[:, 2], label='conf', color='b')
    ax2.set_ylim(0.1, 1)
    ax2.set_ylabel('Conf')
    plt.title('YOLOv5x-epoch450-conf0.176')
    fig.legend(bbox_to_anchor=(0.6, 1.), bbox_transform=ax1.transAxes)
    plt.show()

