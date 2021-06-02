import os
import numpy as np
from shapely.geometry import Polygon


test_path = 'DOTA_demo_view/detection/result_txt/result_before_merge/'
label_path = 'DOTA_demo_view/labelTxt/'
iou_thresh = 0.5


def get_iou():
    a = np.array(x[0:8], dtype=np.float64).reshape(4, 2)
    poly1 = Polygon(a).convex_hull
    b = np.array(y[0:8], dtype=np.float64).reshape(4, 2)
    poly2 = Polygon(b).convex_hull

    if not poly1.intersects(poly2):
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area
        iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
    # print(iou)
    return iou


if __name__ == "__main__":
    path_dir = os.listdir(test_path)
    gt = {}
    pred = {}
    gt_total = 0
    pred_total = 0
    for file in path_dir:
        f = open(test_path+file)
        st = f.readline().split(' ')
        while len(st)-(st[0] == ''):
            if st[0] not in pred:
                pred[st[0]] = []
            data = st[2:10].copy()
            data.append(st[1])
            data.append(pred_total)
            pred[st[0]].append(data)
            pred_total += 1
            st = f.readline().split(' ')
    path_dir = os.listdir(label_path)
    flag = []
    for file in path_dir:
        # file = '7__1__512___0.txt'
        f = open(label_path+file)
        st = f.readline().split(' ')
        name = file.split('.')[0]
        while len(st)-(st[0] == ''):
            if name not in gt:
                gt[name] = []
            data = st[0:8].copy()
            data.append(gt_total)
            flag.append(0)
            gt_total += 1
            gt[name].append(data)
            st = f.readline().split(' ')
        # break
    q = []
    for file in gt:
        if file not in pred:
            continue
        pred[file] = np.array(pred[file], dtype=np.float64)
        idx = np.argsort(pred[file][:, -1])[::-1]
        pred[file] = pred[file][idx]
        for x in pred[file]:
            fp = 0
            p = 0
            iou_max = 0
            for y in gt[file]:
                _iou = get_iou()
                if _iou > iou_max:
                    iou_max = _iou
                    p = int(y[-1])
            if iou_max > iou_thresh and not flag[p]:
                flag[p] = 1
                fp = 1
            q.append([x[8], fp])
    q = np.array(q, dtype=np.float64)
    idx = np.argsort(q[:, 0])[::-1]
    q = q[idx]
    pre = 0
    rec = 0
    fp = 0
    ap = 0
    last = 1
    for x in q:
        if x[1]:
            pre += 1
            rec += 1
            print('rec=%.5f prec=%.5f' % (rec/gt_total, pre/(pre+fp)))
            if rec <= gt_total:
                ap += (last+pre/(pre+fp))/2
        else:
            fp += 1
            last = pre/(pre+fp)
    print('rec=%.5f prec=%.5f' % (1, pre/(pre+fp)))
    # if rec < gt_total:
    #     ap += pre/(pre+fp)*(gt_total-rec)
    ap /= gt_total
    print('AP-%d: %.5f' % (iou_thresh*100, ap))
    print('Oriented Bounding Boxes: %d' % q.shape[0])
    print('Ground-Truth Total: %d' % gt_total)
