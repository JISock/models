import pickle
import numpy as np
import matplotlib.pyplot as plt

def IoU(box1,box2):
    # def IoU(Ymin1, Xmin1, Ymax1, Xmax1, Ymin2, Xmin2, Ymax2, Xmax2):
    box1 = map(float,box1)
    box2 = map(float,box2)
    Ymin1 = box1[0]
    Xmin1 = box1[1]
    Ymax1 = box1[2]
    Xmax1 = box1[3]
    Ymin2 = box2[0]
    Xmin2 = box2[1]
    Ymax2 = box2[2]
    Xmax2 = box2[3]

    # left top corner
    x1BboxA = Xmin1
    y1BboxA = Ymin1
    #right bottom corner
    x2BboxA = Xmax1
    y2BboxA = Ymax1

    x1BboxB = Xmin2
    y1BboxB = Ymin2
    x2BboxB = Xmax2
    y2BboxB = Ymax2

    lengthBboxA = Xmax1-Xmin1
    heightBboxA = Ymax1 - Ymin1
    lengthBboxB = Xmax2 - Xmin2
    heightBboxB = Ymax2 - Ymin2

    # area of the bounding box
    areaA = lengthBboxA * heightBboxA
    areaB = lengthBboxB * heightBboxB

    x1 = max(x1BboxA, x1BboxB)
    y1 = max(y1BboxA, y1BboxB)
    x2 = min(x2BboxA, x2BboxB)
    y2 = min(y2BboxA, y2BboxB)

    # skip if there is no intersection
    w = x2 - x1
    if w <= 0:
        return 0
    h = y2 - y1
    if h <= 0:
        return 0
    intersectAB = w * h
    overlapRatio = float(intersectAB )/ float(areaA + areaB - intersectAB)
    return overlapRatio

def fn_IoU_table(Detection,GT):
    table = np.zeros([Detection.shape[0],GT.shape[0]])

    for dt in range(0,Detection.shape[0]):
        for gt in range(0,GT.shape[0]):
            table[dt,gt] = IoU(Detection[dt,:],GT[gt,:])
    return table

def fn_AssignDetectionToGT(Detection,GT,score,threshold):
    # sort
    ids = score.argsort()[::-1]
    score = [score[i] for i in ids]
    Detection = Detection[ids,:]
    labels = np.zeros([Detection.shape[0],])
    table = fn_IoU_table(Detection,GT)
    for dt in range(Detection.shape[0]):
        v = max(table[dt,:])
        maxind = table[dt,:].argmax()
        if v>= threshold:
            labels[dt] = 1
            table[:,maxind] = -float('Inf')
        else:
            labels[dt] = 0

    return labels,score










# with open('../synthetic_data_training_depth_cups/testing/results_synthetic_dataset.pkl','r') as f:
with open('../synthetic_data_training_depth_cups/testing/Detection_results_and_GT_real_data_46983.pkl', 'r') as f:
# with open('../synthetic_data_training_depth_cups/testing/Detection_results_and_GT_real_data_tejani.pkl', 'r') as f:
    results = pickle.load(f)

# a = IoU(results[0]['detected_boxes'][0,:],results[0]['GroundTruth'][0,:])
threshold = 0.5
accumlabels = []
accumscores = []
numExpected =0
for c,result in enumerate(results):
    labels,score = fn_AssignDetectionToGT(result['detected_boxes'],result['GroundTruth'],result['detected_scores'],threshold)
    accumlabels=accumlabels+list(labels)
    accumscores=accumscores+list(score)
    numExpected+=result['GroundTruth'].shape[0]


#sort highest to lowest
sortind = np.argsort(accumscores)[::-1]
accumlabels = [accumlabels[i] for i in list(sortind)]
accumscores = [accumscores[i] for i in list(sortind)]
binarylabelsTP = [(accumlabels[i]>0.0) for i in range(0,len(accumlabels))]
binarylabelsFP = [(accumlabels[i]<=0.0) for i in range(0,len(accumlabels))]

tp = np.cumsum(binarylabelsTP)
fp = np.cumsum(binarylabelsFP)

precision = np.divide(map(float,tp) , np.add(map(float,tp),map(float,fp)))
recall = np.divide(map(float,tp) , float(numExpected))
deltaRecall = 1.0/numExpected
ap = np.sum(np.multiply(precision,binarylabelsTP))*deltaRecall
plt.plot(recall,precision)