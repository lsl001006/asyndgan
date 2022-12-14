import torch
import numpy as np
import numexpr as ne
import SimpleITK as sitk

        
def dice_coef(output, target):#output为预测结果 target为真实结果
    smooth = 1e-5 #防止0除
 
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
 
    intersection = (output * target).sum()
 
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

        
def sensitivity(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
 
    intersection = (output * target).sum()
 
    return (intersection + smooth) / \
        (target.sum() + smooth)

def specificity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    rev_output = (1-output)
    rev_target = (1-target)
    intersection = (rev_output*rev_target).sum()

    return (intersection + smooth) / \
        (rev_target.sum() + smooth)

def hausdorff_distance(lT,lP):
    """
    lP: label_Prediction
    lT: label_Ground_Truth
    """
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    return hausdorffcomputer.GetAverageHausdorffDistance()#hausdorffcomputer.GetHausdorffDistance()

# fast version of Aggregated Jaccrd Index
def agg_jc_index(mask, pred):
    from tqdm import tqdm_notebook
    """Calculate aggregated jaccard index for prediction & GT mask
    reference paper here: https://www.dropbox.com/s/j3154xgkkpkri9w/IEEE_TMI_NuceliSegmentation.pdf?dl=0
    mask: Ground truth mask, shape = [1000, 1000, instances]
    pred: Prediction mask, shape = [1000,1000], dtype = uint16, each number represent one instance
    Returns: Aggregated Jaccard index for GT & mask 
    """
    def compute_iou(m, pred, pred_mark_isused, idx_pred):
        # check the prediction has been used or not
        if pred_mark_isused[idx_pred]:
            intersect = 0
            union = np.count_nonzero(m)
        else:
            p = (pred == idx_pred)
            # replace multiply with bool operation
            s = ne.evaluate("m&p")
            intersect = np.count_nonzero(s)
            union = np.count_nonzero(m) + np.count_nonzero(p) - intersect
        return (intersect, union)
    
    mask=mask.astype(np.bool)
    c = 0 # count intersection
    u = 0 # count union
    pred_instance = pred.max() # predcition instance number
    pred_mark_used = [] # mask used
    pred_mark_isused = np.zeros((pred_instance+1), dtype=bool)
    
    for idx_m in range(len(mask[0,0,:])):
        m = np.take(mask, idx_m, axis=2)     
        
        intersect_list, union_list = zip(*[compute_iou(m, pred, pred_mark_isused, idx_pred) for idx_pred in range(1, pred_instance+1)])

        iou_list = np.array(intersect_list) / np.array(union_list)    
        hit_idx = np.argmax(iou_list)
        c += intersect_list[hit_idx]
        u += union_list[hit_idx]
        pred_mark_used.append(hit_idx)
        pred_mark_isused[hit_idx+1] = True
        
    pred_mark_used = [x+1 for x in pred_mark_used]
    pred_fp = set(np.unique(pred)) - {0} - set(pred_mark_used)
    pred_fp_pixel = np.sum([np.sum(pred==i) for i in pred_fp])

    u += pred_fp_pixel
    print (c / u)
    return (c / u)