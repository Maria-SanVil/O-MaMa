import numpy as np 
from pycocotools import mask as mask_utils


def add_to_json(dataset, idx_dataset, 
                pred_mask2, confidence,
                processed, pred_json, gt_json):
    
    mask_annotations = dataset.mask_annotations
    reverse = dataset.reverse
    if reverse:
        # img_pth2 ego, img_pth1 exo
        setting = "exo-ego"
        img_pth2, _, img_pth1, _ = dataset.pairs[idx_dataset]
    else:
        setting = "ego-exo"
        img_pth1, _, img_pth2, _ = dataset.pairs[idx_dataset]

    _, take_id, cam, obj, _, idx = img_pth1.split('//')
    _, _, cam2, obj2, _, _ = img_pth2.split('//')
    
    if take_id not in processed:
        processed[take_id] = []
        pred_json[take_id] = {'masks': {}, 'subsample_idx': []}
        gt_json[take_id] = {'masks': {}, 'annotated_frames': {}}

                
    if obj not in processed[take_id]:
        processed[take_id].append(obj)
        #Ensure the object is in the json
        pred_json[take_id]['masks'][obj] = {}
        gt_json[take_id]['masks'][obj] = {}
        pred_json[take_id]['masks'][obj][f'{cam}_{cam2}'] = {}
        gt_json[take_id]['masks'][obj][cam] = {}
        gt_json[take_id]['masks'][obj][cam2] = {}
        gt_json[take_id]['annotated_frames'][obj] = {}
        gt_json[take_id]['annotated_frames'][obj][cam] = []
        gt_json[take_id]['annotated_frames'][obj][cam2] = []

            
    if idx not in pred_json[take_id]['subsample_idx']:
        pred_json[take_id]['subsample_idx'].append(idx)
    if idx not in gt_json[take_id]['masks'][obj][cam].keys():
        gt_json[take_id]['masks'][obj][cam][idx] = mask_annotations[take_id]['masks'][obj][cam][idx]
        gt_json[take_id]['annotated_frames'][obj][cam].append(int(idx))
        if idx in mask_annotations[take_id]['masks'][obj2][cam2]: 
            gt_json[take_id]['masks'][obj][cam2][idx] = mask_annotations[take_id]['masks'][obj][cam2][idx]
        gt_json[take_id]['annotated_frames'][obj][cam2].append(int(idx))

    pred2 = mask_utils.encode(np.asfortranarray(pred_mask2.astype(np.uint8)))
    pred2['counts'] = pred2['counts'].decode('ascii')
    pred_json[take_id]['masks'][obj][f'{cam}_{cam2}'][idx] = {'pred_mask': pred2, 'confidence': confidence}

    return pred_json, gt_json


from evaluation.evaluate_egoexo import evaluate_egoexo
from evaluation.evaluate_exoego import evaluate_exoego

def evaluate(gt_json, pred_json, reverse):
    if reverse:
        setting = "exo-ego"
        output = {'version': "00",  "challenge": "correspondence",setting: {'results': pred_json}}
        gt = {'version': "00",  "challenge": "correspondence",'annotations': gt_json}
        out_dict = evaluate_exoego(gt, output)
    else:
        setting = "ego-exo"
        output = {'version': "00",  "challenge": "correspondence",setting: {'results': pred_json}}
        gt = {'version': "00",  "challenge": "correspondence",'annotations': gt_json}
        out_dict = evaluate_egoexo(gt, output)
    return out_dict