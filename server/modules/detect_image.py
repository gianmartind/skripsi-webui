from modules.bsis import BSIS
from modules.clustermodel import ClusterModel
from modules import util
import cv2
import time
import json

with open('./static/paths.json') as p:
    paths = json.load(p)

models_dir = paths['models_dir']
dataset_dir = paths['dataset_dir']
detection_dir = paths['detection_dir']
curr_itr = 0

def detect_image(model, consistency, uniqueness, img_dir):
    global models_dir
    global dataset_dir
    global detection_dir
    global curr_itr

    #load ClusterModel
    cm = ClusterModel()
    cm.load(f'{models_dir}/{model}')
    maxsize = cm.meta.maxsize

    #get train_data
    train_data = cm.get_local_features(min_consistency=consistency, min_uniqueness=uniqueness)

    #load image
    img_name = img_dir.split('/')[-1]
    img = util.get_image(img_dir, maxheight=maxsize, maxwidth=maxsize)
    
    if cm.meta.desc_type == ClusterModel.Descriptor.SIFT:
        extract_method = cv2.SIFT_create()
        algorithm = BSIS.FLANN_INDEX.KDTREE
    elif cm.meta.desc_type == ClusterModel.Descriptor.ORB:
        extract_method = cv2.ORB_create()
        algorithm = BSIS.FLANN_INDEX.LSH

    #detect keypoints
    kp, desc = extract_method.detectAndCompute(img, None)
    query = {
        'kp': kp,
        'desc': desc
    }

    bsis_param = dict(
        num_rotation=20, 
        algorithm=algorithm,
        k=100, 
        t=3
    )

    #BSIS
    bsis = BSIS(query)
    bsis.set_train_data(train_data)
    most_similar = bsis.run(**bsis_param)

    #Result
    res = list()
    for i, (k, v) in enumerate(sorted(bsis.result.items(), key=lambda i: i[1]['total_weight'], reverse=True)[:10]):
        train_img_class = k.split('_')[0]
        train_img_dir = f'{dataset_dir}/{train_img_class}/{k}'
        train_img = util.get_image(train_img_dir, maxheight=maxsize, maxwidth=maxsize)

        img_matches = util.show_matches(img, v['query_kp'], train_img, v['train_kp'], return_img=True)
        img_matches_name = f'detection{curr_itr}_{i}.jpg'
        img_matches_dir = f'{detection_dir}/{img_matches_name}'
        cv2.imwrite(img_matches_dir, img_matches)

        if v['total_weight'] > 0:
            res.append((k, round(v['total_weight'], 3), img_matches_name))
        else:
            break
    
    if curr_itr == 0:
        curr_itr = 1
    else:
        curr_itr = 0

    return {
        'list': res,
        'total_time': bsis.total_time
    }
