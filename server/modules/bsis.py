"""
Class untuk melakukan BSIS
"""
#%%
import cv2
import numpy as np
from typing import Union
from operator import itemgetter
import time
import os
from glob import glob
from . import util
from scipy.spatial import KDTree

#%%
class BSIS:
    class FLANN_INDEX:
        LINEAR = 0
        KDTREE = 1
        KMEANS = 2
        COMPOSITE = 3
        KDTREE_SINGLE = 4
        HIERARCHICAL = 5
        LSH = 6
        SAVED = 254
        AUTOTUNED = 255

    def __init__(self, query: Union[dict, tuple]):
        if type(query) == dict:
            self.query_kp = query['kp']
            self.query_desc = query['desc']
        elif type(query) == tuple:
            self.query_kp = query[0]
            self.query_desc = query[1]

        self.train_set = None 

        self.pairing_time = 0
        self.total_time = 0

    def set_train_directory(self, directory: str):
        train = TrainSet()
        train.set_directory(directory)
        self.train_set = train
    
    def set_train_data(self, train_data):
        if isinstance(train_data, tuple):
            if len(train_data) == 3:
                kp = train_data[0]
                desc = train_data[1]
                mapper = train_data[2]
                train = TrainSet()
                train.set_data(kp, desc, mapper)
                self.train_set = train
        elif isinstance(train_data, np.ndarray):
                kp = list()
                desc = list()
                mapper = dict()
                for i, lf in enumerate(train_data):
                    kp.append(lf.keypoint)
                    desc.append(lf.descriptor)
                    mapper[i] = lf.img
                kp = tuple(kp)
                desc = np.array(desc)
                train = TrainSet()
                train.set_data(kp, desc, mapper)
                self.train_set = train
        
        
    def make_pairs(self, algorithm, t=4, k=100):
        index_params = dict(algorithm=algorithm, trees=5)
        # search_params = dict(checks=50)   # or pass empty dictionary
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        start_time = time.time()
        cv2.setRNGSeed(1311)
        matches = flann.knnMatch(self.query_desc, self.train_set.train_desc, k=k)
        self.pairing_time = time.time() - start_time
        print('make_pairs ---', self.pairing_time)
        
        pairs = dict()
        pair_idx = 0
        for m in matches:
            m_dists = list()
            for i in m:
                m_dists.append(i.distance)

            st_dev = np.std(m_dists)
            mean = np.mean(m_dists)
            thres = mean - (t * st_dev)
            self.thres_ = thres

            for i in m:
                if i.distance < thres:
                    weight = ((i.distance - mean) / st_dev) ** 2
                    if self.train_set.index_mapper[i.trainIdx] not in pairs.keys():
                        pairs[self.train_set.index_mapper[i.trainIdx]] = list()
    
                    pairs[self.train_set.index_mapper[i.trainIdx]].append(
                        Pair(
                            pair_idx,
                            i.queryIdx,
                            i.trainIdx,
                            self.query_kp[i.queryIdx],
                            self.train_set.train_kp[i.trainIdx],
                            i.distance,
                            weight
                        )
                    )
                    pair_idx += 1
        
        return pairs

    def make_pairs_scipy(self, t=4, k=100):   
        start_time = time.time()
        tree = KDTree(self.train_set.train_desc)
        distance, index = tree.query(self.query_desc, k=k, workers=-1)
        self.pairing_time = time.time() - start_time
        print('make_pairs ---', self.pairing_time)
        
        pairs = dict()
        pair_idx = 0
        
        for queryIdx, (dist, idx) in enumerate(zip(distance, index)):
            st_dev = np.std(dist)
            mean = np.mean(dist)
            thres = mean - (t * st_dev)
            self.thres_ = thres

            for d, trainIdx in zip(dist, idx):
                if d < thres:
                    weight = ((d - mean) / st_dev) ** 2
                    if self.train_set.index_mapper[trainIdx] not in pairs.keys():
                        pairs[self.train_set.index_mapper[trainIdx]] = list()

                    pairs[self.train_set.index_mapper[trainIdx]].append(
                        Pair(
                            pair_idx,
                            queryIdx,
                            trainIdx,
                            self.query_kp[queryIdx],
                            self.train_set.train_kp[trainIdx],
                            d,
                            weight
                        )
                    )
                    pair_idx += 1
                
        return pairs

    def run_scipy(self, num_rotation=1, t=4, k=100):
        start_time = time.time()
        self.pairs = self.make_pairs_scipy(t=t, k=k)
        self.result = dict()
        
        maximum = ('', 0)
        for k, v in self.pairs.items():
            bsis = BSIS_Verify(v)
            bsis.run(num_rotation=num_rotation)
            self.result[k] = {'query_kp': bsis.selected_query_kp, 'train_kp': bsis.selected_train_kp, 'total_weight': bsis.total_weight}
            if bsis.total_weight > maximum[1]:
                maximum = (k, bsis.total_weight)

        self.total_time = time.time() - start_time
        print('total time ---', self.total_time)
        return maximum[0]

    def run(self, algorithm: int, num_rotation=1, t=4, k=100):
        start_time = time.time()
        self.pairs = self.make_pairs(algorithm=algorithm, t=t, k=k)
        self.result = dict()
        
        maximum = ('', 0)
        for k, v in self.pairs.items():
            bsis = BSIS_Verify(v)
            bsis.run(num_rotation=num_rotation)
            self.result[k] = {'query_kp': bsis.selected_query_kp, 'train_kp': bsis.selected_train_kp, 'total_weight': bsis.total_weight}
            if bsis.total_weight > maximum[1]:
                maximum = (k, bsis.total_weight)

        self.total_time = time.time() - start_time
        print('total time ---', self.total_time)
        return maximum[0]

#%%
class TrainSet:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()

        self.train_kp = list()
        self.train_desc = list()
        self.index_mapper = dict()
    
    def set_directory(self, directory):
        index = 0
        for filename in glob(os.path.join(directory, '*.jpg')):
            fname = filename.split('\\')[-1]
            kp, desc = self.sift.detectAndCompute(util.get_image(filename), None)
            for k, d in zip(kp, desc):
                self.train_kp.append(k)
                self.train_desc.append(d)
                self.index_mapper[index] = fname
                index += 1

        self.train_desc = np.array(self.train_desc)
    
    def set_data(self, kp, desc, mapper):
        self.train_kp = kp
        self.train_desc = desc
        self.index_mapper = mapper

#Class untuk menyimpan pasangan antar keypoint
class Pair:
    def __init__(self, pair_idx, query_idx, train_idx, query_kp, train_kp, distance, weight):
        self.pair_idx = pair_idx
        self.query_idx = query_idx
        self.train_idx = train_idx
        self.query_kp = query_kp
        self.train_kp = train_kp
        self.distance = distance
        self.weight = weight
    
    def print_all(self):
        string = f'pair_idx: {self.pair_idx} \n query_idx: {self.query_idx} \n train_idx: {self.train_idx} \n distance: {self.distance} \n weight: {self.weight} \n'
        print(string)

#Class untuk menyimpan informasi pasangan keypoint yang diperlukan pada tahap verifikasi BSIS
class BPair:
    def __init__(self, pair, order, weight, bts, prev):
        self.pair = pair
        self.order = order
        self.weight = weight
        self.bts = bts
        self.prev = prev
        
    def print_all(self):
        try:
            print(self.pair.pair_idx, self.order, self.weight, self.bts, self.prev.idx)
        except:
            print(self.pair.pair_idx, self.order, self.weight, self.bts, None)

#%%
class BSIS_Verify:
    def __init__(self, pairs, k=100, t=4):
        self.pairs = pairs

        self.k = k
        self.t = t

    def get_pair_list_ordered(self, by=0, query_rotation=0, train_rotation=0):
        paired_query_kp = set([(x.query_idx, x.query_kp.pt[0], x.query_kp.pt[1]) for x in self.pairs])
        query_origin = (np.mean([i[1] for i in paired_query_kp]), np.mean([i[2] for i in paired_query_kp]))
        query_order_x_set = [(i[0], util.rotate((i[1], i[2]), query_origin, query_rotation)[by]) for i in paired_query_kp]
        query_order_x_set = sorted(query_order_x_set, key=itemgetter(1))
        query_order_x = dict()
        for i, idx in enumerate(query_order_x_set):
            query_order_x[idx[0]] = i

        paired_train_kp = set([(x.train_idx, x.train_kp.pt[0], x.train_kp.pt[1]) for x in self.pairs])
        train_origin = (np.mean([i[1] for i in paired_train_kp]), np.mean([i[2] for i in paired_train_kp]))
        train_order_x_set = [(i[0], util.rotate((i[1], i[2]), train_origin, train_rotation)[by]) for i in paired_train_kp]
        train_order_x_set = sorted(train_order_x_set, key=itemgetter(1))
        # train_order_x = dict()
        # for i, idx in enumerate(train_order_x_set):
        #     train_order_x[idx[0]] = i

        train_idx_dict = dict()
        for p in self.pairs:
            if p.train_idx not in train_idx_dict.keys():
                train_idx_dict[p.train_idx] = list()
                train_idx_dict[p.train_idx].append(p)
            else:
                train_idx_dict[p.train_idx].append(p)

        ordered_list = list()
        for i, j in train_order_x_set:
            one_kp_pairs = train_idx_dict[i]
            col_list = list()
            for p in one_kp_pairs:
                col_list.append(BPair(p, query_order_x[p.query_idx], p.weight, p.weight, None))
            ordered_list.append(col_list)

        return ordered_list

    def find_best_subsequence(self, D):
        start_time = time.time()
        best_subsequence = BPair(0, 0, 0, 0, None)
        for i in range(1, len(D)):
            for j in range(0, len(D[i])):
                # start_time2 = time.time()
                dBestPrev = BPair(0, 0, 0, 0, None)
                for k in range(0, i):
                    for l in range(0, len(D[k])):
                        if D[k][l].order < D[i][j].order and D[k][l].bts > dBestPrev.bts:
                            dBestPrev = D[k][l]
                # print('find dBestPrev {} - {}'.format(i, j), time.time() - start_time2)
                D[i][j].bts = D[i][j].weight + dBestPrev.bts
                D[i][j].prev = dBestPrev
                
                if D[i][j].bts + D[i][j].weight > best_subsequence.bts + best_subsequence.weight:
                    best_subsequence = D[i][j]

        return best_subsequence

    def find_best_subsequence_(self, D):
        best_subsequence = BPair(0, 0, 0, 0, None)
        #dictionary menyimpan BPair dengan bts terbaik untuk tiap order
        best_per_order = dict()
    
        #iterasi kolom
        for i in range(1, len(D)):
            #iterasi baris pada tiap kolom
            for j in range(0, len(D[i])):
    
                if D[i][j].order not in best_per_order.keys():
                    best_per_order[D[i][j].order] = D[i][j]
                else:
                    if D[i][j].bts > best_per_order[D[i][j].order].bts:
                        best_per_order[D[i][j].order] = D[i][j]
    
                dBestPrev = BPair(0, 0, 0, 0, None)
                for ord in range(0, D[i][j].order):
                    if ord in best_per_order.keys():
                        if best_per_order[ord].bts > dBestPrev.bts:
                            dBestPrev = best_per_order[ord]
    
                D[i][j].bts = D[i][j].weight + dBestPrev.bts
                D[i][j].prev = dBestPrev
    
                if D[i][j].bts + D[i][j].weight > best_subsequence.bts + best_subsequence.weight:
                    best_subsequence = D[i][j]
    
        return best_subsequence

    def run(self, num_rotation=1):
        max_weight = 0
        selected_query_kp = list()
        selected_train_kp = list()
        self.rotation_weight = list()
        rotate_by = int(360.0 / num_rotation)
        for i in range(0, 360, rotate_by):
            query_rotation = i
            train_rotation = 0
            #print('query_rotation: ', query_rotation, '---', 'train_rotation:', train_rotation)

            get_pair_list_start = time.time()
            self.ordered = self.get_pair_list_ordered(query_rotation=query_rotation, train_rotation=train_rotation)
            
            if len(self.ordered) < 1:
                self.selected_query_kp = []
                self.selected_train_kp = []
                self.total_weight = 0
                continue
            #print('get_pair_list_ordered ---', time.time() - get_pair_list_start)

            find_best_start = time.time()
            self.bests = self.find_best_subsequence(self.ordered)
            self.selected_pairs = list()
            while(self.bests.prev != None):
                self.selected_pairs.append(self.bests.pair)
                self.bests = self.bests.prev
            else:
                pass
                #self.selected_pairs.append(self.bests.pair)
            #print('find_best_subsequence ---', time.time() - find_best_start)

            weight = sum([i.weight for i in self.selected_pairs])
            self.rotation_weight.append((i, weight))
            if weight > max_weight:
                max_weight = weight
                selected_query_kp = [i.query_kp for i in self.selected_pairs]
                selected_train_kp = [i.train_kp for i in self.selected_pairs]

        self.selected_query_kp = selected_query_kp
        self.selected_train_kp = selected_train_kp
        self.total_weight = max_weight
