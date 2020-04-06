# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod
import pickle as pkl
import numpy as np

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 0, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide =100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset( data_file , self.batch_size, transform )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 0, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

# dataset 이 episode 마다 5 * (5 + 16) * 3 * 84 * 84 를 return 하면 될듯.
class dataset_mini(object):
    def __init__(self, n_examples, n_episodes, split, seed):
        self.im_width, self.im_height, self.channels = (84, 84, 3)
        self.n_examples = n_examples
        self.n_episodes = n_episodes
        self.split = split
        self.seed = seed
        self.root_dir = '/home/hyoje/pyfiles/datasets/miniImagenet'

        self.n_label = int(self.n_examples)
        self.dataset_l = []

    def load_data_pkl(self):
        """
            load the pkl processed mini-imagenet into label,unlabel
        """
        pkl_name = '{}/data/mini-imagenet-cache-{}.pkl'.format(self.root_dir, self.split)
        print('Loading pkl dataset: {} '.format(pkl_name))

        try:
            with open(pkl_name, "rb") as f:
                data = pkl.load(f, encoding='bytes')
                image_data = data[b'image_data']
                class_dict = data[b'class_dict']
        except:
            with open(pkl_name, "rb") as f:
                data = pkl.load(f)
                image_data = data['image_data']
                class_dict = data['class_dict']

        #         print(data.keys(), image_data.shape, class_dict.keys())
        data_classes = sorted(class_dict.keys())  # sorted to keep the order

        n_classes = len(data_classes)
        print('n_classes:{}, n_label:{}'.format(n_classes, self.n_label))
        dataset_l = np.zeros([n_classes, self.n_label, self.im_height, self.im_width, self.channels], dtype=np.float32)

        for i, cls in enumerate(data_classes):
            idxs = class_dict[cls]
            np.random.RandomState(self.seed).shuffle(idxs)  # fix the seed to keep label,unlabel fixed
            dataset_l[i] = image_data[idxs[0:self.n_label]]
        print('labeled data:', np.shape(dataset_l))

        self.dataset_l = dataset_l
        self.n_classes = n_classes

        del image_data

    def next_data(self, n_way, n_shot, n_query):
        """
            get support,query,unlabel data from n_way
            get unlabel data from n_distractor
        """
        support = np.zeros([n_way, n_shot, self.im_height, self.im_width, self.channels], dtype=np.float32)
        query = np.zeros([n_way, n_query, self.im_height, self.im_width, self.channels], dtype=np.float32)

        selected_classes = np.random.permutation(self.n_classes)[:n_way]
        for i, cls in enumerate(selected_classes[0:n_way]):  # train way
            # labled data
            idx1 = np.random.permutation(self.n_label)[:n_shot + n_query]
            support[i] = self.dataset_l[cls, idx1[:n_shot]]
            query[i] = self.dataset_l[cls, idx1[n_shot:]]


        merge_images = np.concatenate((support, query), axis=1)

        return merge_images