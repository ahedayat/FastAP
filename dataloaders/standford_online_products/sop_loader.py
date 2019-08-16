import gc
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class StandfordOnlineProductsLoader(Dataset):
    def __init__(self, data_root, data_transform=None, pos_neighbor=False, neg_neighbor=False, eval_mode=False, eval_num_retrieval=10):
        """
            /
            |_images/
            |_images_info.txt
            |_super_class_ids.txt

            label = super_class
        """
        assert type(eval_num_retrieval)==int, 'eval_num_retrieval: expected int but get {}'.format(type(eval_num_retrieval))
        self.data_root = data_root
        self.images_path = '{}/images'.format(self.data_root)
        self.data_transform = data_transform
        self.pos_neighbor = pos_neighbor
        self.neg_neighbor = neg_neighbor
        self.eval_mode = eval_mode
        self.eval_num_retrieval = eval_num_retrieval
        
        self.images_info = [ line.split() for line in open('{}/images_info.txt'.format(self.data_root)) ]
        self.super_class_ids = [ int(line.split()[0]) for line in open('{}/super_class_ids.txt'.format(self.data_root)) ]
        
        self.label_to_class = dict()
        self.class_to_image_id = dict()
        self.id_to_index = dict()
        for ix, (image_info) in enumerate(self.images_info):
            super_class_id, class_id, image_id, image_name = image_info
            super_class_id, class_id, image_id = int(super_class_id), int(class_id), int(image_id)

            if super_class_id not in self.label_to_class.keys():
                self.label_to_class[super_class_id] = list()
            self.label_to_class[super_class_id].append(class_id)

            if class_id not in self.class_to_image_id.keys():
                self.class_to_image_id[class_id] = list()
            self.class_to_image_id[class_id].append(image_id)
            self.id_to_index[image_id] = ix
    
    def num_classes(self):
        return len(self.super_class_ids)

    def pil_to_np(self, image):
        image = np.array(image)
        if len(image.shape)!=3 or image.shape[2]!=3:
            image_reshaped = np.zeros( (image.shape[0], image.shape[1], 3) )
            mean_image = np.mean(image, axis=0)
            image_reshaped[:, :, 0] = mean_image
            image_reshaped[:, :, 1] = mean_image
            image_reshaped[:, :, 2] = mean_image
            image = image_reshaped
        return image

    def __train_getitem__(self, ix):
        super_class_id, class_id, image_id, image_name = self.images_info[ix]
        super_class_id, class_id, image_id = int(super_class_id), int(class_id), int(image_id)

        image = Image.open('{}/{}'.format(self.images_path, image_name) )
        
        pos_image, neg_image = [], []
        pos_image_name, neg_image_name = None, None
        if self.pos_neighbor:
            pos_neighbor_classes = self.label_to_class[super_class_id].copy()
            pos_neighbor_classes.remove( class_id ) 
            pos_class = random.choice(pos_neighbor_classes)
            pos_image_id = random.choice(self.class_to_image_id[pos_class])
            pos_index = self.id_to_index[pos_image_id]
            _, _, _, pos_image_name = self.images_info[pos_index]
            pos_image = Image.open('{}/{}'.format(self.images_path, pos_image_name))
            del pos_neighbor_classes, pos_class, pos_image_id, pos_index#, pos_image_name
            gc.collect()

        if self.neg_neighbor:
            neg_super_classes_ids = self.super_class_ids.copy()
            neg_super_classes_ids.remove(super_class_id)
            neg_super_classes_id = random.choice(neg_super_classes_ids)
            neg_class = random.choice( self.label_to_class[neg_super_classes_id] )
            neg_image_id = random.choice(self.class_to_image_id[neg_class])
            neg_index = self.id_to_index[neg_image_id]
            _, _, _, neg_image_name = self.images_info[neg_index]
            neg_image = Image.open('{}/{}'.format(self.images_path, neg_image_name))
            del neg_super_classes_ids, neg_super_classes_id, neg_class, neg_image_id, neg_index#, neg_image_name
            gc.collect()

        if self.data_transform is not None:
            image = self.data_transform( image )
            if self.pos_neighbor:
                pos_image = self.data_transform( pos_image )
            if self.neg_neighbor:
                neg_image = self.data_transform( neg_image )
        image = self.pil_to_np(image)
        if self.pos_neighbor:
            pos_image = self.pil_to_np( pos_image )
        if self.neg_neighbor:
            neg_image = self.pil_to_np( neg_image )
        # print('{}. i:{},{} +:{},{} -:{},{}'.format(ix is None, image is None, image_name is None, pos_image is None, pos_image_name is None, neg_image is None, neg_image_name is None))

        # print('{}. i:{},{} +:{},{} -:{},{}'.format(ix, image.shape, image_name, pos_image.shape, pos_image_name, neg_image.shape, neg_image_name))

        output = np.zeros((len(self.super_class_ids,)))
        output[ self.super_class_ids.index(super_class_id) ] = 1

        image = torch.tensor(image, dtype=torch.float32)
        if self.pos_neighbor:
            pos_image = torch.tensor( pos_image, dtype=torch.float32 )
        if self.neg_neighbor:
            neg_image = torch.tensor( neg_image, dtype=torch.float32 )
        output = torch.tensor(output, dtype=torch.float32)

        # print('{}. i:{} +:{} -:{} t:{} {}'.format(ix, image.size(), pos_image.size(), neg_image.size(), output.size(), image_name))
        
        return image, pos_image, neg_image, output
        
    def __eval_getitem__(self, ix):
        image, pos_image, neg_image, output = self.__train_getitem__(ix)

        super_class_id = torch.argmax(output, dim=0)

        retrival_images = None
        retrival_labels = None
        for label in self.label_to_class.keys():
            classes = list()
            for jx in range(self.eval_num_retrieval):
                if len(classes)==0:
                    classes = self.label_to_class[label].copy()

                rand_class = random.choice( classes )
                rand_image_id = random.choice( self.class_to_image_id[rand_class] )
                rand_index = self.id_to_index[rand_image_id]
                _, _, _, rand_image_name = self.images_info[rand_index] 
                retrival_image = Image.open('{}/{}'.format(self.images_path, rand_image_name))
                if self.data_transform is not None:
                    retrival_image = self.data_transform(retrival_image)
                retrival_image = self.pil_to_np(retrival_image)
                retrival_image = torch.tensor(retrival_image, dtype=torch.float32)
                retrival_image = torch.unsqueeze(retrival_image, dim=0)
                retrival_label = torch.unsqueeze(torch.tensor([label]), dim=0)

                if retrival_images is None:
                    retrival_images = retrival_image
                    retrival_labels = retrival_label
                else:
                    retrival_images = torch.cat((retrival_images, retrival_image), dim=0)
                    retrival_labels = torch.cat((retrival_labels, retrival_label), dim=0)

                classes.remove( rand_class )

        return image, pos_image, neg_image, retrival_images, retrival_labels, super_class_id 

    def __getitem__(self, ix):
        if self.eval_mode:
            return self.__eval_getitem__(ix)
        return self.__train_getitem__(ix)

    def __len__(self):
        return len(self.images_info)