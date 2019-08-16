# image_id    class_id    super_class_id  super_class_name  path    filename
# 1           1           1               bicycle           bicycle_final     111085122871_0.JPG
import os
import shutil

def mkdir(dir_path, dir_name, forced_remove=False):
	new_dir = '{}/{}'.format(dir_path,dir_name)
	if forced_remove and os.path.isdir( new_dir ):
		shutil.rmtree( new_dir )
	if not os.path.isdir( new_dir ):
		os.makedirs( new_dir )

def touch(file_path, file_name, forced_remove=False):
	new_file = '{}/{}'.format(file_path,file_name)
	assert os.path.isdir( file_path ), ' \"{}\" does not exist.'.format(file_path)
	if forced_remove and os.path.isfile(new_file):
		os.remove(new_file)
	if not os.path.isfile(new_file):
		open(new_file, 'a').close()

def write_file(file_path, file_name, content, new_line=True, forced_remove_prev=False):
	touch(file_path, file_name, forced_remove=forced_remove_prev)
	with open('{}/{}'.format(file_path, file_name), 'a') as f:
		f.write('{}'.format(content))
		if new_line:
			f.write('\n')
		f.close()

def copy_file(src_path, src_file_name, dst_path, dst_file_name):
	shutil.copyfile('{}/{}'.format(src_path, src_file_name), '{}/{}'.format(dst_path,dst_file_name))  

def ls(dir_path):
	return os.listdir(dir_path)
    
def get_dicts(raw_data_path, info_file_name):
    superclass_ids = list()
    images = dict()
    images_path = dict()

    for ix, (line) in enumerate(open('{}/{}'.format(raw_data_path, info_file_name))):
        if ix==0:
            continue
        image_id, class_id, superclass_id, path = line.split()
        file_path, file_name = os.path.split(path)
        if superclass_id not in superclass_ids:
            superclass_ids.append( superclass_id )
            images[superclass_id] = dict()
            
        if class_id not in images[superclass_id].keys():
            images[superclass_id][class_id] = list()

        images[superclass_id][class_id].append(image_id)
        images_path[image_id] = ( ('{}/{}'.format(raw_data_path, file_path), file_name) )
    
    return images_path, images, superclass_ids

def preprocess(data_mode, saving_path, images, images_path, super_class_ids, saving_points=(0., 1.) ):
    assert data_mode in ['train', 'val', 'test'], 'data_mode must be one of ["train", "val", "test"]'
    assert saving_points[0]<= saving_points[1] and saving_points[0]>=0. and saving_points[1]<=1. , '**** Error: 0. <= saving_point[0] <= saving_point[1] <= 1'
    
    mkdir(saving_path, data_mode)
    mkdir('{}/{}'.format(saving_path, data_mode), 'images')
    touch('{}/{}'.format(saving_path, data_mode), 'images_info.txt')
    touch('{}/{}'.format(saving_path, data_mode), 'super_class_ids.txt')
    for super_class_id in super_class_ids:
        write_file('{}/{}'.format(saving_path, data_mode), 'super_class_ids.txt', '{}'.format(super_class_id))

    for ix, (super_class_id) in enumerate(images.keys()):
        start_offset = int( len(images[super_class_id].keys()) * saving_points[0] )
        end_offset = int( len(images[super_class_id].keys()) * saving_points[1] )
        num_super_class_data = end_offset - start_offset
        # print(start_offset, end_offset, num_super_class_data)

        for jx, (class_id) in enumerate(images[super_class_id].keys()):
            if jx < start_offset:
                continue
            if jx > end_offset:
                break

            class_images = images[super_class_id][class_id]
            
            for kx, (image_id) in enumerate(class_images):
                image_path, image_name = images_path[image_id]
                copy_file(image_path, image_name, '{}/{}/images'.format(saving_path, data_mode), image_name)
                image_info = '{} {} {} {}'.format(super_class_id, class_id, image_id, image_name)
                write_file( '{}/{}'.format(saving_path, data_mode), 'images_info.txt', image_info )
            print('%s: super class: %s( %d/%d(%.2f%%) ), class: %s( %d/%d(%.2f%%) )' %  (
                                                                                        data_mode,
                                                                                        super_class_id,
                                                                                        ix+1,
                                                                                        len(images.keys()),
                                                                                        (ix+1) / len(images.keys()) * 100,
                                                                                        class_id,
                                                                                        jx-start_offset+1,
                                                                                        num_super_class_data,
                                                                                        (jx-start_offset+1)/(num_super_class_data)*100
                                                                                    ), end='\r'
                )
    print()

def _main():
    train_raw_datapath, train_info_file_name, train_saving_path, train_saving_points = ( 'Stanford_Online_Products', 'Ebay_train.txt', '.', (0., 0.8) )
    val_saving_path, val_saving_points = ( '.', (0.8, 1.) )
    test_raw_datapath, test_info_file_name, test_saving_path, test_saving_points = ( 'Stanford_Online_Products', 'Ebay_test.txt', '.', (0., 1.) )

    train_images_path, train_images, train_super_class_ids = get_dicts(train_raw_datapath, train_info_file_name)
    test_images_path, test_images, test_super_class_ids = get_dicts(test_raw_datapath, test_info_file_name)
    val_images, val_images_path, val_super_class_ids = (train_images, train_images_path, train_super_class_ids)

    preprocess('train', train_saving_path, train_images, train_images_path, train_super_class_ids, train_saving_points)
    preprocess('val', val_saving_path, val_images, val_images_path, val_super_class_ids, val_saving_points)
    preprocess('test', test_saving_path, test_images, test_images_path, test_super_class_ids, test_saving_points)

if __name__ == "__main__":
    _main()

