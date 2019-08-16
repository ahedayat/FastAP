import torch

import numpy as np
import torchvision.transforms as torch_transforms
import dataloaders.standford_online_products as sop
import nets.resnet as resnet
import matplotlib.pyplot as plt
import utils as utility

def plot_representation(model, data_loader, device, gpu, num_workers, n_components):
    outputs_rep = resnet.represent(model, data_loader, device=device, gpu=gpu, num_workers=num_workers, n_components=n_components)
    # for label in outputs_rep.keys():
    #     label_outputs_rep = outputs_rep[label]
    #     labels = [label] * len(label_outputs_rep)
    #     plt.plot(labels, label_outputs_rep)
    # plt.show()


def _main(args):

    #### Preparing Train Dataset ####
    train_data_root = './datasets/standford_online_products/train'
    train_data_transform = torch_transforms.Resize((225,225))
    train_num_retrieval_per_class = 10
    train_pca_n_components=2
    train_pos_neighbor, train_neg_neighbor = (False, False)
    train_dataloader = sop.loader(  train_data_root, \
                                    data_transform=train_data_transform, \
                                    eval_mode=True, \
                                    eval_num_retrieval=train_num_retrieval_per_class, \
                                    neg_neighbor=train_neg_neighbor, \
                                    pos_neighbor=train_pos_neighbor
                                )
    
    #### Preparing Test Dataset ####
    test_data_root = './datasets/standford_online_products/test'
    test_data_transform = torch_transforms.Resize((225,225))
    test_num_retrieval_per_class = 10
    test_pca_n_components=2
    test_pos_neighbor, test_neg_neighbor = (False, False)
    test_dataloader = sop.loader(   test_data_root, \
                                    data_transform=test_data_transform, \
                                    eval_mode=True, \
                                    eval_num_retrieval=test_num_retrieval_per_class, \
                                    neg_neighbor=test_neg_neighbor, \
                                    pos_neighbor=test_pos_neighbor
                                )

    #### Preparing Validation Dataset ####
    val_data_root = './datasets/standford_online_products/val'
    val_num_retrieval_per_class = test_num_retrieval_per_class
    val_data_transform = torch_transforms.Resize((225,225))
    val_pca_n_components=2
    val_pos_neighbor, val_neg_neighbor = (False, False)
    val_dataloader = sop.loader(val_data_root, \
                                data_transform=val_data_transform, \
                                eval_mode=True, \
                                eval_num_retrieval=val_num_retrieval_per_class,\
                                neg_neighbor=val_neg_neighbor, \
                                pos_neighbor=val_pos_neighbor
                                )

    #### Preparing Pytorch ####
    device = args.device
    assert (device in ['cpu', 'multi']) or ( len(device.split(':'))==2 and device.split(':')[0]=='cuda' and int(device.split(':')[1]) < torch.cuda.device_count() ), 'Uknown device: {}'.format( device )
    torch.manual_seed(0)
    if args.device!='multi':
        device = torch.device(args.device)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #### Training Parameters ####
    start_epoch, num_epoch = (args.start_epoch, args.epochs)
    num_workers = args.num_workers
    check_counter = 10

    #### Reports Address ####
    reports_root = './reports'
    analysis_num = args.analysis
    reports_path = '{}/{}'.format( reports_root, analysis_num)
    loading_model_path = '{}/models'.format( reports_path )

    #### Constructing Model ####
    pretrained = args.pretrained
    num_classes = val_dataloader.num_classes()
    
    #### Constructing Model ####
    pretrained = args.pretrained
    num_classes = val_dataloader.num_classes()
    
    model = None
    if args.resnet_type=='resnet18':
        model = resnet.resnet18(pretrained=pretrained, num_classes=num_classes)
    elif args.resnet_type=='resnet34':
        model = resnet.resnet34(pretrained=pretrained, num_classes=num_classes)
    elif args.resnet_type=='resnet50':
        model = resnet.resnet50(pretrained=pretrained, num_classes=num_classes)
    elif args.resnet_type=='resnet101':
        model = resnet.resnet101(pretrained=pretrained, num_classes=num_classes)
    elif args.resnet_type=='resnet152':
        model = resnet.resnet152(pretrained=pretrained, num_classes=num_classes)
    elif args.resnet_type=='resnext50_32x4d':
        model = resnet.resnext50_32x4d(pretrained=pretrained, num_classes=num_classes)
    # elif args.resnet_type=='resnext101_32x8d':
    #     model = resnet.resnext101_32x8d(pretrained=pretrained, num_classes=num_classes)

    model, optimizer = resnet.load( loading_model_path,
                                    'resnet_epoch_{}'.format( start_epoch ),
                                    model
                        )

    if args.gpu and torch.cuda.is_available():
        if device=='multi':
            model = nn.DataParallel(model)
        else:
            model = model.cuda(device=device)

    plot_representation(model, train_dataloader, device, args.gpu and torch.cuda.is_available(), num_workers, train_pca_n_components)
    # plot_representation(model, val_dataloader, device, args.gpu and torch.cuda.is_available(), num_workers, val_pca_n_components)
    # plot_representation(model, test_dataloader, device, args.gpu and torch.cuda.is_available(), num_workers, test_pca_n_components)

if __name__ == '__main__':
    args = utility.get_args()
    _main(args)
    