import torch
import warnings

import torch.nn as nn
import torch.optim as optim
import utils as utility

import torchvision.transforms as torch_transforms
import dataloaders.standford_online_products as sop
import nets.resnet as resnet
import losses as losses

def _main(args):
    warnings.filterwarnings("ignore") 

    #### Preparing Pytorch ####
    device = args.device
    assert (device in ['cpu', 'multi']) or ( len(device.split(':'))==2 and device.split(':')[0]=='cuda' and int(device.split(':')[1]) < torch.cuda.device_count() ), 'Uknown device: {}'.format( device )
    torch.manual_seed(0)
    if args.device!='multi':
        device = torch.device(args.device)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #### Constructing Criterion ####
    # criterion = losses.TripletLoss(1.)
    bin_size = 10
    start_bin, end_bin = (0., 4.)
    criterion = losses.FastAP(bin_size, start_bin, end_bin)
    pos_neighbor, neg_neighbor = (True, True) if type(criterion) in [losses.TripletLoss] else (False, False)

    #### Preparing Dataset ####
    train_data_root = './datasets/standford_online_products/train'
    train_data_transform = torch_transforms.Resize((225,225))
    dataloader = sop.loader( train_data_root, data_transform=train_data_transform, neg_neighbor=neg_neighbor, pos_neighbor=pos_neighbor)


    #### Constructing Model ####
    pretrained = args.pretrained and False
    num_classes = 512
    
    model = None
    if args.resnet_type=='resnet18':
        model = resnet.resnet18(pretrained=pretrained)
    elif args.resnet_type=='resnet34':
        model = resnet.resnet34(pretrained=pretrained)
    elif args.resnet_type=='resnet50':
        model = resnet.resnet50(pretrained=pretrained)
    elif args.resnet_type=='resnet101':
        model = resnet.resnet101(pretrained=pretrained)
    elif args.resnet_type=='resnet152':
        model = resnet.resnet152(pretrained=pretrained)
    # elif args.resnet_type=='resnext50_32x4d':
    #     model = resnet.resnet18(pretrained=pretrained, num_classes=num_classes)
    # elif args.resnet_type=='resnext101_32x8d':
    #     model = resnet.resnext101_32x8d(pretrained=pretrained, num_classes=num_classes)
    model.fc = nn.Linear(512 * 1, num_classes)

    if args.start_epoch!=0:
        model, optim

    if args.gpu and torch.cuda.is_available():
        if device=='multi':
            model = nn.DataParallel(model)
        else:
            model = model.cuda(device=device)


    #### Training Parameters ####
    start_epoch, num_epoch = (args.start_epoch, args.epochs)
    batch_size = args.batch_size
    num_workers = args.num_workers
    check_counter = 10

    #### Reports Address ####
    reports_root = './reports'
    analysis_num = args.analysis
    reports_path = '{}/{}'.format( reports_root, analysis_num)
    saving_model_path = '{}/models'.format( reports_path )
    model_name = 'resnet_{}_{}'.format( start_epoch, start_epoch+num_epoch)

    utility.mkdir( reports_path, 'models', forced_remove=False)

    #### Constructing Optimizer ####
    optimizer = None

    if args.optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if start_epoch!=0:
        print('Loading model...')
        model, optimizer = resnet.load( saving_model_path,
                                        'resnet_epoch_{}'.format( start_epoch-1 ),
                                        model,
                                        optimizer=optimizer
                            )
 
    #### Training Model ####
    resnet.train( 
                resnet=model,
                train_data=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                num_epoch=num_epoch,
                start_epoch=start_epoch,
                batch_size=batch_size,
                num_workers=num_workers,
                check_counter=check_counter,
                gpu=args.gpu and torch.cuda.is_available(),
                report_path=reports_path,
                saving_model_every_epoch=True,
                device=device
                )
    
    #### Saving Model ####
    resnet.save(saving_model_path, model_name, model, optimizer=optimizer)

if __name__ == "__main__":
    args = utility.get_args()
    _main(args)
