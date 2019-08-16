import gc
import os
import torch
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils as utility

from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import decomposition

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def resnet_save( file_path, file_name, resnet, optimizer=None ):
    state_dict = {
        'net_arch' : 'resnet',
        'model' : resnet.state_dict(),
    }
    if optimizer is not None:
        state_dict[ 'optimizer' ] = optimizer.state_dict()

    torch.save( state_dict, '{}/{}.pth'.format(file_path,file_name) )

def resnet_load(file_path, file_name, model, optimizer=None):
    check_points = torch.load('{}/{}.pth'.format(file_path,file_name))
    keys = check_points.keys()


    assert ('net_arch' in keys) and ('model' in keys), 'Cannot read this file in address : {}/{}.pth'.format(file_path,file_name)
    assert check_points['net_arch']=='resnet', 'This file model architecture is not \'resnet\''
    model.load_state_dict( check_points['model'] )
    if optimizer is not None:
        optimizer.load_state_dict(check_points['optimizer'])
    return model, optimizer

def resnet_accuracy(model, output,  retrival_images, retrival_labels, Y, num_classes=12, k=5):
    matched_batch = 0
    batch_size = output.size()[0]
    for b in range(batch_size):
        print('acc->b={}'.format(b))
        print('output[b,:].shape: {}'.format(output[b,:].unsqueeze(dim=0).size()))
        retrival_outputs = model(retrival_images[b,:,:,:,:])
        clf = NearestCentroid()
        clf.fit(retrival_outputs.detach().cpu().numpy(), retrival_labels[b,:].detach().cpu().numpy())
        predicted_label = clf.predict(output[b,:].unsqueeze(dim=0).detach().cpu().numpy())
        matched_batch += 1 if predicted_label==Y[b].detach().cpu().numpy() else 0

    return matched_batch / batch_size

# def resnet_save_distance_vecs( 
#                                 resnet,
#                                 eval_data,
#                                 criterion,
#                                 report_path,
#                                 epoch,
#                                 device,
#                                 batch_size=2,
#                                 num_workers=2,
#                                 check_counter=4,
#                                 gpu=False,
#                                 eval_mode='test'):

def resnet_representation(  resnet,
                            data_loader,
                            device='cpu',
                            gpu=False,
                            num_workers=1,
                            n_components=2):

    data_loader.eval_mode = True
    torch_data_loader = DataLoader( data_loader,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=gpu and torch.cuda.is_available(),
                                    num_workers=num_workers)

    pca = decomposition.PCA(n_components=n_components)

    outputs_rep = dict()
    for ix,(X, _, _, _, _, Y) in enumerate( torch_data_loader ):
        X = X.permute(0,3,1,2)
        X = V(X)
        Y = torch.squeeze(Y, dim=0)
        label = torch.argmax(Y).item()
        if gpu:
            X = X.cuda(device=device)
            if device=='multi':
                X = nn.DataParallel(X)
        output = resnet(X)
        output = torch.squeeze(output, dim=0).detach().cpu().numpy()
        print(output.shape)
        if ix==0:
            pca.fit(output)
        output_rep = pca.transform(output)

        if label not in outputs_rep.keys():
            outputs_rep[label] = list()
        outputs_rep[label].append( output_rep )
    
        print('generating representation: {}/{}'.format(ix, len(data_loader)))

        del X, Y, output
        torch.cuda.empty_cache()
        gc.collect()

    return outputs_rep



def resnet_train(
                 resnet,
                 train_data,
                 optimizer,
                 criterion,
                 report_path,
                 device,
                 num_epoch=1,
                 start_epoch=0, 
                 batch_size=2,
                 num_workers = 1,
                 check_counter=20,
                 gpu=False,
                 saving_model_every_epoch=False):

    utility.mkdir( report_path, 'train_batches_size' )
    utility.mkdir( report_path, 'train_losses' )
    utility.mkdir( report_path, 'train_accuracies' )
    utility.mkdir( report_path, 'models' )

    for epoch in range( start_epoch, start_epoch+num_epoch ):

        data_loader = DataLoader( train_data,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory= gpu and torch.cuda.is_available(),
                                num_workers=num_workers)
        
        batches_size = list()
        losses = list()
        accuracies = list()

        curr_loss = 0
        for ix,(X, X_pos, X_neg, Y) in enumerate( data_loader ):
            X = X.permute(0,3,1,2)
            X, Y = V(X), V(Y)
            if train_data.pos_neighbor:
                X_pos = X_pos.permute(0,3,1,2)
                X_pos = V(X_pos)
            if train_data.neg_neighbor:            
                X_neg = X_neg.permute(0,3,1,2)
                X_neg = V(X_neg)
            
            if gpu:
                X, Y = X.cuda(device=device), Y.cuda(device=device)
                if train_data.pos_neighbor:
                    X_pos = X_pos.cuda(device=device)
                if train_data.neg_neighbor:
                    X_neg = X_neg.cuda(device=device)
                if device=='multi':
                    X, Y = nn.DataParallel(X), nn.DataParallel(Y)
                    if train_data.pos_neighbor:
                        X_pos = nn.DataParallel(X_pos)
                    if train_data.neg_neighbor:
                        X_neg = nn.DataParallel(X_neg)

            output = resnet(X)
            pos_output, neg_output = None, None
            if train_data.pos_neighbor:
                pos_output = resnet( X_pos )
            if train_data.neg_neighbor:
                neg_output = resnet( X_neg )

            target = Y
            if isinstance( criterion, nn.CrossEntropyLoss ):
                target = torch.argmax( target, dim=1 )

            loss = criterion( output, pos_output, neg_output, target )
            # acc = resnet_accuracy(output, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prev_loss = curr_loss
            curr_loss = loss.item()

            batches_size.append( output.size()[0] )
            losses.append( curr_loss )
            # accuracies.append( acc )

            print( 'epoch=%d, batch=%d(x%d), prev_loss=%.5f, curr_loss=%.5f, delta=%.5f' % (
                                                                                            epoch,
                                                                                            ix,
                                                                                            output.size()[0],
                                                                                            prev_loss,
                                                                                            curr_loss,
                                                                                            curr_loss-prev_loss
                                                                                            ) )
            if ix%check_counter==(check_counter-1):
                # print()
                pass
            
            del X, Y, output       
            torch.cuda.empty_cache()
            gc.collect()

        torch.save( torch.tensor( batches_size ), 
                    '{}/train_batches_size/train_batches_size_epoch_{}.pt'.format(
                                                                                    report_path,
                                                                                    epoch
                                                                                 )
                  )
        torch.save( torch.tensor( losses ), 
                    '{}/train_losses/train_losses_epoch_{}.pt'.format(
                                                                        report_path,
                                                                        epoch
                                                                     )
                  )
        # torch.save( torch.tensor( accuracies ), 
        #             '{}/train_accuracies/train_accuracies_epoch_{}.pt'.format(
        #                                                                         report_path,
        #                                                                         epoch
        #                                                                      )
        #           )
        if saving_model_every_epoch:
            resnet_save( 
                        '{}/models'.format( report_path ),
                        'resnet_epoch_{}'.format( epoch ),
                        resnet,
                        optimizer=optimizer
                       )

def resnet_eval( 
                resnet,
                eval_data,
                criterion,
                report_path,
                epoch,
                device,
                batch_size=2,
                num_workers=2,
                check_counter=4,
                gpu=False,
                eval_mode='test'):

    assert eval_mode in ['val', 'test'], 'eval mode must be \'val\' or \'test\''

    utility.mkdir(report_path, '{}_batches_size'.format(eval_mode))
    utility.mkdir(report_path, '{}_losses'.format(eval_mode))
    utility.mkdir(report_path, '{}_accuracies'.format(eval_mode))
    eval_data.eval_mode=False

    data_loader = DataLoader(   eval_data,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=gpu and torch.cuda.is_available(),
                                num_workers=num_workers)
    resnet = resnet.eval()

    batches_size = list()
    losses = list()
    accuracies = list()

    curr_loss = 0
    for ix,(X, X_pos, X_neg, Y) in enumerate( data_loader ):
    # for ix,(X, X_pos, X_neg, Retrivals_images, Retrivals_labels, Y) in enumerate( data_loader ):
        X = X.permute(0,3,1,2)
        # Retrivals_images = Retrivals_images.permute(0,1,4,2,3)
        X, Y = V(X), V(Y)
        # Retrivals_images, Retrivals_labels =  V(Retrivals_images), V(Retrivals_labels)
        if eval_data.pos_neighbor:
            X_pos = X_pos.permute(0,3,1,2)
            X_pos = V(X_pos)
        if eval_data.neg_neighbor:            
            X_neg = X_neg.permute(0,3,1,2)
            X_neg = V(X_neg)
            
        if gpu:
            X, Y = X.cuda(device=device), Y.cuda(device=device)
            # Retrivals_images, Retrivals_labels = Retrivals_images.cuda(device=device), Retrivals_labels.cuda(device=device)
            if eval_data.pos_neighbor:
                X_pos = X_pos.cuda(device=device)
            if eval_data.neg_neighbor:
                X_neg = X_neg.cuda(device=device)
            if device=='multi':
                X, Y =  nn.DataParallel(X), nn.DataParallel(Y)
                # Retrivals_images, Retrivals_labels = nn.DataParallel(Retrivals_images), nn.DataParallel(Retrivals_labels)
                if eval_data.pos_neighbor:
                    X_pos = nn.DataParallel(X_pos)
                if eval_data.neg_neighbor:
                    X_neg = nn.DataParallel(X_neg)
        
        output = resnet(X)
        pos_output, neg_output = None, None
        # if eval_data.pos_neighbor:
        #     pos_output = resnet( X_pos )
        # if eval_data.neg_neighbor:
        #     neg_output = resnet( X_neg )

        target = Y
        if isinstance( criterion, nn.CrossEntropyLoss ):
            target = torch.argmax( target, dim=1 )

        loss = criterion( output, pos_output, neg_output, target )
        # acc = resnet_accuracy(resnet, output,  Retrivals_images, Retrivals_labels, Y, num_classes=eval_data.num_classes(), k=5)
        acc=0.

        prev_loss = curr_loss
        curr_loss = loss.item()

        batches_size.append( output.size()[0] )
        losses.append( curr_loss )
        # accuracies.append( acc )


        print( 'batch=%d(x%d), prev_loss=%.5f, curr_loss=%.5f, delta=%.5f, acc=%.3f%%' % (
                                                                                            ix,
                                                                                            output.size()[0],
                                                                                            prev_loss,
                                                                                            curr_loss,
                                                                                            curr_loss-prev_loss,
                                                                                            acc*100
                                                                                          ) )
        if ix%check_counter==(check_counter-1):
            # print()
            pass
            
        del X, Y, output, X_pos, X_neg
        torch.cuda.empty_cache()
        gc.collect()

        torch.save( torch.tensor( batches_size ), 
                    '{}/{}_batches_size/{}_batches_size_epoch_{}.pt'.format(
                                                                            report_path,
                                                                            eval_mode,
                                                                            eval_mode,
                                                                            epoch
                                                                           )
                  )
        torch.save( torch.tensor( losses ), 
                    '{}/{}_losses/{}_losses_epoch_{}.pt'.format(
                                                                report_path,
                                                                eval_mode,
                                                                eval_mode,
                                                                epoch
                                                               )
                  )
        # torch.save( torch.tensor( accuracies ), 
        #             '{}/{}_accuracies/{}_accuracies_epoch_{}.pt'.format(
        #                                                                 report_path,
        #                                                                 eval_mode,
        #                                                                 eval_mode,
        #                                                                 epoch
        #                                                                )
        #           )

