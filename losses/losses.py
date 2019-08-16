import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, target, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

class FastAP(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, bin_size, start_bin=0., end_bin=4.):
        super(FastAP, self).__init__()
        self.bin_size = bin_size
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.bin_length = (self.end_bin - self.start_bin)/self.bin_size
        self.bin_length = torch.tensor(self.bin_length, dtype=torch.float32) 
        self.bins = np.arange(self.start_bin+self.bin_length/2, self.end_bin, self.bin_length)
        self.bins = torch.tensor(self.bins, dtype=torch.float32)
        self.epsilon = 1e-7
        # self.delta_const = 

    def query_bin_counter(self, query, retrievals):
        query_distances = torch.sum( (retrievals-query)**2, dim=1 )
        query_distances = torch.unsqueeze(query_distances, dim=1).repeat((1, self.bins.size()[0]))
        
        query_bins = self.bins.clone().repeat( (retrievals.size()[0], 1) )
        query_bin_length = self.bin_length.clone()
        if query_distances.device!='cpu':
            device = query_distances.device
            query_bins = query_bins.cuda(device=device)
            query_bin_length = query_bin_length.cuda(device=device)
            
        delta = 1- torch.abs(query_bins-query_distances) / query_bin_length
        delta[ delta<0 ] = 0.
        h = torch.sum(delta, dim=0)
        H = torch.cumsum(h, 0)
        return h, H




    def forward(self, output, pos_output, neg_output, Y):
        labels = torch.argmax(Y,dim=1)
        neighbors = labels.repeat(Y.size()[0],1)
        labels = torch.unsqueeze(labels, dim=1)
        neighbors = neighbors==labels
        
        for ix in range(output.size()[0]):
            query = output[ix, :]
            retrievals = torch.cat( (output[:ix, :], output[ix+1:, :]) )
            query_neighbors = neighbors[ix, :]
            query_neighbors = torch.cat( (query_neighbors[:ix], query_neighbors[ix+1:]) )
            pos_retrievals = retrievals[query_neighbors]

            h, H = self.query_bin_counter(query, retrievals)
            h_plus, H_plus = self.query_bin_counter(query, pos_retrievals)
            query_n_plus = torch.sum(neighbors[ix, :], dim=0)

            loss = (h_plus*H_plus) / (H+self.epsilon)
            loss = torch.sum(loss, dim=0)
            loss = loss / query_n_plus

            return loss