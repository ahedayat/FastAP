import torch
import numpy as np
import matplotlib.pyplot as plt

def _plot_losses(reprot_path, data_mode, start_epoch, end_epoch):
    assert data_mode in ['train', 'val', 'test'], 'data_mode must be "train" or "val" or "test".'
    losses_path = '{}/{}_losses'.format(reprot_path, data_mode)
    losses = list()
    avg_losses = list()
    epochs = list()
    for epoch in range(start_epoch, end_epoch+1):
        if data_mode=='test' and epoch!=end_epoch:
            continue
        epoch_loss = torch.load('{}/{}_losses_epoch_{}.pt'.format(losses_path, data_mode, epoch))
        print( torch.mean(epoch_loss))
        losses.append( epoch_loss )
        epochs.append(epoch)
        avg_losses.append( torch.mean(epoch_loss) )
    
    plt.plot(epochs, avg_losses)
    plt.xticks(range(0,10), [ix+1 for ix in range(start_epoch, end_epoch+1)])
    plt.title('{} loss'.format(data_mode))
    plt.grid()
    plt.show()

def _main():
    analysis_num = 5
    reports_path = './reports/{}'.format(analysis_num)
    start_epoch, end_epoch = 0,9

    # _plot_losses(reports_path, 'train', start_epoch, end_epoch )
    _plot_losses(reports_path, 'val', start_epoch, end_epoch )
    _plot_losses(reports_path, 'test', start_epoch, end_epoch )

if __name__ == "__main__":
    _main()