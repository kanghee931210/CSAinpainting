import time
from util.data_load import Data_load
from models.models import create_model
import torch
import os
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import argparse
import param
from dataloader import Id

parser = argparse.ArgumentParser()

parser.add_argument("__path", required= False, help = 'data input path')
args = parser.parse_args()

opt = param.Opion()

transform_mask = transforms.Compose(
    [transforms.Resize((opt.fineSize,opt.fineSize)),
     transforms.ToTensor(),
    ])
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.Resize((opt.fineSize,opt.fineSize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

dataset_train = Id(args.path, opt)
iterator_train = (data.DataLoader(dataset_train, batch_size=opt.batchSize,shuffle=True))

print(len(dataset_train))

model = create_model(opt)
total_steps = 0
save_dir = './measure/save'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

iter_start_time = time.time()
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

    epoch_start_time = time.time()
    epoch_iter = 0

    #     image, mask, gt = [x.cuda() for x in next(iterator_train)]
    for image, mask in (iterator_train):
        image = image.cuda()
        mask = mask.cuda()
        mask = mask[0][0]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 1)
        mask = mask.byte()

        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(image, mask)  # it not only sets the input data with mask, but also sets the latent mask.
        model.set_gt_latent()
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            real_A, real_B, fake_B = model.get_current_visuals()
            # real_A=input, real_B=ground truth fake_b=output
            pic = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0
            torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (
                save_dir, epoch, total_steps + 1, len(dataset_train)), nrow=2)
        if total_steps % 1 == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            print(errors)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()