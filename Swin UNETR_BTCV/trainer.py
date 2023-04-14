
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

checkpoint = config['DEFAULT']['checkpoint']
logdir = config['DEFAULT']['logdir']
pretrained_dir = config['DEFAULT']['pretrained_dir']
data_dir = config['DEFAULT']['data_dir']
json_list = config['DEFAULT']['json_list']
pretrained_model_name = config['DEFAULT']['pretrained_model_name']
save_checkpoint = config.getboolean('DEFAULT', 'save_checkpoint')
max_epochs = config.getint('DEFAULT', 'max_epochs')
batch_size = config.getint('DEFAULT', 'batch_size')
sw_batch_size = config.getint('DEFAULT', 'sw_batch_size')
optim_lr = config.getfloat('DEFAULT', 'optim_lr')
optim_name = config['DEFAULT']['optim_name']
reg_weight = config.getfloat('DEFAULT', 'reg_weight')
momentum = config.getfloat('DEFAULT', 'momentum')
noamp = config.getboolean('DEFAULT', 'noamp')
val_every = config.getint('DEFAULT', 'val_every')
distributed = config.getboolean('DEFAULT', 'distributed')
world_size = config.getint('DEFAULT', 'world_size')
rank = config.getint('DEFAULT', 'rank')
dist_url = config['DEFAULT']['dist-url']
dist_backend = config['DEFAULT']['dist-backend']
norm_name = config['DEFAULT']['norm_name']
workers = config.getint('DEFAULT', 'workers')
feature_size = config.getint('DEFAULT', 'feature_size')
in_channels = config.getint('DEFAULT', 'in_channels')
out_channels = config.getint('DEFAULT', 'out_channels')
use_normal_dataset = config.getboolean('DEFAULT', 'use_normal_dataset')
a_min = config.getfloat('DEFAULT', 'a_min')
a_max = config.getfloat('DEFAULT', 'a_max')
b_min = config.getfloat('DEFAULT', 'b_min')
b_max = config.getfloat('DEFAULT', 'b_max')
space_x = config.getfloat('DEFAULT', 'space_x')
space_y = config.getfloat('DEFAULT', 'space_y')
space_z = config.getfloat('DEFAULT', 'space_z')
roi_x = config.getint('DEFAULT', 'roi_x')
roi_y = config.getint('DEFAULT', 'roi_y')
roi_z = config.getint('DEFAULT', 'roi_z')
dropout_rate = config.getfloat('DEFAULT', 'dropout_rate')
dropout_path_rate = config.getfloat('DEFAULT', 'dropout_path_rate')
RandFlipd_prob = config.getfloat('DEFAULT', 'RandFlipd_prob')
RandRotate90d_prob = config.getfloat('DEFAULT', 'RandRotate90d_prob')
RandScaleIntensityd_prob = config.getfloat('DEFAULT', 'RandScaleIntensityd_prob')
RandShiftIntensityd_prob = config.getfloat('DEFAULT', 'RandShiftIntensityd_prob')
infer_overlap = config.getfloat('DEFAULT', 'infer_overlap')
lrschedule = config['DEFAULT']['lrschedule']
warmup_epochs = config.getint('DEFAULT', 'warmup_epochs')
resume_ckpt = config.getboolean('DEFAULT', 'resume_ckpt')
smooth_dr = config.getfloat('DEFAULT', 'smooth_dr')
smooth_nr = config.getfloat('DEFAULT', 'smooth_nr')

use_ssl_pretrained = config.getboolean('DEFAULT', 'use_ssl_pretrained')
use_checkpoint = config.getboolean('DEFAULT', 'use_checkpoint')
spatial_dims =config.getint('DEFAULT', 'spatial_dims')
squared_dice =config.getboolean('DEFAULT', 'squared_dice')

amp = config['DEFAULT'].getboolean('noamp')

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(rank), target.cuda(rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=amp):
            logits = model(data)
            loss = loss_func(logits, target)
            #
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=batch_size * world_size
            )
        else:
            run_loss.update(loss.item(), n=batch_size)
        if rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func,  model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(rank), target.cuda(rank)
            with autocast(enabled=amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(rank)

            if distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return run_acc.avg


def save_checkpoint(model, epoch,  filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if logdir is not None and rank == 0:
        writer = SummaryWriter(log_dir=logdir)
        if rank == 0:
            print("Writing Tensorboard logs to ", logdir)
    scaler = None
    if amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, max_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, 
        )
        if rank == 0:
            print(
                "Final training  {}/{}".format(epoch, max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % val_every == 0:
            if distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if rank == 0 and logdir is not None and save_checkpoint:
                        save_checkpoint(
                            model, epoch, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if rank == 0 and logdir is not None and save_checkpoint:
                save_checkpoint(model, epoch, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(logdir, "model_final.pt"), os.path.join(logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
