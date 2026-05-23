import sys
import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(args, model, optimizer, data_loader, device, epoch, loss_weight):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0

    pbar = tqdm(data_loader, desc=f"  train", leave=False, ncols=60)
    for step, (images, images_labels, audio, audio_labels) in enumerate(pbar):
        images = images.to(device)
        audio = audio.to(device)
        images_labels = images_labels.to(device)

        optimizer.zero_grad()
        out, feat, cls_v, cls_a = model(images, audio)
        loss = loss_fn(out, images_labels)
        loss.backward()
        optimizer.step()

        accu_num += out.argmax(dim=1).eq(images_labels).sum()
        accu_loss += loss.detach()
        sample_num += images.shape[0]

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training", loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(args, model, data_loader, device, epoch, loss_weight):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0

    pbar = tqdm(data_loader, desc=f"  val  ", leave=False, ncols=60)
    for step, (images, images_labels, audio, audio_labels) in enumerate(pbar):
        images = images.to(device)
        audio = audio.to(device)
        images_labels = images_labels.to(device)

        out, feat, cls_v, cls_a = model(images, audio)
        loss = loss_fn(out, images_labels)
        accu_num += out.argmax(dim=1).eq(images_labels).sum()
        accu_loss += loss
        sample_num += images.shape[0]

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
