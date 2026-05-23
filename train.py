import os
import math
import argparse
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data_processing.my_dataset import MyDataSet
from model.AVoiD import AVoiD_mm
from utils.utils import train_one_epoch, evaluate
from data_processing.preprocess import read_split_data


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    random.seed(0)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, train_audio_path, train_audio_label, \
    val_images_path, val_images_label, val_audio_path, val_audio_label = \
        read_split_data(args.video_data_path, args.audio_data_path)

    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val":   transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label,
                              audio_path=train_audio_path,  audio_class=train_audio_label,
                              transform=data_transform["train"])
    val_dataset   = MyDataSet(images_path=val_images_path,  images_class=val_images_label,
                              audio_path=val_audio_path,    audio_class=val_audio_label,
                              transform=data_transform["val"])

    nw = min(os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8)
    print(f"Using {nw} dataloader workers")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True,  pin_memory=True,
                                               num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size,
                                               shuffle=False, pin_memory=True,
                                               num_workers=nw, collate_fn=val_dataset.collate_fn)

    model = AVoiD_mm(args, num_classes=args.num_classes, has_logits=False).to(device)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print(f"  training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_weight = torch.nn.Parameter(torch.ones(1)).to(device)

    # Resume must come AFTER optimizer/scheduler are created
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume != "":
        assert os.path.exists(args.resume), f"checkpoint not found: {args.resume}"
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}  (best val_acc {best_val_acc:.4f})")
    elif args.weights != "":
        assert os.path.exists(args.weights), f"weights not found: {args.weights}"
        print(model.load_state_dict(torch.load(args.weights, map_location=device), strict=False))

    print(f"\nTraining {args.num_classes}-class model for {args.epochs} epochs on {device}\n"
          f"{'epoch':>6}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>8}  {'val_acc':>7}")
    print("-" * 55)

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(args, model=model, optimizer=optimizer,
                                                data_loader=train_loader, device=device,
                                                epoch=epoch, loss_weight=loss_weight)
        scheduler.step()
        val_loss, val_acc = evaluate(args, model=model, data_loader=val_loader,
                                     device=device, epoch=epoch, loss_weight=loss_weight)

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  {val_loss:>8.4f}  {val_acc:>7.4f}")

        for tag, val in zip(["train_loss", "train_acc", "val_loss", "val_acc", "lr"],
                            [train_loss, train_acc, val_loss, val_acc,
                             optimizer.param_groups[0]["lr"]]):
            tb_writer.add_scalar(tag, val, epoch)
        tb_writer.flush()

        # Save best + periodic checkpoint to ckpt_dir (survives Colab resets if on Drive)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best.pth"))

        if epoch % args.save_every == 0 or val_acc >= best_val_acc:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
            }, os.path.join(args.ckpt_dir, "checkpoint.pth"))

    tb_writer.close()
    print(f"\nDone. Best val_acc: {best_val_acc:.4f}  Weights saved to: {args.ckpt_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes',     type=int,   default=4)
    parser.add_argument('--epochs',          type=int,   default=2000)
    parser.add_argument('--batch-size',      type=int,   default=32)
    parser.add_argument('--lr',              type=float, default=0.01)
    parser.add_argument('--lrf',             type=float, default=0.01)
    parser.add_argument('--video_data-path', type=str,   default="./video")
    parser.add_argument('--audio_data-path', type=str,   default="./audio")
    parser.add_argument('--weights',         type=str,   default='')
    parser.add_argument('--resume',          type=str,   default='',
                        help='path to checkpoint.pth to resume from')
    parser.add_argument('--ckpt-dir',        type=str,   default='./weights',
                        help='directory to save checkpoints (use Drive path on Colab)')
    parser.add_argument('--save-every',      type=int,   default=10,
                        help='save checkpoint every N epochs')
    parser.add_argument('--freeze-layers',   type=bool,  default=True)
    parser.add_argument('--device',          default='cuda:0')

    opt = parser.parse_args()
    main(opt)
