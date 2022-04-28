
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import confusion_matrix
from torch.optim import SGD, Adam
from models.resnet import resnet18, resnet34
from models.generator import Generator, linear, snlinear, deconv2d, sndeconv2d

import pytorch_lightning as pl


def accNaccPerCls(pred, label, num_class):
    cm = torch.nan_to_num(confusion_matrix(pred, label, num_classes=num_class))
    acc = torch.nan_to_num(cm.trace() / cm.sum())
    acc_per_cls = torch.nan_to_num(cm.diagonal() / cm.sum(0))

    return cm, acc, acc_per_cls


class FcNAdvModuel(nn.Module):
    def __init__(self, linear):
        super(FcNAdvModuel, self).__init__()
        self.fc = linear(in_features=512, out_features=10)
        self.adv = linear(in_features=512, out_features=1)

    def forward(self, x):
        return self.fc(x), self.adv(x)

class ACGAN(pl.LightningModule):
    def __init__(self,
                 model,
                 num_class,
                 bn,
                 sp,
                 learning_rate,
                 image_size,
                 image_channel,
                 std_channel,
                 latent_dim,
                 **kwargs):
        super(ACGAN, self).__init__()
        self.save_hyperparameters()

        G = Generator(linear=linear,
                      deconv=deconv2d,
                      image_size=image_size,
                      image_channel=image_channel,
                      std_channel=std_channel,
                      latent_dim=latent_dim,
                      bn=bn)

        if model == 'resnet18':
            self.D = resnet18(num_classes=num_class, sp=sp)
        elif model == 'resnet34':
            self.D = resnet34(num_classes=num_class, sp=sp)

        self.D.fc = FcNAdvModuel(linear=linear)

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image)
        loss = self.criterion(logit, label)
        pred = logit.argmax(-1)

        cm, acc, acc_per_cls = accNaccPerCls(pred, label, self.hparams.num_class)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        metrics = {"train_acc": acc}
        metrics.update({ f"cls_{idx}" : acc for idx, acc in enumerate(acc_per_cls)})
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image)
        loss = self.criterion(logit, label)

        pred = logit.argmax(-1)
        cm, acc, acc_per_cls = accNaccPerCls(pred, label, self.hparams.num_class)

        metrics = {"val_loss":loss,
                   "val_acc": acc}
        metrics.update({ f"cls_{idx}" : acc for idx, acc in enumerate(acc_per_cls)})

        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        # val_acc = val_step_outputs['val_acc'].cpu()
        # val_loss = val_step_outputs['val_loss'].cpu()
        #
        # self.log('validation_acc', val_acc, prog_bar=True)
        # self.log('validation_loss', val_loss, prog_bar=True)
        self.log_dict(val_step_outputs)

    def test_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image)
        loss = self.criterion(logit, label)

        pred = logit.argmax(-1)
        cm, acc, acc_per_cls = accNaccPerCls(pred, label, self.hparams.num_class)

        metrics = {"test_loss":loss,
                   "test_acc": acc}
        metrics.update({ f"cls_{idx}" : acc for idx, acc in enumerate(acc_per_cls)})
        self.log_dict(metrics)
        return metrics


    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        optimizer = SGD(self.parameters(),
                        momentum=self.hparams.momentum,
                        lr=self.hparams.learning_rate,
                        weight_decay=self.hparams.weight_decay,
                        nesterov=self.hparams.nesterov)

        def lr_lambda(epoch):
            if epoch >= self.hparams.step2:
                lr = self.hparams.gamma * self.hparams.gamma
            elif epoch >= self.hparams.step1:
                lr = self.hparams.gamma
            else:
                lr = 1
            """Warmup"""
            if epoch < self.hparams.warmup_epoch:
                lr = lr * float(1 + epoch) / self.hparams.warmup_epoch
            print("learning_rate", lr)
            return lr
        lr_scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        parser.add_argument("--model", default='resnet18', type=str)
        parser.add_argument("--num_class", default=10, type=int)
        parser.add_argument("--sp", default=False, type=bool)

        parser.add_argument('--momentum', type=float, default=0.0001)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--nesterov', type=bool, default=True)
        parser.add_argument('--warmup_epoch', type=int, default=5)
        parser.add_argument('--step1', type=int, default=160)
        parser.add_argument('--step2', type=int, default=180)
        parser.add_argument('--gamma', type=float, default=0.1)
        return parent_parser


if __name__ == "__main__":
    model = ACGAN()







