import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

# from sklearn.metrics import confusion_matrix
from torchmetrics.functional import confusion_matrix

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import SGD, Adam
from models.resnet import resnet18
from lightning.data_module.cifar10_data_modules import ImbalancedMNISTDataModule


def accNaccPerCls(pred, label, num_class):
    cm = torch.nan_to_num(confusion_matrix(pred, label, num_classes=num_class))
    acc = torch.nan_to_num(cm.trace() / cm.sum())
    acc_per_cls = torch.nan_to_num(cm.diagonal() / cm.sum(0))

    return cm, acc, acc_per_cls

class Resnet_classifier(pl.LightningModule):
    def __init__(self,
                 num_class,
                 learning_rate,
                 momentum,
                 weight_decay,
                 nesterov,
                 warmup_epoch,
                 step1,
                 step2,
                 gamma):
        super(Resnet_classifier, self).__init__()
        self.save_hyperparameters()

        self.model = resnet18(num_classes=10)
        self.model.fc = nn.Linear(in_features=512, out_features=10)
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
                   "val_acc": acc,
                   "hp_metric": acc}
        metrics.update({ f"cls_{idx}" : acc for idx, acc in enumerate(acc_per_cls)})

        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        # val_acc = val_step_outputs['val_acc'].cpu()
        # val_loss = val_step_outputs['val_loss'].cpu()
        #
        # self.log('validation_acc', val_acc, prog_bar=True)
        # self.log('validation_loss', val_loss, prog_bar=True)

        self.log_dict(val_step_outputs,)


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
        parser.add_argument('--momentum', type=float, default=0.0001)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--nesterov', type=bool, default=True)
        parser.add_argument('--warmup_epoch', type=int, default=5)
        parser.add_argument('--step1', type=int, default=160)
        parser.add_argument('--step2', type=int, default=180)
        parser.add_argument('--gamma', type=float, default=0.1)
        return parent_parser



def cli_main():
    pl.seed_everything(1234)  # 다른 환경에서도 동일한 성능을 보장하기 위한 random seed 초기화

    parser = ArgumentParser()
    parser.add_argument("--num_class", default=10, type=int)
    parser.add_argument("--image_size", default=32, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--imb_factor", default=0.01, type=float)
    parser.add_argument("--balanced", default=False, type=bool)
    parser.add_argument("--retain_epoch_size", default=False, type=bool)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    parser = Resnet_classifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args('')
    dm = ImbalancedMNISTDataModule.from_argparse_args(args)

    model = Resnet_classifier(args.num_class,
                              args.learning_rate,
                            args.momentum,
                            args.weight_decay,
                            args.nesterov,
                            args.warmup_epoch,
                            args.step1,
                            args.step2,
                            args.gamma)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:d}_{val_loss:.4}_{val_acc:.4}",
        verbose=True,
        # save_last=True,
        save_top_k=1,
        monitor='val_acc',
        mode='max',
    )
    logger = TensorBoardLogger(save_dir="tb_logs",
                               name="resnet18_original_cifar10_0.01",
                               default_hp_metric=False
                               )

    trainer = pl.Trainer(max_epochs=200,
                         # callbacks=[EarlyStopping(monitor='val_loss')],
                         callbacks=[checkpoint_callback],
                         strategy=DDPStrategy(find_unused_parameters=False),
                         accelerator='gpu',
                         gpus=-1,
                         logger=logger
                         )

    trainer.fit(model, datamodule=dm)

    result = trainer.test(model, dataloaders=dm.test_dataloader())

    print(result)


if __name__ == '__main__':
    cli_main()





