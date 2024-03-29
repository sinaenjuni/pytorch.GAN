import pytorch_lightning as pl
from argparse import ArgumentParser

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from lightning.data_module.cifar10_data_modules import ImbalancedMNISTDataModule

from lightning.models.resnet import Resnet_classifier
from lightning.models.acgan import ACGAN

def cli_main():
    pl.seed_everything(1234)  # 다른 환경에서도 동일한 성능을 보장하기 위한 random seed 초기화

    parser = ArgumentParser()
    parser.add_argument("--augmentation", default=False, type=bool)
    parser.add_argument("--image_size", default=32, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--imb_factor", default=0.1, type=float)
    parser.add_argument("--balanced", default=False, type=bool)
    parser.add_argument("--retain_epoch_size", default=False, type=bool)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--target_model', type=int, default=99)


    parser = Resnet_classifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args('')
    dm = ImbalancedMNISTDataModule.from_argparse_args(args)

    # model = Resnet_classifier(args.model,
    #                           args.num_class,
    #                           args.sp,
    #                           args.learning_rate,
    #                             args.momentum,
    #                             args.weight_decay,
    #                             args.nesterov,
    #                             args.warmup_epoch,
    #                             args.step1,
    #                             args.step2,
    #                             args.gamma)

    # acgan = ACGAN.load_from_checkpoint(f"/home/sin/git/pytorch.GAN/src/lightning/models/tb_logs/acgan_cifar10_0.01/version_1/checkpoints/epoch={args.target_model}.ckpt")
    acgan = ACGAN.load_from_checkpoint(f"/home/sin/git/pytorch.GAN/src/lightning/models/tb_logs/acgan_cifar10_0.1_balancing/version_0/checkpoints/epoch={args.target_model}.ckpt")

    model = Resnet_classifier(**vars(args))

    model.model.conv1 = acgan.D.conv1
    model.model.layer1 = acgan.D.layer1
    model.model.layer2 = acgan.D.layer2
    model.model.layer3 = acgan.D.layer3
    model.model.layer4 = acgan.D.layer4
    model.model.fc = acgan.D.fc


    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename="{epoch:d}_{loss/val:.4}_{acc/val:.4}",
        verbose=True,
        # save_last=True,
        save_top_k=1,
        monitor='acc/val',
        mode='max',
    )
    logger = TensorBoardLogger(save_dir="tb_logs",
                               name=f"resnet18_pre_cifar10_{args.imb_factor}_bal",
                               default_hp_metric=False
                               )
    logger.log_hyperparams
    trainer = pl.Trainer(max_epochs=args.epoch,
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
