from torch.utils.tensorboard import SummaryWriter
import os


log_dir = './tb_logs/ACGAN/im_mnist/original/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb = SummaryWriter(log_dir=log_dir)


for i in range(5):
    tb.add_hparams({'lr': 0.1*i, 'bsize': i},
                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})