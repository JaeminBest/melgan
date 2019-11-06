import os
import time
import logging
import argparse
import shutil

from utils.train import train, data_check
from utils.hparams import HParam
from utils.writer import MyWriter
from datasets.dataloader import create_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/default.yaml' if not os.environ.get('CUSTOM','') else 'config/custom.yaml',
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())
    
    pt_dir = os.path.join(hp.log.chkpt_dir, args.name)
    log_dir = os.path.join(hp.log.log_dir, args.name)
    if os.path.isfile(pt_dir):
        os.system("cp -R {} {}".format(pt_dir,'/app/backup/ckpt'))
        shutil.rmtree(pt_dir)
    if os.path.isfile(log_dir):
        os.system("cp -R {} {}".format(log_dir,'/app/backup/log'))
        shutil.rmtree(log_dir)
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    writer = MyWriter(hp, log_dir)

    assert hp.audio.hop_length == 256, \
        'hp.audio.hop_length must be equal to 256, got %d' % hp.audio.hop_length
    assert hp.data.train != '' and hp.data.validation != '', \
        'hp.data.train and hp.data.validation can\'t be empty: please fix %s' % args.config

    # check invalid data
    if not args.checkpoint_path:
        checkloader = create_dataloader(hp, args, -1)
        data_check(args,checkloader,hp,hp_str) # take 3 hours
        os.system("python data_preparation.py --folder /data -s -d")
    
    trainloader = create_dataloader(hp, args, 1)
    valloader = create_dataloader(hp, args, 0)

    train(args, pt_dir, args.checkpoint_path, trainloader, valloader, writer, logger, hp, hp_str)
