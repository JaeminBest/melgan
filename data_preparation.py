'''
generate final_train.txt
if custom(-c):
  download data from s3, fill folder(--folder) 
  and then generate final_train.txt and final_val.txt
else:
  load data from folder(--folder) and then generate final_train(val).txt
  
'''


import os
from pathlib import Path
from tqdm import tqdm
from librosa.core import load
import argparse
import json
import librosa
import subprocess
import multiprocessing as mp
import shutil
import torch
import numpy as np
import random

from ..credential import s3
from ..models import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--folder", type=Path, default='/app/data/train')
  parser.add_argument("--data-param-path", type=Path, default='/app/params.json')
  parser.add_argument("--train-file", type=Path, default='/app/data/final_train.txt')
  parser.add_argument("--val-file", type=Path, default='/app/data/final_val.txt')
  parser.add_argument("--tot-file", type=Path, default='/app/data/final_tot.txt')
  args = parser.parse_args([])
  return args

# assume that data already spanned
def invalid_data_checker(args,tot_list):
  print("start invalid data checker",flush=True)
  for rootpath,dirs,files in os.walk(str(args.folder)):
    for file in files:
      filepath = os.path.join(rootpath,file)
      if not range_test(filepath):
        os.remove(filepath)
        tot_list.remove(int(file.split('.npz')[0]))
        print("deleted!!", filepath,flush=True)
        continue
  return tot_list
      
def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files      

def range_test(melpath):
  npzzz = np.load(melpath)
  audio = npzzz['audio']
  mel = npzzz['mel'].T
  mel = torch.from_numpy(mel).squeeze(0)
  mel_segment_length = 16000 // 256 + 2
  max_mel_start = mel.size(1) - mel_segment_length
  try:
    mel_start = random.randint(0, max_mel_start)
  except:
    return False
  return True

def pre_exist_check(log_id):
  bucket = s3.Bucket('mindlogic-tts')
  pre_exist_data = bucket.objects.filter(Prefix='take/{}/final_train.txt'.format(log_id)).all()
  datalist = [el.key for el in pre_exist_data]
  if datalist.find('take/{}/final_train.txt'.format(log_id))==-1:
    return False
  return True

def multi_s3_download(root,split_id, mel_path):
  print("[DOWNLOAD START] split id: {}".format(split_id),flush=True)
  new_path = root / Path("{}.npz".format(split_id))
  if os.path.isfile(str(new_path)):
    return
  s3.Object('mindlogic-tts',mel_path.split+'-2').download_file(str(new_path))
  return


def preparation(log_id:int, step:int):
  args = parse_args()
  args.folder.mkdir(exist_ok=True, parents=True)
  
  tklog = TakeLog.objects.get(pk=log_id)
  speaker_list = tklog.speaker_list
  
  # download from s3 (step 1)
  if step==1:
    # data download using multiprocess pool
    with mp.Pool(processes=3) as pool:
      pool.map(multi_s3_download,[(args.folder, el.id, el.mel_path,) for el in tklog.bucket.data.all()])

  # default step
  if os.path.isfile(args.tot_file):
    os.remove(args.tot_file)
  tot_mapper = open(args.tot_file,'a')
  tot_list = list()
  for rootpath,dirs,files in os.walk(str(args.folder)):
    for file in files:
      filepath = os.path.join(rootpath,file)
      if file.find('.npz')==-1:
        continue
      tot_list.append(file.split('.npz')[0])
      tot_mapper.write(filepath+'\n')
  tot_mapper.close()
    
  # training/validation set selection (step 2)
  if step==2:
    val_list = list()
    print("training/validation file selection start",flush=True)
    if os.path.isfile(args.train_file):
      os.remove(args.train_file)
    trn_mapper = open(args.train_file,'a')
    if os.path.isfile(args.val_file):
      os.remove(args.val_file)
    val_mapper = open(args.val_file,'a')
    if os.path.isfile(args.tot_file):
      os.remove(args.tot_file)
    tot_mapper = open(args.tot_file,'a')
    
    tot_list = invalid_data_checker(args,tot_list)
    
    for el in tot_list:
      filepath = str(args.folder/Path("{}.npz".format(el)))
      tot_mapper.write(filepath+'\n')
    tot_mapper.close()
    
    # data download + random validation selection
    bucket = s3.Bucket('mindlogic-tts')
    
    for speaker_id in speaker_list:
      tar_spkr = Speaker.objects.get(pk=speaker_id)
      parse_type = "nltk" if tar_spkr.augflag==5 else "celeb"
      tmp_val_list = list()
      
      if parse_type=='nltk':
        dirpath = args.folder / Path('speaker-{}'.format(speaker_id))
        os.makedirs(str(dirpath),exist_ok=True)
        spkr_dataset = bucket.objects.filter(Prefix='nltk/speaker-{}/'.format(speaker_id)).all()
      else:
        dirpath = args.folder/Path('speaker-{}'.format(speaker_id))
        os.makedirs(str(dirpath),exist_ok=True)
        spkr_dataset = bucket.objects.filter(Prefix='celeb/speaker-{}/'.format(speaker_id)).all()
      
      for el in spkr_dataset:
        parse_split_id = int(el.key.split('/')[-1].split('.')[0])
        tmp_tot_list = []
        if el.key.endswith('.npz-2') and os.path.isfile(str(args.folder/Path("{}.npz".format(parse_split_id)))):
          tmp_tot_list.append("/".join(el.key.split('/')[1:]))
      tmp_val_list = random.sample(tmp_tot_list,len(tmp_tot_list)//10)
      val_list += tmp_val_list
    print("VAL_LIST",val_list,flush=True)
    
    # rearrange to structure
    for rootpath,dirs,files in os.walk(str(args.folder)):
      for file in files:
        filepath = os.path.join(rootpath,file)
        search_key = int(file.split('.npz')[0])
        if search_key in val_list:
          val_mapper.write(filepath+'\n')
        else:
          trn_mapper.write(filepath+'\n')
          
    trn_mapper.close()
    val_mapper.close()
    
    with open(args.train_file,'r') as f:
      s3.Object('mindlogic-tts','take/{}/final_train.txt'.format(log_id)).put(Body=f)
    with open(args.val_file,'r') as f:
      s3.Object('mindlogic-tts','take/{}/final_val.txt'.format(log_id)).put(Body=f)
    with open(args.tot_file,'r') as f:
      s3.Object('mindlogic-tts','take/{}/final_tot.txt'.format(log_id)).put(Body=f)

  # load checkpoint and resume training (step 3)
  if step==3:
    if pre_exist_check(log_id):
      s3.Object('mindlogic-tts','take/{}/final_train.txt'.format(log_id)).download_file('/app/data/final_train.txt')
      s3.Object('mindlogic-tts','take/{}/final_val.txt'.format(log_id)).download_file('/app/data/final_val.txt')
      s3.Object('mindlogic-tts','take/{}/final_tot.txt'.format(log_id)).download_file('/app/data/final_tot.txt')
    else:
      return
    
    tmp_totlist = files_to_list('/app/data/final_tot.txt')
    totlist = [int(el.split('/')[-1].split('.npz')[0]) for el in tmp_totlist]
    splitlist = SplitSource.objects.filter(pk__in=totlist)
    
    with mp.Pool(processes=3) as pool:
      pool.map(multi_s3_download,[(args.folder, el.id, el.mel_path,) for el in splitlist])
  
  print("finished",flush=True)
  

def transfer_preparation(log_id:int, ckpt_id:int, step:int):
  return