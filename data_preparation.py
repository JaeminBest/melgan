'''
generate final_train.txt
if custom(-c):
  download data from s3, fill folder(--folder) 
  and then generate final_train.txt and final_val.txt
else:
  load data from folder(--folder) and then generate final_train(val).txt
  
'''


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from pathlib import Path
from tqdm import tqdm
from librosa.core import load
import argparse
import json
import librosa
import shutil
import torch
import random

from utils.credential import s3
from utils.audio import convert_sr

SR = 22050


def span_path(args,full_filepath):
  path_cvt = Path(full_filepath)
  if args.folder==path_cvt.parents[0]:
    return full_filepath
  parse_speaker_id = full_filepath.split('/')[-4].split('-')[-1]
  parse_file_id = full_filepath.split('/')[-3].split('-')[-1]
  parse_filename = full_filepath.split('/')[-1]
  new_path = args.folder / Path("{}-{}-{}".format(parse_speaker_id,parse_file_id,parse_filename))
  os.system('mv {} {}'.format(full_filepath,str(new_path)))
  return new_path

def reconstruct_path(key):
  splits = key.split('-')
  return "speaker-{}/file-{}/wav/{}".format(splits[0],splits[1],splits[2])

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--folder", type=Path, required=True)
  parser.add_argument("--data-param-path", type=Path, default='/app/params.json')
  parser.add_argument("--train-file", type=Path, default='/app/data/final_train.txt')
  parser.add_argument("--val-file", type=Path, default='/app/data/final_val.txt')
  parser.add_argument('-c', type=bool, default=(True if os.environ.get('CUSTOM','')=='1' else False))
  parser.add_argument('-d', action='store_true') # download flag if you change params.json you need to set this flag
  parser.add_argument('-s', action='store_true') # setting training_file, validation_file
  args = parser.parse_args()
  return args

# assume that data already spanned
def invalid_data_checker(args):
  print("start invalid data checker",flush=True)
  for rootpath,dirs,files in os.walk(str(args.folder)):
    for file in files:
      if file.find('.wav')==-1:
        continue
      filepath = os.path.join(rootpath,file)
      print(filepath,flush=True)
      path_cvt = Path(filepath)
      if args.folder==path_cvt.parents[0]:
        data, sampling_rate = load(filepath, sr=SR)
        if data.shape==(0,):
          print("deleted!!!!!!",flush=True)
          os.remove(filepath)
          os.remove(filepath.split('.wav')[0]+'.mel')
          continue


def main():
  args = parse_args()
  args.folder.mkdir(exist_ok=True, parents=True)
    
  # download from s3
  if not args.d:
    # custom dataset
    if args.c:
      # json load
      with open(args.data_param_path) as f:
        json_data = json.load(f)
      data_list = json_data['data']
      
      # data download
      for data in data_list:
        parse_id = int(data.split('-')[-1])
        parse_type = data.split('-')[0]
        
        if parse_type=='nltk':
          dirpath = args.folder / Path('speaker-{}'.format(parse_id))
          os.makedirs(str(dirpath),exist_ok=True)
          os.system("aws s3 cp s3://mindlogic-tts-cache/nltk/speaker-{} {} --recursive".format(parse_id,os.path.join(*dirpath.parts)))
        else:
          dirpath = args.folder/Path('speaker-{}'.format(parse_id))
          os.makedirs(str(dirpath),exist_ok=True)
          os.system("aws s3 cp s3://mindlogic-tts-cache/celeb/speaker-{} {} --recursive".format(parse_id,os.path.join(*dirpath.parts)))
      
      # rearrange to structure
      print("rearrange start",flush=True)
      for rootpath,dirs,files in os.walk(str(args.folder)):
        for file in files:
          if file.find('.wav')==-1:
            continue
          filepath = os.path.join(rootpath,file)
          convert_sr(filepath,SR)
          new_path = span_path(args,filepath)
    
      print("dummy clean start",flush=True)
      # delete dummy directory
      for data in data_list:
        parse_id = int(data.split('-')[-1])
        parse_type = data.split('-')[0]
        dirpath = os.path.join(args.folder,'speaker-{}'.format(parse_id))
        if os.path.isdir(dirpath):
          shutil.rmtree(dirpath)
      
    else:
      print("building list of files...",flush=True)
      for rootpath,dirs,files in os.walk(str(args.folder)):
        for file in files:
          if file.find('.wav')==-1:
            continue
          filepath = os.path.join(rootpath,file)
          convert_sr(filepath,SR)
  
  if args.s:
    invalid_data_checker(args)
    print("training/validation file selection start",flush=True)
    if os.path.isfile(args.train_file):
      os.remove(args.train_file)
    trn_mapper = open(args.train_file,'a')
    if os.path.isfile(args.val_file):
      os.remove(args.val_file)
    val_mapper = open(args.val_file,'a')
    
    if args.c:
      # json load
      with open(args.data_param_path) as f:
        json_data = json.load(f)
      data_list = json_data['data']
      
      # data download + random validation selection
      bucket = s3.Bucket('mindlogic-tts')
      val_list = list()
      for data in data_list:
        parse_id = int(data.split('-')[-1])
        parse_type = data.split('-')[0]
        tmp_tot_list = list()
        tmp_val_list = list()
        
        if parse_type=='nltk':
          dirpath = args.folder / Path('speaker-{}'.format(parse_id))
          os.makedirs(str(dirpath),exist_ok=True)
          spkr_dataset = bucket.objects.filter(Prefix='nltk/speaker-{}/'.format(parse_id)).all()
        else:
          dirpath = args.folder/Path('speaker-{}'.format(parse_id))
          os.makedirs(str(dirpath),exist_ok=True)
          spkr_dataset = bucket.objects.filter(Prefix='celeb/speaker-{}/'.format(parse_id)).all()
        
        tmp_tot_list = ["/".join(el.key.split('/')[1:]) for el in spkr_dataset]
        tmp_val_list = random.sample(tmp_tot_list,len(tmp_tot_list)//10)
        val_list += tmp_val_list
      print(val_list)
      
      # rearrange to structure
      print("rearrange start",flush=True)
      for rootpath,dirs,files in os.walk(str(args.folder)):
        for file in files:
          if file.find('.wav')==-1:
            continue
          filepath = os.path.join(rootpath,file)
          search_key = reconstruct_path(file)
          if search_key in val_list:
            val_mapper.write(filepath+'\n')
          else:
            trn_mapper.write(filepath+'\n')
      
    else:
      tot_list = list()
      val_list = list()
      print("scanning list of files...",flush=True)
      for rootpath,dirs,files in os.walk(str(args.folder)):
        for file in files:
          filepath = os.path.join(rootpath,file)
          if filepath.find('.wav')!=-1:
            tot_list.append(filepath)
      val_list = random.sample(tot_list,len(tot_list)//10)
      
      print("writing file list...")
      for el in tot_list:
        if el in val_list:
          val_mapper.write(el+'\n')
        else:
          trn_mapper.write(el+'\n')

    trn_mapper.close()
    val_mapper.close()
    

  print("finished",flush=True)
  

if __name__ == "__main__":
  main()

