import os

def convert_sr(source_file,sr):
  os.system("mv {} {}".format(source_file,source_file+"_2"))
  os.system('ffmpeg -y -i {} -acodec pcm_s16le -ac 1 -ar {} {}'.format(source_file+"_2", sr, source_file))
  os.remove(source_file+"_2")
  return