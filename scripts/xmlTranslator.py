from os import listdir
from os.path import isfile, join
import random
year = 2017
path = './devkit/%s/Annotations' % year
only_files = [f for f in listdir(path) if isfile(join(path, f))]
target_path = './devkit/%s/ImageSets/' % year
train_file = open(target_path + 'train.txt', 'w')
test_file = open(target_path + 'val.txt', 'w')
test_portion = 0.1
for file_name in only_files:
    a = random.uniform(0, 1.0)
    if a < test_portion:
        test_file.write(file_name.split('.')[0])
        test_file.write('\n')
    else:
        train_file.write(file_name.split('.')[0])
        train_file.write('\n')
test_file.close()
train_file.close()
