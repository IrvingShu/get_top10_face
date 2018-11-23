import os
import sys
import shutil

src_folder = '/workspace/data/qyc/data/v19/img'
 
dst_folder = '/workspace/data/qyc/data/v19/include_lfw'

if __name__ == '__main__':
    with open('./result/up-thres-new_0.5-intra-lfwface-to-v19.txt') as f:
        lines = f.readlines()
        result = []
        for line in lines:
            label = line.strip().split(' ')[1]
            result.append(label)
        label_set = list(set(result))
        
        for i in range(len(label_set)):

            dst = os.path.join(dst_folder, label_set[i])
            shutil.move(os.path.join(src_folder, label_set[i]), dst)
