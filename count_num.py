import os
import sys

def get_label_featurelist_dict(path):
    with open(path) as f:
        lines = f.readlines()
        label_img_dict = dict()
        current_label = ''
        img_list = []
        i = 0
        class_num = 0

        test = []
        for line in lines:
            i = i + 1
            label = line.strip().split('/')[0]
            test.append(label)
            if label != current_label:
                class_num = class_num + 1
                if len(img_list) > 0:
                    label_img_dict[current_label] = img_list
                    img_list = []
                current_label = label
                img_list.append(line.strip())
                if i == len(lines):
                    label_img_dict[current_label] = img_list
            else:
                img_list.append(line.strip())
                if i == len(lines):
                    label_img_dict[current_label] = img_list
        print(class_num)
        return label_img_dict


if __name__ == '__main__':
    #img_dict = get_label_featurelist_dict('/workspace/data/qyc/blueface_ansia_95660_and_deepint_ansia_143050/train.lst')
    img_dict = get_label_featurelist_dict('/workspace/data/qyc/data/lfw_all_5747/img.lst')
    f = open('./lfw_clean_faceemore_up_2.lst', 'w')
  
    count = 0
    for key in img_dict:
        if len(img_dict[key]) > 1:
            for i in range(len(img_dict[key])):
                f.write(img_dict[key][i] + '\n')
    
    print(count)  

    
  
