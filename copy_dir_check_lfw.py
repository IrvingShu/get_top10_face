import os
import sys
import shutil

src_folder = '/workspace/data/qyc/data/faces_emore/img'
 
dst_folder = '/workspace/data/qyc/data/faces_emore/lfw'

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
            label = line.strip().split(' ')[0]
            test.append(label)
            if label != current_label:
                class_num = class_num + 1
                if len(img_list) > 0:
                    label_img_dict[current_label] = img_list
                    img_list = []
                current_label = label
                img_list.append(line.strip().split(' ')[1])
                if i == len(lines):
                    label_img_dict[current_label] = img_list
            else:
                img_list.append(line.strip().split(' ')[1])
                if i == len(lines):
                    label_img_dict[current_label] = img_list
        return label_img_dict


if __name__ == '__main__':

    name_img_dict = get_label_featurelist_dict('/workspace/data/qyc/qyc_work/check_deepglint/lfw_faceemore_up_2.lst')
    for key in name_img_dict:
        for item in name_img_dict[key]:
            base_save_dir = os.path.join(dst_folder, str(i))
            if not os.path.exists(base_save_dir):
                os.makedirs(base_save_dir)

            dst1 = os.path.join(base_save_dir, dir1)
            dst2 = os.path.join(base_save_dir, dir2)
            
            shutil.copytree(os.path.join(src_folder, dir1), dst1)
            shutil.copytree(os.path.join(src_folder, dir2), dst2)
            i = i + 1
