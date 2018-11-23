import os
import sys

import matio
import numpy as np
import argparse
import time

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
        return label_img_dict,class_num


def get_label_center_fea_dict(root_folder, feature_path, fea_dims):
    print('Start load feature')
    (label_featurelist_dict, class_num) = get_label_featurelist_dict(feature_path)
    print('Finished load feature')
    label_center_dict = dict()
    # Calulate feature center
    print('class num is: %d' %(class_num))

    fea_mat = np.zeros((class_num, fea_dims), dtype=np.float32)
    count = 0
    for key in label_featurelist_dict:
        feature_sum = np.zeros(fea_dims, dtype=np.float64)
        cur_fea_list = label_featurelist_dict[key]
        for i in range(len(cur_fea_list)):
            full_path = os.path.join(root_folder, cur_fea_list[i])
            x_vec = load_feat(full_path) # matio.load_mat(full_path).flatten()
            feature_sum += x_vec
        feature_center = feature_sum / len(cur_fea_list)
        fea_mat[count] = feature_center

        label_center_dict[count] = key
        count = count + 1
    print('Finishe cal center')
    return label_center_dict,fea_mat

def load_npy(npy_file):
    mat = None
    if os.path.exists(npy_file):
        mat = np.load(npy_file)
    else:
        err_info = 'Can not find file: ' + npy_file
        raise Exception(err_info)
    return mat

def load_feat(feat_file, flatten=True):
    feat = None
    if feat_file.endswith('npy'):
        feat = load_npy(feat_file)
    elif feat_file.endswith('bin'):
        feat =matio.load_mat(feat_file)
    else:
        raise Exception(
            'Unsupported feature file. Only support .npy and .bin (OpenCV Mat file)')
    if flatten:
        feat = feat.flatten()
    return feat

#merge 
def get_extra_inter_class(from_label_index, from_feamat, to_label_index, to_feamat, inter_threshold, intra_threshold):
    inter_result = []
    intra_result = []
    class_num = from_feamat.shape[0]

    row_block = 1
    if class_num > 50000:
        row_block = 100000
    else:
        row_block = class_num
   
    loop_num = class_num // row_block
    print('loop_num: %d' %(loop_num))
    out_result = []
    count = 0

    from_feamat_norm = np.linalg.norm(from_feamat, axis=1)
    to_feamat_norm = np.linalg.norm(to_feamat, axis=1)
    count = 0
    for i in range(loop_num):
        part = from_feamat[i * row_block : (i+1) * row_block, :]
        fea_dist_mat = np.dot(part, to_feamat.T)
        (m, n) = fea_dist_mat.shape
        for j in range(m):
            max_sim = -1
            max_sim_label = ''            
            for k in range(n):
                sim = fea_dist_mat[j][k] / (from_feamat_norm[i * row_block + j] * to_feamat_norm[k] + 0.00001)
                max_sim = sim
                max_sim_label = to_label_index[k]

                if max_sim >= intra_threshold:
                    print(from_label_index[i * row_block + j] + ' ' +  max_sim_label + ' ' + str(max_sim))
                    intra_result.append(from_label_index[i * row_block + j] + ' ' +  max_sim_label + ' ' + str(max_sim))                 
        count = (i * 1 + 1) * row_block
    print('Final Batch')
    print(count, class_num)
    if count < class_num:
        part = from_feamat[count:class_num, :]
        fea_dist_mat = np.dot(part, to_feamat.T)
        (m,n) = fea_dist_mat.shape
        for j in range(m):
            max_sim = -1
            max_sim_label = ''
            for k in range(n):
                sim = fea_dist_mat[j][k] / (from_feamat_norm[count + j] * to_feamat_norm[k] + 0.00001)
                max_sim = sim
                max_sim_label = to_label_index[k]
                if max_sim >= intra_threshold:
                    print(from_label_index[count + j] + ' ' +  max_sim_label + ' ' + str(max_sim))
                    intra_result.append(from_label_index[count + j] + ' ' +  max_sim_label + ' ' + str(max_sim)) 

    return inter_result, intra_result
            

def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--from-feature-root-folder', type=str, help='feature root folder')
    parser.add_argument('--from-feature-list-path', type=str, help='feature list path') 
    parser.add_argument('--to-feature-root-folder', type=str, help='feature root folder')
    parser.add_argument('--to-feature-list-path', type=str, help='feature list path')
    parser.add_argument('--feature-dims', type=int, help='feature dims', default=512)
 
    parser.add_argument('--inter-threshold', type=float, help='less threshold will merge it as new label') 
    parser.add_argument('--intra-threshold', type=float, help='above threshold will merge it as existed label') 
    parser.add_argument('--save-inter-path', type=str, help='save inter path')
    parser.add_argument('--save-intra-path', type=str, help='save intra path')
    
    return parser.parse_args(argv)

def main(args):
    print('===> args:\n', args)
    from_fea_root_folder = args.from_feature_root_folder
    from_fea_list_path = args.from_feature_list_path

    to_fea_root_folder = args.to_feature_root_folder
    to_fea_list_path = args.to_feature_list_path

    fea_dims = args.feature_dims
    inter_threshold = args.inter_threshold
    intra_threshold = args.intra_threshold
        
    #read all input feture
    #
    print('read from feature')
    (from_label_index , from_feamat)= get_label_center_fea_dict(from_fea_root_folder, from_fea_list_path, fea_dims)    
    print('######################')
    print('read to feature')
    (to_label_index, to_feamat) = get_label_center_fea_dict(to_fea_root_folder, to_fea_list_path, fea_dims)

    #
    extra_inter_data, extra_intra_data = get_extra_inter_class(from_label_index, from_feamat, to_label_index, to_feamat,inter_threshold, intra_threshold)
    save_inter_path = args.save_inter_path
    save_intra_path = args.save_intra_path

    with open(save_inter_path , 'w') as f, open(save_intra_path, 'w') as f1:
        for i in range(len(extra_inter_data)):
            f.write(extra_inter_data[i]+ '\n')
        for j in range(len(extra_intra_data)):
            f1.write(extra_intra_data[j] + '\n')        
    print('Finished merge inter label')
if __name__ == '__main__':
    start_time = time.time()
    main(parse_args(sys.argv[1:]))
    end_time = time.time()
    print('time cost: %f' %(end_time - start_time))
    
