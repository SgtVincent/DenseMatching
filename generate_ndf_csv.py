import pandas as pd
import glob
import os
import shutil
from collections import OrderedDict
import numpy as np
import random
image_folder = '../pwarpc_generate_data/train'
classes = ['bowl','mug','bottle']
new_datadir = 'data/ndf_simulator_images'
new_image_folder = os.path.join(new_datadir,'images')
new_mask_folder = os.path.join(new_datadir,'masks')
csv_suffix = '_pairs_ndf.csv'

if not os.path.exists(new_image_folder):
    os.makedirs(new_image_folder,exist_ok = True)
if not os.path.exists(new_mask_folder):
    os.makedirs(new_mask_folder,exist_ok = True)


df = pd.DataFrame(columns = ['source_image','target_image','class'])

for i,cls in enumerate(classes):
    cls_id = i + 1
    current_dict = OrderedDict()
    count = 0
    for img_path in glob.glob(os.path.join(image_folder,cls,'*rgb.png')):
        img_file = os.path.basename(img_path)
        new_img_path = os.path.join(new_image_folder,img_file)
        shutil.copyfile(img_path,new_img_path)

        mask_file = img_file.replace('rgb','mask')
        mask_path = img_path.replace('rgb','mask')
        new_mask_path = os.path.join(new_mask_folder, mask_file)
        shutil.copyfile(mask_path,new_mask_path)

        camera_id = img_file.split('rgb')[0][-1]
        count += 1
        if camera_id in current_dict.keys():
            current_dict[camera_id].append(img_file)
        else:
            current_dict[camera_id] = [img_file]

    for ls in current_dict.values():
        random.shuffle(ls)

    while len(current_dict.keys()):
        cam_id = np.random.choice(list(current_dict.keys()))
        other_cam_id = [cam for cam in current_dict.keys() if cam != cam_id]
        if len(current_dict[cam_id]):
            src_file = current_dict[cam_id].pop(0)
            if len(other_cam_id):
                trg_cam_id = np.random.choice(other_cam_id)
                while not len(current_dict[trg_cam_id]):
                    del current_dict[trg_cam_id]
                    other_cam_id.remove(trg_cam_id)
                    if len(other_cam_id):
                        trg_cam_id = np.random.choice(other_cam_id)
                    else:
                        trg_cam_id = cam_id
                trg_file = current_dict[trg_cam_id].pop(0)

            else:
                trg_file = current_dict[cam_id].pop(0)
            df = df.append({'source_image':src_file,'target_image':trg_file,'class':cls_id},ignore_index=True)
        else:
            del current_dict[cam_id]

print(df)
df.to_csv(os.path.join(new_datadir,'all'+ csv_suffix),index=False)
val_df = df.loc[::5,:]
print(val_df)
val_df.to_csv(os.path.join(new_datadir,'val'+ csv_suffix),index=False)
train_df = df[~df.isin(val_df)].dropna()
print(train_df)
train_df.to_csv(os.path.join(new_datadir,'train'+ csv_suffix),index=False)
