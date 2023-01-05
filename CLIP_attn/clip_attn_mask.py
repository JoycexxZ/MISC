import clip_m as clip
from distutils import text_file
import torch
from PIL import Image
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import shutil

SAVE_DIR = "/public/home/fengyf2/projects/3dVG/data/frames_square/{}/rn50_top10"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "RN50"
# model_name = "ViT-B/32"
model, preprocess = clip.load(model_name, device=device)

raw_labels = []
with open('/public/home/fengyf2/projects/3dVG/data/scannet/scannetv2-labels.combined.tsv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        # print(row)
        raw_labels.append(row['raw_category'])

refering_file = json.load(open("/public/home/fengyf2/projects/3dVG/data/ScanRefer_filtered_val.json", 'r'))
text_list = {}
flag = 0
for refering in refering_file:
    if refering['scene_id'] not in text_list:
        text_list[refering['scene_id']] = []
        
    description = refering['description']
    _des = description.replace('.', ' ').replace(',', ' ')
    text_list[refering['scene_id']] += ['a ' + n for n in raw_labels if ' ' + n + ' ' in _des \
                  and n != 'floor' and n != 'wall' and n != 'ceiling' and n != 'object']
text_list = {k: list(set(t)) for k, t in text_list.items()}

for scene_name in tqdm(text_list):
    if os.path.exists(SAVE_DIR.format(scene_name)):
        shutil.rmtree(SAVE_DIR.format(scene_name))
    img_folder = f"/public/home/fengyf2/projects/3dVG/data/frames_square/{scene_name}/color/"
    img_list = os.listdir(img_folder)
    image_list = []
    for idx, img_path in enumerate(img_list):
        image_list.append(preprocess(Image.open(os.path.join(img_folder, img_path))))
    image_list = torch.stack(image_list, dim=0).to(device)

    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    plt.rcParams['figure.figsize'] = [20, 20]
    with torch.no_grad(): 
        print(text_list[scene_name])
        text = clip.tokenize(text_list[scene_name], truncate=True).to(device)
        
        text_feat = model.encode_text(text)
        image_feat, seq_feat = model.encode_image(image_list)
        if model_name == 'RN50':
            seq_feat = seq_feat.permute(1, 0, 2)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True) # (num_text, 512)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True) # (num_img, 512)
        seq_feat = seq_feat / seq_feat.norm(dim=-1, keepdim=True) # (num_img, 49, 512)
        
        log_scale = model.logit_scale.exp()
        mask = 4 * text_feat @ seq_feat.transpose(-1, -2) # (num_img, num_text, 49)
        if model_name == "ViT-B/32":
            mask_th = -0.9
            mask = -mask
        else:
            mask_th = 0.5
        # mask = mask.softmax(dim=2)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 7, 7)
        mask = torch.nn.functional.interpolate(mask, size=(224, 224), mode='bilinear')
        mask = mask > mask_th
        
        logits_per_image = log_scale * text_feat @ image_feat.transpose(-1, -2)
        logits_per_text = logits_per_image.t()

        probs = logits_per_text.softmax(dim=0).cpu().numpy()
    
    K = 10
    for idx, t in enumerate(text_list[scene_name]):
        topk = np.argsort(probs[:, idx])[::-1][:K]
        for k in range(K):
            img_m = mask[topk[k], idx].detach().cpu().numpy() * 255
            img_m = Image.fromarray(img_m.astype(np.uint8))
            if not os.path.exists(SAVE_DIR.format(scene_name)):
                os.makedirs(SAVE_DIR.format(scene_name))
            img_m.save(os.path.join(SAVE_DIR.format(scene_name), f"{scene_name}_{t.replace('a ', '').replace(' ', '-')}_{k}_{img_list[topk[k]].replace('.jpg', '')}.png"))