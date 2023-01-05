import os
from random import random
import sys
from copy import deepcopy
import argparse
import json
import clip
import PIL.ImageFilter as filter
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np
from tqdm import tqdm
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from scanrefer.model_util_scannet import ScannetDatasetConfig
from scanrefer.config import CONF
from scanrefer.dataset import ScannetReferenceDataset
from utils import proj_util

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_dataloader(args, scanrefer, all_scene_list, split):   
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=args.use_height,
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        augment=args.use_augment,
        lang_emb_type=args.lang_emb_type
    )
    # sampler = DistributedSampler(dataset)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    return dataset, dataloader

def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes):
    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
    if num_scenes == -1: 
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes
    
    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list

def get_scene_proj(points, colors, views=4, height=224, width=224, save=False, scene_name=""):
    """

    Args:
        points (Torch.Tensor): [B, num_points, 3]
        colors (Torch.Tensor): [B, num_points, 3]
        views (int): 
    Returns:
        projs: [views, 3, h, w]
    """
    
    if views == 4:
        # angles1 = torch.tensor([[0, -np.pi/3, 0],
        #             #    [0, -np.pi/3, np.pi/4],
        #                [0, -np.pi/3, 2*np.pi/4],
        #             #    [0, -np.pi/3, 3*np.pi/4],
        #                [0, -np.pi/3, 4*np.pi/4],
        #             #    [0, -np.pi/3, 5*np.pi/4],
        #                [0, -np.pi/3, 6*np.pi/4]
        #             #    [0, -np.pi/3, 7*np.pi/4]
        #                ])
        # angles2 = torch.tensor([
        #                [0, -np.pi/3, 0],
        #                [0, -np.pi/3, np.pi/4],
        #                [0, -np.pi/3, 2*np.pi/4],
        #                [0, -np.pi/3, 3*np.pi/4],
        #                [0, -np.pi/3, 4*np.pi/4],
        #                [0, -np.pi/3, 5*np.pi/4],
        #                [0, -np.pi/3, 6*np.pi/4],
        #                [0, -np.pi/3, 7*np.pi/4]
        # ])
        angles = torch.tensor([                       
                       [0, -np.pi/2, 0],
                       [0, -np.pi/2, np.pi/8],
                       [0, -np.pi/2, 2*np.pi/8],
                       [0, -np.pi/2, 3*np.pi/8],
                       [0, -np.pi/2, 4*np.pi/8],
                       [0, -np.pi/2, 5*np.pi/8],
                       [0, -np.pi/2, 6*np.pi/8],
                       [0, -np.pi/2, 7*np.pi/8],
                       [0, -np.pi/2, 8*np.pi/8],
                       [0, -np.pi/2, 9*np.pi/8],
                       [0, -np.pi/2, 10*np.pi/8],
                       [0, -np.pi/2, 11*np.pi/8],
                       [0, -np.pi/2, 12*np.pi/8],
                       [0, -np.pi/2, 13*np.pi/8],
                       [0, -np.pi/2, 14*np.pi/8],
                       [0, -np.pi/2, 15*np.pi/8],
                       
                       [0, -np.pi/2, 0]
                    #    [0, -np.pi/2, 2*np.pi/4],
                    #    [0, -np.pi/2, 4*np.pi/4],
                    #    [0, -np.pi/2, 6*np.pi/4],
                       ])
        
        
        

    imgs1, info, sel = proj_util.points2img(points, angle=angles, translation='mid', image_height=height, image_width=width, colors=colors, size_x=2, size_y=2, return_info=True, around=16)
    # imgs2, info2 = proj_util.points2img(points, angle=angles2, translation='mid', image_height=height, image_width=width, colors=colors, size_x=2, size_y=2, return_info=True, step=True)
    imags = []
    
    if save:
        for v in range(8):
            img_path = os.path.join("scene_rot", f"{scene_name}_view{v+1}.png")
            img_ = transforms.ToPILImage()(imgs1[0, v].detach().cpu())
            img_.save(img_path)
    
    for v in range(angles.shape[0]):
        img = transforms.ToPILImage()(imgs1[0, v].detach().cpu())
        img = img.filter(filter.MaxFilter)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                    (0.26862954, 0.26130258, 0.27577711))(img)
        imags.append(img)
        
    # for v in range(8):
    #     img = transforms.ToPILImage()(imgs2[0, v].detach().cpu())
    #     img = img.filter(filter.MaxFilter)
    #     img = transforms.ToTensor()(img)
    #     img = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
    #                                 (0.26862954, 0.26130258, 0.27577711))(img)
    #     imags.append(img)
    
    return torch.stack(imags), info, sel


class Logger(object):
    def __init__(self, file_path: str = "./Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_path, "x")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
    
class MetricCalculator(object):
    
    def __init__(self, config):
        self.counter = {}
        # keys = ["overall", "average", "top score"] + [f"view{i+1}" for i in range(8)]
        keys = ["overall", "unique", "multiple"]
        objs = ["total"] + list(config.type2class.keys())
        for key in keys:
            self.counter[key] = {}
            for obj in objs:
                self.counter[key][obj] = {"correct": 0, "total": 0}
        self.class2type = config.class2type
        self.config = config
        self.pred_scores = {}
        for key in keys:
            self.pred_scores[key] = {}
            for obj in config.type2class.keys():
                self.pred_scores[key][obj] = np.zeros(18)
        
    def step(self, inputs, pred, ref_center_label, t, mode="default"):
        # similarity = similarity.detach().cpu().numpy() # (batch_size, 18)
        # pred_labels = np.argmax(similarity, axis=1)
        
        # b_size = pred.shape[0]
        # num_fine_img = (ref_center_label[0:8, 2] <= t['through'][0:8]).sum()
        fine_image = []
                
        for m in range(20):
            if m < 16:
                if ref_center_label[m, 2] <= t['through'][m]:
                    fine_image.append(m)
        
        if mode == "default":
            for b in range(1):
                gt_labels = inputs["sem_cls_label"].squeeze(0).cpu().numpy()
                box_id = torch.argmax(inputs["ref_box_label"].squeeze(0)).item()
                gt_label = self.class2type[gt_labels[box_id]]
                # pred_label = self.class2type[int(pred_labels[b])]
                # view = f"view{inputs['view'][b].item()}"
                
                self.counter["overall"]["total"]["total"] += 1
                self.counter["overall"][gt_label]["total"] += 1
                if inputs["unique_multiple"][0] == 0:
                    self.counter["unique"]["total"]["total"] += 1
                    self.counter["unique"][gt_label]["total"] += 1
                else:
                    self.counter["multiple"]["total"]["total"] += 1
                    self.counter["multiple"][gt_label]["total"] += 1
                
                if pred >= 4:
                    if ref_center_label[pred, 2] <= t['through'][pred]:
                        self.counter["overall"]["total"]["correct"] += 1
                        self.counter["overall"][gt_label]["correct"] += 1
                        if inputs["unique_multiple"][0] == 0:
                            self.counter["unique"]["total"]["correct"] += 1
                            self.counter["unique"][gt_label]["correct"] += 1
                        else:
                            self.counter["multiple"]["total"]["correct"] += 1
                            self.counter["multiple"][gt_label]["correct"] += 1
                else:
                    if len(fine_image) == 0:
                        self.counter["overall"]["total"]["correct"] += 1
                        self.counter["overall"][gt_label]["correct"] += 1
                        if inputs["unique_multiple"][0] == 0:
                            self.counter["unique"]["total"]["correct"] += 1
                            self.counter["unique"][gt_label]["correct"] += 1
                        else:
                            self.counter["multiple"]["total"]["correct"] += 1
                            self.counter["multiple"][gt_label]["correct"] += 1
                            
        elif mode == "exist":
            gt_labels = inputs["sem_cls_label"].squeeze(0).cpu().numpy()
            box_id = torch.argmax(inputs["ref_box_label"].squeeze(0)).item()
            gt_label = self.class2type[gt_labels[box_id]]
            # pred_label = self.class2type[int(pred_labels[b])]
            # view = f"view{inputs['view'][b].item()}"
                
            self.counter["overall"]["total"]["total"] += 1
            self.counter["overall"][gt_label]["total"] += 1
            if inputs["unique_multiple"][0] == 0:
                self.counter["unique"]["total"]["total"] += 1
                self.counter["unique"][gt_label]["total"] += 1
            else:
                self.counter["multiple"]["total"]["total"] += 1
                self.counter["multiple"][gt_label]["total"] += 1
            for i in range(pred.shape[0]):
                if pred[i] < 16:
                    if ref_center_label[pred[i], 2] <= t['through'][pred[i]]:
                        self.counter["overall"]["total"]["correct"] += 1
                        self.counter["overall"][gt_label]["correct"] += 1
                        if inputs["unique_multiple"][0] == 0:
                            self.counter["unique"]["total"]["correct"] += 1
                            self.counter["unique"][gt_label]["correct"] += 1
                        else:
                            self.counter["multiple"]["total"]["correct"] += 1
                            self.counter["multiple"][gt_label]["correct"] += 1
                        break
                else:
                    if len(fine_image) == 0:
                        self.counter["overall"]["total"]["correct"] += 1
                        self.counter["overall"][gt_label]["correct"] += 1
                        if inputs["unique_multiple"][0] == 0:
                            self.counter["unique"]["total"]["correct"] += 1
                            self.counter["unique"][gt_label]["correct"] += 1
                        else:
                            self.counter["multiple"]["total"]["correct"] += 1
                            self.counter["multiple"][gt_label]["correct"] += 1
                        break
                    
        elif mode == "cover":
            gt_labels = inputs["sem_cls_label"].squeeze(0).cpu().numpy()
            box_id = torch.argmax(inputs["ref_box_label"].squeeze(0)).item()
            gt_label = self.class2type[gt_labels[box_id]]
            # pred_label = self.class2type[int(pred_labels[b])]
            # view = f"view{inputs['view'][b].item()}"
                
            self.counter["overall"]["total"]["total"] += len(fine_image)
            self.counter["overall"][gt_label]["total"] += len(fine_image)
            if inputs["unique_multiple"][0] == 0:
                self.counter["unique"]["total"]["total"] += len(fine_image)
                self.counter["unique"][gt_label]["total"] += len(fine_image)
            else:
                self.counter["multiple"]["total"]["total"] += len(fine_image)
                self.counter["multiple"][gt_label]["total"] += len(fine_image)
            for i in range(pred.shape[0]):
                if pred[i] >= 4:
                    if ref_center_label[pred[i], 2] <= t['through'][pred[i]]:
                        self.counter["overall"]["total"]["correct"] += 1
                        self.counter["overall"][gt_label]["correct"] += 1
                        if inputs["unique_multiple"][0] == 0:
                            self.counter["unique"]["total"]["correct"] += 1
                            self.counter["unique"][gt_label]["correct"] += 1
                        else:
                            self.counter["multiple"]["total"]["correct"] += 1
                            self.counter["multiple"][gt_label]["correct"] += 1
                else:
                    if ref_center_label[pred[i], 2] <= 0:
                        self.counter["overall"]["total"]["correct"] += 1
                        self.counter["overall"][gt_label]["correct"] += 1
                        if inputs["unique_multiple"][0] == 0:
                            self.counter["unique"]["total"]["correct"] += 1
                            self.counter["unique"][gt_label]["correct"] += 1
                        else:
                            self.counter["multiple"]["total"]["correct"] += 1
                            self.counter["multiple"][gt_label]["correct"] += 1
                            
        elif mode == "acc":
            gt_labels = inputs["sem_cls_label"].squeeze(0).cpu().numpy()
            box_id = torch.argmax(inputs["ref_box_label"].squeeze(0)).item()
            gt_label = self.class2type[gt_labels[box_id]]
            # pred_label = self.class2type[int(pred_labels[b])]
            # view = f"view{inputs['view'][b].item()}"
                
            self.counter["overall"]["total"]["total"] += pred.shape[0]
            self.counter["overall"][gt_label]["total"] += pred.shape[0]
            if inputs["unique_multiple"][0] == 0:
                self.counter["unique"]["total"]["total"] += pred.shape[0]
                self.counter["unique"][gt_label]["total"] += pred.shape[0]
            else:
                self.counter["multiple"]["total"]["total"] += pred.shape[0]
                self.counter["multiple"][gt_label]["total"] += pred.shape[0]
            for i in range(pred.shape[0]):
                if pred[i] >= 4:
                    if ref_center_label[pred[i], 2] <= t['through'][pred[i]]:
                        self.counter["overall"]["total"]["correct"] += 1
                        self.counter["overall"][gt_label]["correct"] += 1
                        if inputs["unique_multiple"][0] == 0:
                            self.counter["unique"]["total"]["correct"] += 1
                            self.counter["unique"][gt_label]["correct"] += 1
                        else:
                            self.counter["multiple"]["total"]["correct"] += 1
                            self.counter["multiple"][gt_label]["correct"] += 1
                else:
                    if ref_center_label[pred[i], 2] <= 0:
                        self.counter["overall"]["total"]["correct"] += 1
                        self.counter["overall"][gt_label]["correct"] += 1
                        if inputs["unique_multiple"][0] == 0:
                            self.counter["unique"]["total"]["correct"] += 1
                            self.counter["unique"][gt_label]["correct"] += 1
                        else:
                            self.counter["multiple"]["total"]["correct"] += 1
                            self.counter["multiple"][gt_label]["correct"] += 1
    
    def get_metric(self):
        metric = {}
        eps = 1e-6
        for view in self.counter.keys():
            metric[view] = {}
            for obj in self.counter[view].keys():
                metric[view][obj] = self.counter[view][obj]["correct"] / (self.counter[view][obj]["total"] + eps)
        
        keys = ["overall", "unique", "multiple"]
        objs = ["total"] + list(self.config.type2class.keys())
        for key in keys:
            for obj in objs:
                for k in ["correct", "total"]:
                    self.counter[key][obj][k] = 0
        top5_scores = {}
        return metric, top5_scores


def main(args):
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(
        SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # angles = torch.tensor([                       
    #                    [0, -np.pi/2, 0],
    #                    [0, -np.pi/2, np.pi/4],
    #                    [0, -np.pi/2, 2*np.pi/4],
    #                    [0, -np.pi/2, 3*np.pi/4],
    #                    [0, -np.pi/2, 4*np.pi/4],
    #                    [0, -np.pi/2, 5*np.pi/4],
    #                    [0, -np.pi/2, 6*np.pi/4],
    #                    [0, -np.pi/2, 7*np.pi/4],
                       
    #                    [0, -np.pi/2, 0],
    #                    [0, -np.pi/2, 2*np.pi/4],
    #                    [0, -np.pi/2, 4*np.pi/4],
    #                    [0, -np.pi/2, 6*np.pi/4],
    #                    ])
    
    angles = torch.tensor([                       
                       [0, -np.pi/2, 0],
                       [0, -np.pi/2, np.pi/8],
                       [0, -np.pi/2, 2*np.pi/8],
                       [0, -np.pi/2, 3*np.pi/8],
                       [0, -np.pi/2, 4*np.pi/8],
                       [0, -np.pi/2, 5*np.pi/8],
                       [0, -np.pi/2, 6*np.pi/8],
                       [0, -np.pi/2, 7*np.pi/8],
                       [0, -np.pi/2, 8*np.pi/8],
                       [0, -np.pi/2, 9*np.pi/8],
                       [0, -np.pi/2, 10*np.pi/8],
                       [0, -np.pi/2, 11*np.pi/8],
                       [0, -np.pi/2, 12*np.pi/8],
                       [0, -np.pi/2, 13*np.pi/8],
                       [0, -np.pi/2, 14*np.pi/8],
                       [0, -np.pi/2, 15*np.pi/8],
                       
                       [0, -np.pi/2, 0],
                       [0, -np.pi/2, 2*np.pi/4]
                    #    [0, -np.pi/2, 4*np.pi/4],
                    #    [0, -np.pi/2, 6*np.pi/4],
                       ])
    angles = angles.to(device)
    rot_mat = proj_util.euler2mat(angles).transpose(1, 2)
    views = angles.shape[0]
    
    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train")
    config = ScannetDatasetConfig()
    metric_calculator = MetricCalculator(config)
    
    # device = 'cpu'
    # print("processing data...")
    cover_mean = 0
    idx = 0
    K = 3
    base = "scene0000_00"
    process_mode = ["exist", "cover"]
    dis_score = {}
    keys = ["overall", "unique", "multiple"]
    objs = ["total"] + list(config.type2class.keys())
    for key in keys:
        dis_score[key] = {}
        for obj in objs:
            dis_score[key][obj] = torch.zeros(11)
    
    
    for i in range(1):
        print(f"processing {process_mode[i]} ...")
        with torch.no_grad():
            model, preprocess = clip.load('ViT-B/32', "cuda:0")
            for inputs in tqdm(train_dataloader):
                # To device
                for key in inputs:
                    if torch.torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(device)
                        inputs[key].requires_grad_(False)
                        
                scene_name = inputs["scene_id"][0]
                # if scene_name != "scene0006_00":
                #     continue
                # print(scene_name)
                # gt_labels = inputs["sem_cls_label"].squeeze(0).cpu().numpy()
                # object_id = inputs["object_id"].squeeze(0).cpu().numpy()
                ref_center_label = inputs["ref_center_label"]
                points = inputs["point_clouds"]
                colors = inputs["pcl_color"]
                imgs, t, sel  = get_scene_proj(points, colors)
                imgs = imgs.cuda()
                
                
                gt_labels = inputs["sem_cls_label"].squeeze(0).cpu().numpy()
                box_id = torch.argmax(inputs["ref_box_label"].squeeze(0)).item()
                gt_label = config.class2type[gt_labels[box_id]]
                
                lang = inputs["raw_lang"][0]
                lang = lang.replace(',', '.')
                lang = lang.split('.')
                # print(lang)
                # raise NotImplementedError("!!")
                # fake_lang = "a white cabinet in the corner of the room ."
                # lang[0] += "in point cloud"
                
                # lang[0] = f"a {gt_label} in point cloud"
                lang_token = clip.tokenize(lang[0], truncate=True)
                lang_token = lang_token.cuda()
                
                # print(ref_center_label.shape)
                # print(t['trans'].shape)
                ref_center_label -= t['trans'].reshape(1, 3).to('cuda:0')
                ref_center_label = ref_center_label.repeat(views, 1)
                # print(ref_center_label)
                
                ref_center_label = torch.matmul(ref_center_label.unsqueeze(1), rot_mat)
                ref_center_label = ref_center_label.reshape(views, -1)
                # print(ref_center_label.reshape(8, -1))
                logits_per_image, logits_per_text = model(imgs, lang_token)
                
                probs = logits_per_text.softmax(dim=-1).cpu().numpy()
                
                # pred = np.argmax(probs, axis=1)
                pred_topk = probs[0].argsort()[::-1][:K]
                
                sel_p = torch.zeros_like(points).squeeze(0)
                for v in pred_topk:
                    sel_p = torch.clamp(sel_p + sel[v], min=0, max=1) 
                    
                cover = (sel_p.sum()/3) / sel_p.shape[0]
                
                # num_fine_img = (ref_center_label[0:4, 2] <= 0).sum() + (ref_center_label[4:12, 2] <= t).sum()
                # print(num_fine_img)
                # print(t)
                # print(ref_center_label)
                # raise NotImplementedError("!!")
                
                # pred = np.random.randint(12)
                fine_image = []
                
                for m in range(12):
                    if m < 8:
                        if ref_center_label[m, 2] <= t['through'][m]:
                            fine_image.append(m)
                if len(fine_image) == 0:
                    for j in range(8, 12):
                        fine_image.append(j)
                        
                # if scene_name == "scene0006_00" and gt_label == "table":
                #     # metric_calculator.step(inputs, pred_topk, ref_center_label, t, mode=process_mode[i])
                #     print(scene_name)
                #     print(lang[0])
                #     print(pred_topk)
                #     print(fine_image)
                # if scene_name == "scene0007_00":
                #     break
                
                
                # if base != scene_name:
                #     print(base)
                #     metric, top5_counter = metric_calculator.get_metric()
                #     for view in metric.keys():
                #         print(f"====== {view} ======")
                #         for obj in metric[view].keys():
                #             # score_bin = math.floor(metric[view][obj] * 10)
                #             # dis_score[view][obj][score_bin] += 1
                #             print(f"{obj}: {metric[view][obj]:.4f}")
                #         print("\n")
                #     base = scene_name
                    
                metric_calculator.step(inputs, pred_topk, ref_center_label, t, mode=process_mode[i])
                # idx += 1
                
                # if idx > 200:
                #     break
                    
                    
                cover_mean += cover
                
                # idx += 1
                
                # if idx > 1000:
                #     print(cover_mean/idx)
                #     break
                #     # raise NotImplementedError("!!")
                rn = np.random.randint(50)
                if rn == 0:
                    print(scene_name)
                    print(lang[0])
                    print(logits_per_text.cpu().numpy())
                    print(pred_topk)
                    print(fine_image)
                    print(cover)
                #     print(f"idx: {idx}" )
                #     print(probs)
                    print(ref_center_label)
                    idx += 1
                
                if idx == 10:
                    raise NotImplementedError("!!")
                # if inputs["unique_multiple"][0] == 1:
                #     total_u += 1
                # else:
                #     total_m += 1
                # idx += 1
                # if idx > 10:
                    # print(acc)
                    # raise NotImplementedError("!!")
                    # break
                

            # print("unique:")
            # print(acc_u/total_u)
            # print("multiple:")
            # print(acc_m/total_m)
            metric, top5_counter = metric_calculator.get_metric()
            # sys.stdout = Logger(f"./test_scene_12view_top5_{process_mode[i]}_results.txt")
            for view in metric.keys():
                print(f"====== {view} ======")
                for obj in metric[view].keys():
                    print(f"{obj}: {metric[view][obj]:.4f}")
                print("\n")
            # sys.stdout = Logger(f"./test_scene_12view_top5_dis_results.txt")
            
            # for view in dis_score.keys():
            #     print(f"====== {view} ======")
            #     for obj in dis_score[view].keys():
            #         print(f"{obj}: ")
            #         for i in range(11):
            #             print(f"{i}: {dis_score[view][obj][i]}")
            #     print("\n")
            
            
def process_scene_imgs(args):
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(
        SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }
    
    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print("processing data...")
    with torch.no_grad():
        for inputs in tqdm(train_dataloader):
            # To device
            for key in inputs:
                if torch.torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(device)
                    inputs[key].requires_grad_(False)
                    
            scene_name = inputs["scene_id"][0]
            # gt_labels = inputs["sem_cls_label"].squeeze(0).cpu().numpy()
            # object_id = inputs["object_id"].squeeze(0).cpu().numpy()
            points = inputs["point_clouds"]
            colors = inputs["pcl_color"]
            imgs = get_scene_proj(points, colors, save=True, scene_name=scene_name)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=1)
    parser.add_argument("--num_points", type=int, help="lang num max", default=50000)
    parser.add_argument("--use_height", action="store_true", help="Use height in input.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")    
    parser.add_argument("--use_augment", action="store_true", help="Use augmentation.")
    parser.add_argument("--lang_emb_type", type=str, help="lang emb type", default="clip")
    
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    args = parser.parse_args()
    
    main(args)
    # process_scene_imgs(args)