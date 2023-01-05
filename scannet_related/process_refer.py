import os
import sys
from copy import deepcopy
import argparse
import json

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from scannet.model_util_scannet import ScannetDatasetConfig
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

def get_gt_proj(centers, sizes, points, colors, views=4, height=224, width=224):
    """

    Args:
        centers (Torch.Tensor): [64, 3]
        sizes (Torch.Tensor): [64, 3]
        points (Torch.Tensor): [50000, 3]
        colors (Torch.Tensor): [50000, 3]
        views (int): 
    Returns:
        projs: [true_proposals, views, 3, h, w]
    """
    
    if views == 8:
        angles = torch.tensor([[0, -np.pi/2, 0],
                               [np.pi/2, -np.pi/2, 0],
                               [np.pi/2, -np.pi/2, np.pi],
                               [0, -np.pi/2, np.pi],
                               [0, -np.pi/3, np.pi/4],
                               [0, -np.pi/3, 3*np.pi/4],
                               [0, -np.pi/3, 5*np.pi/4],
                               [0, -np.pi/3, 7*np.pi/4]])
    elif views == 1:
        # raise NotImplementedError(f'Unknown views {views}')
        angles = torch.tensor([[np.pi/4, np.pi/4, -np.pi/4]])
    else:
        raise NotImplementedError(f'Unknown views {views}')
    
    num_proposals = centers.shape[0]
    num_points = points.shape[0]

    zero_num = (sizes.sum(dim=1) == 0).sum()
    true_proposals = num_proposals - zero_num
    box_min = centers[:true_proposals, :] - sizes[:true_proposals, :] / 2 # [true_proposals, 3]
    box_max = centers[:true_proposals, :] + sizes[:true_proposals, :] / 2 # [true_proposals, 3]
    _points = []
    _colors = []
    _imgs = []
    
    for n in range(true_proposals):
        indices = (points >= box_min[n]) \
                    & (points <= box_max[n])
        indices = indices[:, 0] & indices[:, 1] & indices[:, 2]
        _points_sel = points[indices]
        _colors_sel = colors[indices]
        _points_padding = _points_sel[-1].repeat((num_points - _points_sel.shape[0]), 1)
        _colors_padding = _colors_sel[-1].repeat((num_points - _colors_sel.shape[0]), 1)
        _points_sel = torch.cat([_points_sel, _points_padding], dim=0)
        _colors_sel = torch.cat([_colors_sel, _colors_padding], dim=0)
        _points.append(_points_sel)
        _colors.append(_colors_sel)
        if len(_points) >= 8 or n == true_proposals - 1:
            _points = torch.stack(_points, dim=0)
            _colors = torch.stack(_colors, dim=0)
            imgs = proj_util.points2img(_points, _colors, angles, 'min', height, width, size_x=8, size_y=8, return_info=False)
            _imgs.append(imgs)
            _points = []
            _colors = []
            
    if len(_imgs) == 0:
        return None, 0

    imgs = torch.vstack(_imgs)
    # print(_points.shape)
    # print(_colors.shape)
    return imgs, true_proposals


def main(args):
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
    last_scene_name = ""
    last_scene_name2 = ""
    with torch.no_grad():
        for batch_idx, inputs in enumerate(train_dataloader):
            # To device
            for key in inputs:
                if torch.torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(device)
                    inputs[key].requires_grad_(False)
                    
            scene_name = inputs["scene_id"][0]
            gt_labels = inputs["sem_cls_label"].squeeze(0).cpu().numpy()
            object_id = inputs["object_id"].squeeze(0).cpu().numpy()
            
            '''
            # Point in box all
            out_folder = "ref_box_all"
            if scene_name != last_scene_name:
                last_scene_name = scene_name
                print("scene changes")
                imgs, true_proposals = get_gt_proj(centers=inputs["center_label"].squeeze(0),
                                                sizes=inputs["size_gts"].squeeze(0),
                                                points=inputs["point_clouds"].squeeze(0),
                                                colors=inputs["pcl_color"].squeeze(0),
                                                views=8)
                assert true_proposals == inputs["num_bbox"].item()
                if true_proposals == 0:
                    continue
                print(f"\n========= {scene_name}: {true_proposals} boxes - {object_id} =========")
                for p in range(true_proposals):
                    gt_label = DC.class2type[gt_labels[p]]
                    for v in range(8):
                        img_path = os.path.join(out_folder, f"{scene_name}_{p}_{gt_label}_proj{v+1}.png")
                        # print(imgs.shape)
                        img = transforms.ToPILImage()(imgs[p, v].detach().cpu())
                        img.save(img_path)
                print("ref_box_all finished")
                    
            # Point in refering box
            out_folder = "ref_box"
            box_id = torch.argmax(inputs["ref_box_label"].squeeze(0)).item()
            centers = (inputs["center_label"].squeeze(0))[box_id].unsqueeze(0)
            sizes = (inputs["size_gts"].squeeze(0))[box_id].unsqueeze(0)
            imgs, true_proposals = get_gt_proj(centers=centers,
                                               sizes=sizes,
                                               points=inputs["point_clouds"].squeeze(0),
                                               colors=inputs["pcl_color"].squeeze(0),
                                               views=8)
            gt_label = DC.class2type[gt_labels[box_id]]
            for v in range(8):
                img_path = os.path.join(out_folder, f"{scene_name}_{box_id}_{gt_label}_proj{v+1}.png")
                img = transforms.ToPILImage()(imgs[0, v].detach().cpu())
                img.save(img_path)
            # print("ref_box finished")
            '''
            
            # Get points directly
            angles = torch.tensor([[0, -np.pi/2, 0],
                        [np.pi/2, -np.pi/2, 0],
                        [np.pi/2, -np.pi/2, np.pi],
                        [0, -np.pi/2, np.pi],
                        [0, -np.pi/3, np.pi/4],
                        [0, -np.pi/3, 3*np.pi/4],
                        [0, -np.pi/3, 5*np.pi/4],
                        [0, -np.pi/3, 7*np.pi/4]])
            '''
            out_folder = "ref_points_all"
            if scene_name != last_scene_name2:
                last_scene_name2 = scene_name
                print("scene changes")
                true_proposals = inputs["num_bbox"].item()
                point_instance_label = inputs["point_instance_label"].squeeze(0)
                for p in range(true_proposals):
                    pts = inputs["point_clouds"].squeeze(0)[point_instance_label == p]
                    colors = inputs["pcl_color"].squeeze(0)[point_instance_label == p]
                    gt_label = DC.class2type[gt_labels[p]]
                    img = proj_util.points2img(points=pts.unsqueeze(0), colors=colors.unsqueeze(0), 
                                            angle=angles, translation='min',
                                            image_height=224, image_width=224, size_x=8, size_y=8)
                    for v in range(8):
                        img_path = os.path.join(out_folder, f"{scene_name}_{p}_{gt_label}_proj{v+1}.png")
                        img_ = transforms.ToPILImage()(img[0, v].detach().cpu())
                        img_.save(img_path)
                print("points finished")
            '''
            # Get refering points
            out_folder = "ref_points"
            point_instance_label = inputs["point_instance_label"].squeeze(0)
            box_id = torch.argmax(inputs["ref_box_label"].squeeze(0)).item()
            
            pts = inputs["point_clouds"].squeeze(0)[point_instance_label == box_id]
            colors = inputs["pcl_color"].squeeze(0)[point_instance_label == box_id]
            gt_label = DC.class2type[gt_labels[box_id]]
            img = proj_util.points2img(points=pts.unsqueeze(0), colors=colors.unsqueeze(0),
                                       angle=angles, translation='min',
                                       image_height=224, image_width=224, size_x=8, size_y=8)
            for v in range(8):
                img_path = os.path.join(out_folder, f"{scene_name}_{box_id}_{gt_label}_proj{v+1}.png")
                img_ = transforms.ToPILImage()(img[0, v].detach().cpu())
                img_.save(img_path)
                
            print(f"\n========= {scene_name}:  boxes - {object_id} =========")
            
            # print("ref_points finished")
            
            # For test only
            # if batch_idx > 10:
            #     break
                

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
    parser.add_argument("--lang_emb_type", type=str, help="lang emb type", default="glove")
    
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    args = parser.parse_args()
    
    main(args)