"""!
@file segmentation.py
@brief File containing an implementation of the Mask R-CNN model for object detection and segmentation.
@author Diogo J. Paulo
@version 1.0
"""
import argparse
from tqdm import tqdm  # progress bar
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import shutil
import json
import os
import torch
import sys
import torchvision
from torchvision import datasets, models
from torch.utils.data import DataLoader
import copy
import math
import cv2
import albumentations as A  # data augmentation library
from albumentations.pytorch import ToTensorV2
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def get_transforms(train):
    """!
        Function that applies the transformations to the images and bounding boxes.

        @param train (Boolean) that indicates if the transformations are applied to the training set or not.

        @return transform (A.Compose) The transformations to be applied to the images and bounding boxes.
        """
    if train:
        transform = A.Compose([
            A.Resize(600, 600),  # our input size can be 600px
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2(),
            # A.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600),  # our input size can be 600px
            ToTensorV2(),
            # A.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ], bbox_params=A.BboxParams(format='coco'))
    return transform


class FigDetection(datasets.VisionDataset):
    """!
        Class representing the figs.

        @param datasets.VisionDataset: Pytorch dataset class.
        """

    def __init__(self, root, split='train', transforms=None):
        """!
        Constructor of the class.

        @param root (string) Path to the dataset.
        @param split (string) Split of the dataset (train, valid, test).
        @param transforms (callable, optional) Optional transform to be applied on a sample and target.
        """
        super().__init__(root, transforms)
        self.split = split  # train, valid, test
        if self.split == 'train':
            self.coco = COCO(os.path.join(root, split, "train.json"))  # annotatiosn stored here
        elif self.split == 'valid':
            self.coco = COCO(os.path.join(root, split, "valid.json"))
        elif self.split == 'test':
            self.coco = COCO(os.path.join(root, split, "test.json"))
        else:
            raise ValueError("Split must be train, valid or test")
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]

    def _load_image(self, id: int):
        """!
        Function that loads an image.

        @param id (int) Id of the image to be loaded.

        @return a tuple (image, (height, width)) where height and width are the dimensions of the image.
        """
        path = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, (image.shape[0], image.shape[1])

    def _load_target(self, id):
        """!
        Function that loads the target of an image.

        @param id (int) Id of the image to be loaded.

        @return a list containing the annotations of the image.
        """
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        """!
        Function that returns the image and the target of an image.

        @param index (int) Index of the image to be loaded.

        @:return a tuple (image, target) where target is a dictionary containing the annotations of the image.
        """
        id = self.ids[index]
        image, shape = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))

        boxes = [t['bbox'] + [t['category_id']] for t in target]  # required annotation format for albumentations
        segmentation_masks = [t['segmentation'] for t in target]

        transformed = None
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)

        image = transformed['image']
        boxes = transformed['bboxes']
        # remove [] from segmentation masks
        segmentation_masks1 = segmentation_masks[0]
        segmentation_masks2 = segmentation_masks[1]
        segmentation_masks = segmentation_masks1 + segmentation_masks2
        for i in range(len(segmentation_masks)):
            if not segmentation_masks[i]:
                segmentation_masks.pop(i)

        new_segmentation_masks = []  # convert from list of polygons to binary masks
        for mask in segmentation_masks:
            polygon = np.array(mask).reshape(-1, 2).astype(np.int32)
            mask1 = np.zeros((shape[0], shape[1]), dtype=np.uint8)  # same shape as image
            cv2.fillPoly(mask1, [polygon], 1)
            mask1 = cv2.resize(mask1, (600, 600))
            new_segmentation_masks.append(mask1)

        new_boxes = []  # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        new_boxes = np.mean(new_boxes, axis=0)

        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        segmentation_masks = torch.tensor(new_segmentation_masks, dtype=torch.uint8)

        targ = {'boxes': boxes, 'labels': torch.tensor([t['category_id'] for t in target], dtype=torch.int64),
                'masks': segmentation_masks}  # here is our transformed target

        image = image / 255

        return image, targ

    def __len__(self):
        """!
        Function that returns the length of the dataset.

        @:return the length of the dataset.
        """
        return len(self.ids)


def create_dataset(all_annotations, imgs_path):
    """!
    Function that splits the dataset into train, test and valid folders.

    @param all_annotations (COCO) Annotations of the dataset.
    @param imgs_path (str) Path to the images of the dataset.

    @:return None
    """
    # dataset_path is current path/directory and create a new folder called dataset
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    # split images inside dataset_path into train*0.8, test*0.2 and valid=0.2*train
    # all_images = [i for i in os.listdir(imgs_path) if i.endswith('.jpg')]
    all_images = get_all_images(imgs_path)
    train_images = all_images[:int(len(all_images) * 0.8)]  # 0.8*all
    test_images = all_images[int(len(all_images) * 0.8):]  # 0.2*all
    val_images = train_images[:int(len(train_images) * 0.2)]  # 0.2*train
    train_images = train_images[int(len(train_images) * 0.2):]  # 0.8*train
    # split also annotations json file into train, test and valid json files in the same way
    train_annotations = COCO()
    # 0.8*train
    train_annotations.dataset['images'] = [all_annotations.dataset['images'][i] for i in
                                           range(len(all_annotations.dataset['images'])) if
                                           all_annotations.dataset['images'][i]['file_name'] in [j.split('\\')[-1] for j
                                                                                                 in train_images]]
    train_annotations.dataset['annotations'] = [all_annotations.dataset['annotations'][i] for i in
                                                range(len(all_annotations.dataset['annotations'])) if
                                                all_annotations.dataset['annotations'][i]['image_id'] in [j['id'] for j
                                                                                                          in
                                                                                                          train_annotations.dataset[
                                                                                                              'images']]]
    train_annotations.dataset['categories'] = all_annotations.dataset['categories']
    train_annotations.createIndex()
    train_annotations.dataset['info'] = all_annotations.dataset['info']
    train_annotations.dataset['licenses'] = all_annotations.dataset['licenses']

    # 0.2*test
    test_annotations = COCO()
    test_annotations.dataset['images'] = [all_annotations.dataset['images'][i] for i in
                                          range(len(all_annotations.dataset['images'])) if
                                          all_annotations.dataset['images'][i]['file_name'] in [j.split('\\')[-1] for j
                                                                                                in test_images]]
    test_annotations.dataset['annotations'] = [all_annotations.dataset['annotations'][i] for i in
                                               range(len(all_annotations.dataset['annotations'])) if
                                               all_annotations.dataset['annotations'][i]['image_id'] in [j['id'] for j
                                                                                                         in
                                                                                                         test_annotations.dataset[
                                                                                                             'images']]]
    test_annotations.dataset['categories'] = all_annotations.dataset['categories']
    test_annotations.createIndex()
    test_annotations.dataset['info'] = all_annotations.dataset['info']
    test_annotations.dataset['licenses'] = all_annotations.dataset['licenses']
    # 0.2*val
    val_annotations = COCO()
    val_annotations.dataset['images'] = [all_annotations.dataset['images'][i] for i in
                                         range(len(all_annotations.dataset['images'])) if
                                         all_annotations.dataset['images'][i]['file_name'] in [j.split('\\')[-1] for j
                                                                                               in val_images]]
    val_annotations.dataset['annotations'] = [all_annotations.dataset['annotations'][i] for i in
                                              range(len(all_annotations.dataset['annotations'])) if
                                              all_annotations.dataset['annotations'][i]['image_id'] in [j['id'] for j in
                                                                                                        val_annotations.dataset[
                                                                                                            'images']]]
    val_annotations.dataset['categories'] = all_annotations.dataset['categories']
    val_annotations.createIndex()
    val_annotations.dataset['info'] = all_annotations.dataset['info']
    val_annotations.dataset['licenses'] = all_annotations.dataset['licenses']

    # create train, test and valid folders
    if not os.path.exists(os.path.join(dataset_path, 'train')):
        os.mkdir(os.path.join(dataset_path, 'train'))
    if not os.path.exists(os.path.join(dataset_path, 'test')):
        os.mkdir(os.path.join(dataset_path, 'test'))
    if not os.path.exists(os.path.join(dataset_path, 'valid')):
        os.mkdir(os.path.join(dataset_path, 'valid'))

    # copy images to train, test and valid folders
    for image in train_images:
        source_img = image
        image = image.split('\\')[-1]
        shutil.copy(source_img, os.path.join(dataset_path, 'train', image))
    for image in test_images:
        source_img = image
        image = image.split('\\')[-1]
        shutil.copy(source_img, os.path.join(dataset_path, 'test', image))
    for image in val_images:
        source_img = image
        image = image.split('\\')[-1]
        shutil.copy(source_img, os.path.join(dataset_path, 'valid', image))

    with open(dataset_path + '/train' + '/train.json', 'w') as f:
        json.dump(train_annotations.dataset, f)
    with open(dataset_path + '/test' + '/test.json', 'w') as f:
        json.dump(test_annotations.dataset, f)
    with open(dataset_path + '/valid' + '/valid.json', 'w') as f:
        json.dump(val_annotations.dataset, f)


def collate_fn(batch):
    """!
    Function to collate the data in the batch

    @param batch: batch of data

    @return: tuple of images and targets
    """
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, loader, device, epoch):
    """!
    Function to train the model for one epoch.

    @param model: model to train
    @param optimizer: optimizer to use
    @param loader: data loader
    @param device: device to use
    @param epoch: current epoch

    @return: None
    """
    model.train()

    all_losses = []
    all_losses_dict = []

    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)  # the model computes the loss automatically if we pass in targets
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)

        losses.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)  # clip the gradients
        optimizer.step()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig")  # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)

    all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing
    print(
        "Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
            epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
            all_losses_dict['loss_classifier'].mean(),
            all_losses_dict['loss_box_reg'].mean(),
            all_losses_dict['loss_rpn_box_reg'].mean(),
            all_losses_dict['loss_objectness'].mean()
        ))


def train(model, dataset_path):
    """!
    Function to train the model.

    @param model: model to train
    @param dataset_path: path to the dataset

    @return: None
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_dataset = FigDetection(root=dataset_path, transforms=get_transforms(True))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)
    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.0005)

    for epoch in range(100):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        torch.save(model.state_dict(), f'maskrcnn_prickly_pears_{epoch}.pth')


def inference(model_path, img, type):
    """!
    Function to perform inference on the image.

    @param model_path: model path to use
    @param img: image to perform inference on

    @return: Tuple (mask, pred_boxes) where mask is the mask of the image and pred_boxes are the predicted boxes
    """
    model = maskrcnn()

    if type == 'prickly_pear':
        classes = ['background', 'Prickly pear']
    elif type == 'fig':
        classes = ['background', 'Fig']
    else:
        raise ValueError('Type must be prickly_pear or fig')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.load_state_dict(torch.load(model_path))

    model.eval()
    torch.cuda.empty_cache()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (600, 600))
    img = img / 255
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img, dtype=torch.float32)
    img_int = torch.tensor(img * 255, dtype=torch.uint8)
    with torch.no_grad():
        prediction = model([img.to(device)])
        pred = prediction[0]

    pred_labels = [classes[i] for i in pred['labels'][pred['scores'] > 0.8]]
    test_img = draw_bounding_boxes(img_int,
                                   pred['boxes'][pred['scores'] > 0.8],
                                   pred_labels,
                                   width=4).permute(1, 2, 0)
    test_img = np.array(test_img)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    mask = pred['masks'][pred['scores'] > 0.8]  # for multiple masks
    if len(mask) == 0:
        return None, None
    mask = mask[0]
    mask = mask.squeeze(1)
    mask = mask.permute(1, 2, 0)
    mask = mask.cpu().numpy()

    if type == 'prickly_pear':
        mask = np.where(mask > 0.4, 255, 0).astype(np.uint8)  # threshold to get the mask for prickly pears
    elif type == 'fig':
        mask = np.where(mask > 0.2, 255, 0).astype(np.uint8)  # threshold to get the mask for normal figs

    pred_boxes = pred['boxes'][pred['scores'] > 0.8]

    return mask, pred_boxes, test_img


def maskrcnn():
    """!
    Function to create the model (MaskRCNN). The model predicts two classes: background and (fig or prickly pear).

    @return: MaskRCNN model adapted to our problem.
    """

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                   2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                                                  hidden_layer,
                                                                                  2)
    return model


def align_image(img, mask, model_path, type):
    """!
    Function to align the image using the mask.

    @param img: image to align
    @param mask: mask to use for aligning the image
    @param model_path: model path to use for inference

    @return: the aligned image
    """
    img = cv2.imread(img)
    weight, height = img.shape[1], img.shape[0]
    # Extrapolate mask and rotate image
    mask = cv2.resize(mask, (weight, height))
    countours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    elipse = cv2.fitEllipse(max(countours, key=cv2.contourArea))

    theta = elipse[2]
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D(elipse[0], theta, 1), (img.shape[1], img.shape[0]))
    mask = cv2.warpAffine(mask, cv2.getRotationMatrix2D(elipse[0], theta, 1), (img.shape[1], img.shape[0]))
    # centroid of the mask
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # max point of the mask
    max_point = np.where(mask == 255)
    max_point = (max_point[1][0], max_point[0][0])
    min_point = np.where(mask == 255)
    min_point = (min_point[1][-1], min_point[0][-1])

    if math.dist(max_point, (cx, cy)) < math.dist(min_point, (cx, cy)):
        img = cv2.flip(img, 0)

    # crop the image
    _, boxes2, _ = inference(model_path, img, type)
    if boxes2 is None:
        return None
    weight1, height1 = img.shape[1], img.shape[0]
    boxes2 = boxes2.cpu().numpy().astype(np.int32)
    boxes2 = boxes2[0]
    boxes2[0] = int(boxes2[0] * (weight1 / 600))
    boxes2[1] = int(boxes2[1] * (height1 / 600))
    boxes2[2] = int(boxes2[2] * (weight1 / 600))
    boxes2[3] = int(boxes2[3] * (height1 / 600))
    # crop the image in format x, y, width, height
    img = img[boxes2[1]:boxes2[3], boxes2[0]:boxes2[2]]

    return img


def crop_image(img, model, type):
    """!
    Function to crop the image using the bounding boxes.

    @param img: image to align
    @param model: model to use for inference

    @return: the cropped image
    """
    _, boxes2 = inference(model, img, type)
    if boxes2 is None:
        return None
    weight1, height1 = img.shape[1], img.shape[0]
    boxes2 = boxes2.cpu().numpy().astype(np.int32)
    boxes2 = boxes2[0]
    boxes2[0] = int(boxes2[0] * (weight1 / 600))
    boxes2[1] = int(boxes2[1] * (height1 / 600))
    boxes2[2] = int(boxes2[2] * (weight1 / 600))
    boxes2[3] = int(boxes2[3] * (height1 / 600))
    # crop the image in format x, y, width, height
    img = img[boxes2[1]:boxes2[3], boxes2[0]:boxes2[2]]
    return img


def align_folder(source_dir, desti_dir, model_path, type):
    """!
    Function to align a folder of images.

    @return: None
    """
    img_dir = Path(source_dir)
    folders = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]
    sorted_folders = sorted(folders, key=lambda x: int(x))
    list_images = []
    for folder in sorted_folders:
        list_images += [os.path.join(img_dir, folder, f) for f in os.listdir(os.path.join(img_dir, folder))]
    desti_dir = Path(desti_dir)

    for i in range(0, len(list_images), 4):
        # create folders
        if not os.path.exists(desti_dir / '{:02d}'.format(i // 4 + 1)):
            os.makedirs(desti_dir / '{:02d}'.format(i // 4 + 1))
        for j in range(4):
            img = cv2.imread(list_images[i + j])
            mask, boxes, _ = inference(model_path, img, type)
            if mask is None or boxes is None:
                continue
            img = align_image(list_images[i + j], mask, model_path, type)
            if img is None:
                continue
            destination_path = desti_dir / '{:02d}'.format(i // 4 + 1) / list_images[i + j].split('\\')[-1]
            cv2.imwrite(str(destination_path), img)
        print('Folder {} done'.format(i // 4 + 1))


def mask_folder(source_dir, desti_dir, model_path, type):
    """!
    Function that creates a folder with the masks of the images. If the mask or the boxes are None, the image is
    discarded.

    @return: None
    """
    img_dir = Path(source_dir)
    folders = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]
    sorted_folders = sorted(folders, key=lambda x: int(x))
    list_images = []
    for folder in sorted_folders:
        list_images += [os.path.join(img_dir, folder, f) for f in os.listdir(os.path.join(img_dir, folder))]
    desti_dir = Path(desti_dir)

    for i in range(0, len(list_images), 4):
        # create folders
        if not os.path.exists(desti_dir / '{:02d}'.format(i // 4 + 1)):
            os.makedirs(desti_dir / '{:02d}'.format(i // 4 + 1))
        for j in range(4):
            img = cv2.imread(list_images[i + j])
            mask, boxes = inference(model_path, img, type)
            if mask is None or boxes is None:
                continue
            img = cv2.bitwise_and(img, img, mask=mask)
            img[np.where((img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
            destination_path = desti_dir / '{:02d}'.format(i // 4 + 1) / list_images[i + j].split('/')[-1]
            cv2.imwrite(str(destination_path), img)
        print('Folder {} done'.format(i // 4 + 1))


def crop_folder(source_dir, desti_dir, model_path, type):
    """!
    Function that creates a folder with the cropped images.

    @return: None
    """
    img_dir = Path(source_dir)
    folders = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]
    sorted_folders = sorted(folders, key=lambda x: int(x))
    list_images = []
    for folder in sorted_folders:
        list_images += [os.path.join(img_dir, folder, f) for f in os.listdir(os.path.join(img_dir, folder))]
    desti_dir = Path(desti_dir)

    for i in range(0, len(list_images), 4):
        # create folders
        if not os.path.exists(desti_dir / '{:02d}'.format(i // 4 + 1)):
            os.makedirs(desti_dir / '{:02d}'.format(i // 4 + 1))
        for j in range(4):
            img = cv2.imread(list_images[i + j])
            img = crop_image(img, model_path, type)
            if img is None:
                continue
            destination_path = desti_dir / '{:02d}'.format(i // 4 + 1) / list_images[i + j].split('/')[-1]
            cv2.imwrite(str(destination_path), img)
        print('Folder {} done'.format(i // 4 + 1))


def get_all_images(source_dir):
    """!
    Function that gathers all the images in one folder.

    @return: None
    """
    folder = Path(source_dir)
    list_images = pd.Series((folder.glob(r'**/*.jpg'))).astype(str).sort_values().reset_index(drop=True)
    list_images = list(list_images)

    return list_images


def get_mask_annotation(all_annotations, img, all_images_path):
    """!
    Function that gets the mask annotation of an image.

    @param all_annotations: annotations of the dataset
    @param img: image to get the mask annotation

    @return: the mask annotation
    """
    img_size = cv2.imread(all_images_path + '/' + img)
    img_size = img_size.shape
    img_id = [i['id'] for i in all_annotations.dataset['images'] if i['file_name'] == img][0]
    print(img_id)
    print(img)
    ann_ids = all_annotations.getAnnIds(imgIds=img_id)
    anns = all_annotations.loadAnns(ann_ids)

    # get segmentation masks
    segmentation_masks = [t['segmentation'] for t in anns]
    segmentation_masks1 = segmentation_masks[0]
    segmentation_masks2 = segmentation_masks[1]
    segmentation_masks = segmentation_masks1 + segmentation_masks2
    for i in range(len(segmentation_masks)):
        if segmentation_masks[i] == []:
            segmentation_masks.pop(i)

    # convert to binary masks
    new_segmentation_masks = []
    for mask in segmentation_masks:
        polygon = np.array(mask).reshape(-1, 2).astype(np.int32)
        mask1 = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        cv2.fillPoly(mask1, [polygon], 1)
        new_segmentation_masks.append(mask1)

    # save in binary masks format
    mask = new_segmentation_masks[0]
    mask = np.where(mask == 1, 255, 0).astype(np.uint8)  # threshold to get the mask for prickly pears
    cv2.imwrite(img + '_mask.jpg', mask)

    return mask


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command')

    create_dataset_parser = subparsers.add_parser('create-dataset')
    create_dataset_parser.add_argument('--all_annotations', type=str, required=True,
                                       help='Path to the annotations json file')
    create_dataset_parser.add_argument('--imgs_path', type=str, required=True,
                                       help='Path to the images')

    train_parser = subparsers.add_parser('train')

    inference_parser = subparsers.add_parser('inference')
    inference_parser.add_argument('--model', type=str, required=True,
                                  help='Path to the model')
    inference_parser.add_argument('--img', type=str, required=True,
                                  help='Path to the image')
    inference_parser.add_argument('--type', type=str, required=True,
                                  help='Type of the image: prickly_pear or fig')

    align_parser = subparsers.add_parser('align')
    align_parser.add_argument('--model', type=str, required=True,
                              help='Path to the model')
    align_parser.add_argument('--img', type=str,
                              help='Path to the image')
    align_parser.add_argument('--type', type=str, required=True,
                              help='Type of the image: prickly_pear or fig')

    align_folder_parser = subparsers.add_parser('align-folder')
    align_folder_parser.add_argument('--source_dir', type=str, required=True,
                                     help='Path to the source directory')
    align_folder_parser.add_argument('--desti_dir', type=str, required=True,
                                     help='Path to the destination directory')
    align_folder_parser.add_argument('--model', type=str, required=True,
                                     help='Path to the model')
    align_folder_parser.add_argument('--type', type=str, required=True,
                                     help='Type of the image: prickly_pear or fig')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = create_parser()

    if args.command == 'create-dataset':
        all_annotations = COCO(args.all_annotations)
        create_dataset(all_annotations, args.imgs_path)
    elif args.command == 'train':
        model = maskrcnn()
        train(model, 'dataset')
    elif args.command == 'inference':
        img = cv2.imread(args.img)
        mask, pred_boxes, test_img = inference(args.model, img, args.type)
        if mask is None or pred_boxes is None:
            print('No mask or boxes')
        else:
            cv2.imwrite('bounding_boxes_detection.jpg', test_img)
            cv2.imwrite('mask.jpg', mask)
    elif args.command == 'align':
        img = cv2.imread(args.img)
        mask, boxes, _ = inference(args.model, img, args.type)
        if mask is None or boxes is None:
            print('No mask or boxes')
        else:
            aligned_img = align_image(args.img, mask, args.model, args.type)
            cv2.imwrite('aligned_image.jpg', aligned_img)
    elif args.command == 'align-folder':
        align_folder(args.source_dir, args.desti_dir, args.model, args.type)
