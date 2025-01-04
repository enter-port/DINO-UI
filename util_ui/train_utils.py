import torch
import cv2
import numpy as np
from torchvision.ops import nms
from utils.data_utils import recover_input
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def postprocess_output(hms, whs, offsets, confidence, device):
    """
    The post process of model output.
    Args:
        hms: heatmap
        whs: the height and width of bounding box
        offsets: center point offset
        confidence: the threshold of heatmap
        dev: torch device

    Returns:  The list of bounding box(x, y, w, h, score, label).

    """
    batch, output_h, output_w, c = hms.shape

    detections = []
    for b in range(batch):
        # (h, w, c) -> (-1, c)
        heat_map = hms[b].view([-1, c])
        # (h, w, 2) -> (-1, 2)
        wh = whs[b].view([-1, 2])
        # (h, w, 2) -> (-1, 2)
        offset = offsets[b].view([-1, 2])

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        xv, yv = xv.flatten().float(), yv.flatten().float()

        xv = xv.to(device)     # x axis coordinate of feature point
        yv = yv.to(device)     # y axis coordinate of feature point

        # torch.max[0] max value
        # torch.max[1] index of max value
        score, label = torch.max(heat_map, dim=-1)
        mask = score > confidence

        # Choose height, width and offset by confidence mask
        wh_mask = wh[mask]
        offset_mask = offset[mask]

        if len(wh_mask) == 0:
            detections.append([])
            continue

        # Adjust center of predict box
        xv_mask = torch.unsqueeze(xv[mask] + offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + offset_mask[..., 1], -1)

        # Get the (xmin, ymin, xmax, ymax)
        half_w, half_h = wh_mask[..., 0:1] / 2, wh_mask[..., 1:2] / 2
        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)

        # Bounding box coordinate normalize
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h

        # Concatenate the prediction
        detect = torch.cat(
            [bboxes, torch.unsqueeze(score[mask], -1), torch.unsqueeze(label[mask], -1).float()], dim=-1)
        detections.append(detect)

    return detections


def decode_bbox(prediction, input_shape, dev, image_shape=None, remove_pad=False, need_nms=False, nms_thres=0.4):
    """
    Decode postprocess_output output
    Args:
        prediction: postprecess_output output
        input_shape: model input shape
        dev: torch device
        image_shape: image shape
        remove_pad: model input is padding image, you should set remove_pad=True if you want to remove this pad
        need_nms: whether use NMS to remove redundant detect box
        nms_thres: nms threshold

    Returns:  The list of bounding box(x1, y1, x2, y2 score, label).

    """
    output = [[] for _ in prediction]

    for b, detection in enumerate(prediction):
        if len(detection) == 0:
            continue

        if need_nms:
            keep = nms(detection[:, :4], detection[:, 4], nms_thres)
            detection = detection[keep]

        output[b].append(detection)

        output[b] = torch.cat(output[b])
        if output[b] is not None:
            bboxes = output[b][:, 0:4]

            input_shape = torch.tensor(input_shape, device=dev)
            bboxes *= torch.cat([input_shape, input_shape], dim=-1)

            if remove_pad:
                assert image_shape is not None, \
                    "If remove_pad is True, image_shape must be set the shape of original image."
                ih, iw = input_shape
                h,  w = image_shape
                scale = min(iw/w, ih/h)
                nw, nh = int(scale * w), int(scale * h)
                dw, dh = (iw - nw) // 2, (ih - nh) // 2

                bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / scale
                bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / scale

            output[b][:, :4] = bboxes

    return output

def get_color_map():
    """
    Create color map.

    Returns: numpy array.

    """
    color_map = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            color_map[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return color_map

def draw_bbox(image, bboxes, labels, class_names, scores=None, show_name=False):
    """
    Draw bounding box in image.
    Args:
        image: image
        bboxes: coordinate of bounding box
        labels: the index of labels
        class_names: the names of class
        scores: bounding box confidence
        show_name: show class name if set true, otherwise show index of class

    Returns: draw result

    """
    color_map = get_color_map()
    image_height, image_width = image.shape[:2]
    draw_image = image.copy()

    for i, c in list(enumerate(labels)):
        bbox = bboxes[i]
        c = int(c)
        color = [int(j) for j in color_map[c]]
        if show_name:
            predicted_class = class_names[c]
        else:
            predicted_class = c

        if scores is None:
            text = '{}'.format(predicted_class)
        else:
            score = scores[i]
            text = '{} {:.2f}'.format(predicted_class, score)

        x1, y1, x2, y2 = bbox

        x1 = max(0, np.floor(x1).astype(np.int32))
        y1 = max(0, np.floor(y1).astype(np.int32))
        x2 = min(image_width, np.floor(x2).astype(np.int32))
        y2 = min(image_height, np.floor(y2).astype(np.int32))

        thickness = int((image_height + image_width) / (np.sqrt(image_height**2 + image_width**2)))
        fontScale = 0.35

        t_size = cv2.getTextSize(text, 0, fontScale, thickness=thickness * 2)[0]
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), color=color, thickness=thickness)
        cv2.rectangle(draw_image, (x1, y1), (x1 + t_size[0], y1 - t_size[1]), color, -1)  # filled
        cv2.putText(draw_image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale, (255, 255, 255), thickness//2, lineType=cv2.LINE_AA)

    return draw_image


def get_summary_image(images, input_shape, category, thresh,
                      hms_true, whs_true, offsets_true,
                      hms_pred, whs_pred, offsets_pred, device):
    summary_images = []
    
    outputs_true = postprocess_output(hms_true, whs_true, offsets_true, 0.999, device)
    outputs_true = decode_bbox(outputs_true, (input_shape[1], input_shape[0]), device, need_nms=True, nms_thres=0.4)
    outputs_pred = postprocess_output(hms_pred, whs_pred, offsets_pred, thresh, device)
    outputs_pred = decode_bbox(outputs_pred, (input_shape[1], input_shape[0]), device, need_nms=True, nms_thres=0.4)
    images = images.cpu().numpy()
    for i in range(len(images)):
        image = images[i]
        image = recover_input(image.copy())
        output_true = outputs_true[i]
        output_pred = outputs_pred[i]
        if len(output_true) != 0:
            output_true = output_true.data.cpu().numpy()
            labels_true = output_true[:, 5]
            bboxes_true = output_true[:, :4]
        else:
            labels_true = []
            bboxes_true = []
        
        if len(output_pred) != 0:
            output_pred = output_pred.data.cpu().numpy()
            labels_pred = output_pred[:, 5]
            bboxes_pred = output_pred[:, :4]
        else:
            labels_pred = []
            bboxes_pred = []
        
        image_true = draw_bbox(image, bboxes_true, labels_true, category, show_name=True)
        image_pred = draw_bbox(image, bboxes_pred, labels_pred, category, show_name=True)
        summary_images.append(np.hstack((image_true, image_pred)).astype(np.uint8))

    return summary_images


# The following tool functions are used for dino training
# mainly copy-paste from DINO
    
import json
import torch
import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_dict(args, model_without_ddp: nn.Module):
    try:
        param_dict_type = args.param_dict_type
    except:
        param_dict_type = 'default'
    assert param_dict_type in ['default', 'ddetr_in_mmdet', 'large_wd']

    # by default
    if param_dict_type == 'default':
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            }
        ]
        return param_dicts

    if param_dict_type == 'ddetr_in_mmdet':
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                        if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() 
                        if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() 
                        if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            }
        ]        
        return param_dicts

    if param_dict_type == 'large_wd':
        param_dicts = [
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr,
                    "weight_decay": 0.0,
                }
            ]

    return param_dicts

def create_directories(base_path: str, subdirs: list):
    
    for subdir in subdirs:
        dir_path = os.path.join(base_path, subdir)
        os.makedirs(dir_path, exist_ok=True)
    return [os.path.join(base_path, subdir) for subdir in subdirs]


def save_model(model, epoch, weight_path, optimizer=None):
    """
    save params and traning status of the model 
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    model_filename = f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, os.path.join(weight_path, model_filename))
    
def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k,v in item.items()}
    else:
        raise NotImplementedError("Call Shilong if you use other containers! type: {}".format(type(item)))
    
def visualize_and_save(res_info, image_name, image, save_path, epoch, vis = False):
    """
    可视化并保存图像，绘制预测框与真实框
    :param res_info: 预测框信息 (Tensor) [x_min, y_min, x_max, y_max, score, label]
    :param image_name: 图像名称
    :param image: 图像数据 (ndarray, RGB格式)
    :param save_path: 保存路径
    """
    image_bgr = image[:, :, ::-1]  # RGB -> BGR
    image_bgr = image_bgr.astype(np.uint8)

    # draw predicted bounding boxes
    for box in res_info:
        x_min, y_min, x_max, y_max, score, label = box.tolist()
        if score > 0.3:
            color = (0, 0, 255) 
            cv2.rectangle(image_bgr, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            cv2.putText(image_bgr, f"Pred: {int(label)}: {score:.2f}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if vis: 
        cv2.imshow(f"Image: {image_name}", image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    save_path = os.path.join(save_path, f"epoch_{epoch}_{image_name}.png")
    cv2.imwrite(save_path, image_bgr)
    
def deprocess_input(image):
    # 逆标准化：从 (C, H, W) 到 (H, W, C)，并恢复原始的像素值
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]

    # 先转换回 (H, W, 3)
    image = np.transpose(image, (1, 2, 0))

    # 逆归一化：image * std + mean
    image = image * std + mean  # 恢复到[0, 1]
    
    # 逆标准化：恢复到[0, 255]之间的像素值
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)  # 确保在 [0, 255] 范围内并转换为 uint8 类型

    return image