import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou

class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = 20
        
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        TODO:
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        boxes = boxes.clone().detach()
        boxes[:, 0] = boxes[:, 0] / self.S - 0.5 * boxes[:, 2]
        boxes[:, 1] = boxes[:, 1] / self.S - 0.5 * boxes[:, 3]
        boxes[:, 2] = boxes[:, 0] / self.S + 0.5 * boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] / self.S + 0.5 * boxes[:, 3]

        return boxes

    def find_best_iou_boxes(self, box_pred_list, box_target):
        """
        TODO:
        Parameters:
        box_pred_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        4) hint: use torch.diagnoal() on results of compute_iou
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """
        box_target_xyxy = self.xywh2xyxy(box_target)

        best_iou = torch.zeros(box_target.size(0), 1).to(box_target.device)
        best_boxes = torch.zeros(box_target.size(0), 5).to(box_target.device)  # Initialize with 5 elements

        for box_pred in box_pred_list:
            box_pred_xyxy = self.xywh2xyxy(box_pred[:, :4])
            iou = compute_iou(box_pred_xyxy, box_target_xyxy)

            max_iou, max_index = iou.max(dim=1, keepdim=True)

            max_index = max_index.squeeze(-1)  # To ensure it's used as an index
            best_iou = torch.where(max_iou > best_iou, max_iou, best_iou)

            # Update best_boxes with the box that has the highest IOU
            for i in range(box_target.size(0)):
                best_boxes[i] = box_pred[max_index[i]]
        return best_iou, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        TODO:
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        class_loss = F.mse_loss(classes_pred[has_object_map], classes_target[has_object_map], reduction='sum')
        return class_loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        TODO:
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        loss = 0.0
        for pred_boxes in pred_boxes_list:
          loss += F.mse_loss(pred_boxes[~has_object_map][:, 4], torch.zeros_like(pred_boxes[~has_object_map][:, 4]), reduction='sum')

        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        TODO:
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        contain_loss = F.mse_loss(box_pred_conf, box_target_conf, reduction='sum')
        return contain_loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        TODO:
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        reg_loss = F.mse_loss(box_pred_response, box_target_response, reduction='sum')
        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        TODO:
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """

        N = pred_tensor.size(0)

        total_loss = 0.0
        inv_N = 1.0 / N
        # TODO: When you calculate the classification loss, no-object loss, regression loss, contain_object_loss
        # you need to multiply the loss with inv_N. e.g: inv_N * self.get_regression_loss(...)

        #TODO: split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        pred_boxes_list = [pred_tensor[..., i*5:i*5+5] for i in range(self.B)]  # for each B bounding boxes
        pred_cls = pred_tensor[..., self.B*5:]

        #TODO: compcute classification loss
        cls_loss = inv_N * self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)

        #TODO: compute no-object loss
        no_obj_loss = inv_N * self.get_no_object_loss(pred_boxes_list, has_object_map)

        #TODO: Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        '''
        object_mask = has_object_map.unsqueeze(-1)
        reshaped_pred_boxes = [pred_box[object_mask.expand(-1,-1,-1,5)].view(-1, 5) for pred_box in pred_boxes_list]
        reshaped_target_boxes = target_boxes[object_mask.expand(-1,-1,-1,4)].view(-1, 4)
        '''
        '''
        object_present = has_object_map.view(N, -1)
        reshaped_target_boxes = target_boxes.view(N, -1, 4)[object_present]
        reshaped_pred_boxes = [pred_boxes.view(N, -1, 5)[object_present] for pred_boxes in pred_boxes_list]
        #TODO: find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(reshaped_pred_boxes, reshaped_target_boxes)
        '''
        target_boxes_flat = target_boxes[has_object_map].view(-1, 4)
        pred_boxes_flat_list = [pred_boxes[has_object_map].view(-1, 5) for pred_boxes in pred_boxes_list]

        best_iou, best_boxes = self.find_best_iou_boxes(pred_boxes_flat_list, target_boxes_flat)

        #TODO: compute regression loss between the found best bbox and GT bbox for all the cell containing objects

        reg_loss = inv_N * self.get_regression_loss(best_boxes[:, :4], target_boxes_flat)
        '''
        reg_loss = self.get_regression_loss(best_boxes, reshaped_target_boxes)
        total_loss += inv_N * reg_loss
        '''
        #TODO: compute contain_object_loss
        '''
        contain_obj_loss = self.get_contain_conf_loss(best_boxes[..., 4], best_ious)
        total_loss += inv_N * contain_obj_loss
        '''
        contain_obj_loss = inv_N * self.get_contain_conf_loss(best_boxes[:, 4].unsqueeze(-1), best_iou)
        #TODO: compute final loss
        total_loss = reg_loss + contain_obj_loss + no_obj_loss + cls_loss
        #TODO: construct return loss_dict
        loss_dict = {
          'total_loss': total_loss,
          'reg_loss': reg_loss,
          'containing_obj_loss': contain_obj_loss,
          'no_obj_loss': no_obj_loss,
          'cls_loss': cls_loss,
        }
        return loss_dict