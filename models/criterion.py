# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

import torch 
import torch.nn.functional as F
from torch import nn
import os
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from collections import deque
from models.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from datasets.scannet200.owis_splits import KNOWN_CLASSES_IDS, PREV_KNOWN_CLASSES_IDS

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, train_is_true,num_classes, 
                 matcher, 
                 weight_dict, 
                 eos_coef, 
                 losses,
                 num_points, 
                 oversample_ratio, 
                 importance_sample_ratio,
                 class_weights, 
                 num_querries,
                 hidden_dim, 
                 store_path,
                 clustering_start_iter, 
                 clustering_update_mu_iter, 
                 enable_baseline_clustering, 
                 enable_multiple_stage_clustering,
                 clustering_momentum, 
                 store_size, 
                 OW_task,
                 split,
                 train_oracle,
                 save_dir,
                 margin,
                 finetune,
                 baseline_finetune,
                 use_obj_loss):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        
        global KNOWN_CLASSES_IDS
        

        
        self.train_is_true = train_is_true
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        if self.class_weights != -1:
            assert len(self.class_weights) == self.num_classes, "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)

        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.OW_task = OW_task
        self.num_seen_cls = 200
        self.num_querries = num_querries
        self.store_size = store_size
        self.finetune = finetune
        self.baseline_finetune = baseline_finetune
        self.use_obj_loss = use_obj_loss
        
        self.PRED_IDS = KNOWN_CLASSES_IDS[split][OW_task]
        self.PREV_IDS = PREV_KNOWN_CLASSES_IDS[split][OW_task]
        if 200 not in self.PRED_IDS:
            self.PRED_IDS.append(200)
        
           
        self.store = Queue((self.num_seen_cls+1,self.store_size,hidden_dim), self.PRED_IDS,save_dir)
        # self.means =  self.store.get_means()
        self.clustering_start_iter = clustering_start_iter
        self.clustering_update_mu_iter = clustering_update_mu_iter
        self.enable_baseline_clustering = enable_baseline_clustering
        self.enable_multiple_stage_clustering = enable_multiple_stage_clustering
        self.save_dir = save_dir
        self.clustering_momentum = clustering_momentum
        self.train_oracle = train_oracle
        self.margin = margin
        if os.path.exists(self.save_dir+'/classes_means.pkl'):
            self.means = torch.load(self.save_dir+'/classes_means.pkl').detach().cpu()
        
        if self.finetune:
            self.means = torch.tensor([])
        self.kld_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
    
    #########################
    ####contrastive loss#####
    #########################
    def get_clustering_loss(self, L_queries,outputs, targets, indices, iter):
        device = targets[0]['labels'].device
        if (not self.enable_baseline_clustering) or self.finetune:
            return {"c_loss": torch.tensor(0.0).to(device)}
        c_loss = torch.tensor(0.0).to(device)
        if self.finetune:
            c_loss = self.clstr_loss_l2_cdist(L_queries, targets, indices)
        else:
            if (iter > self.clustering_start_iter//2) and (iter < self.clustering_start_iter):
                self.store.update_store(outputs, targets, indices)
            elif iter == self.clustering_start_iter:
                try:
                    self.means =  self.store.get_means().to(device)
                    c_loss = self.clstr_loss_l2_cdist(L_queries, targets, indices)
                except:
                    pass
                self.store.update_store(outputs, targets, indices)
                
            elif iter > self.clustering_start_iter:
                if iter % self.clustering_update_mu_iter == 0:
                    try:
                        self.means = self.clustering_momentum*self.means.to(device)+(1-self.clustering_momentum)*self.store.get_means().to(device)
                        torch.save(self.means, self.save_dir+'/classes_means.pkl')

                    except:
                        try:
                            self.means = self.store.get_means() #when resuming the training the and the classes_means.pkl file is deleted the sel.means will be initilized from the current features 
                        except:
                            pass
        
        
                try:
                    c_loss = self.clstr_loss_l2_cdist(L_queries, targets, indices)
                except:
                    pass
                self.store.update_store(outputs, targets, indices)
    

        return {"c_loss":  c_loss} 
            
    
    def clstr_loss_l2_cdist(self, L_queries, targets, indices):
        """
        Get the foreground input_features, generate distributions for the class,
        get probability of each feature from each distribution;
        Compute loss: if belonging to a class -> likelihood should be higher
                      else -> lower
        :param input_features:
        :param proposals:
        :return:
        """
        device = targets[0]['labels'].device

        for batch_id, (map_id, target_id) in enumerate(indices):
            tg_labels = targets[batch_id]['labels'][target_id] if batch_id == 0 else torch.cat([tg_labels,targets[batch_id]['labels'][target_id]])
            ref_qerries = L_queries[batch_id][map_id] if batch_id == 0 else torch.cat([ref_qerries,L_queries[batch_id][map_id]])
            
        
        if self.store.prev_means != None:
            distances = torch.cdist(ref_qerries, torch.cat([self.store.prev_means,self.means.to(device)]), p=2)
            LABELS_LIST = self.PREV_IDS+self.PRED_IDS
        else:
            distances = torch.cdist(ref_qerries, self.means.to(device), p=2)
            LABELS_LIST = self.PRED_IDS
            
        # distances[:,-1] = 100*distances[:,-1]
        cc_labels = []    
        for index in range(ref_qerries.shape[0]):
            for cls_index in LABELS_LIST:
                if  tg_labels[index] ==  cls_index:
                    cc_labels.append(1)
                else:
                    cc_labels.append(-1)
                            

        loss = F.hinge_embedding_loss(distances, torch.tensor(cc_labels).reshape((-1,len(LABELS_LIST))).to(device),reduction='mean', margin=self.margin)
        
        return loss    
    #########################
    ####contrastive loss#####
    #########################
    def loss_obj(self, outputs, targets, indices, num_masks, mask_type):
        if not self.use_obj_loss:
            losses = {"_loss_obj": torch.tensor(0.0)}
        else:    
            __loss_obj = []
            for batch_id, (map_id, target_id) in enumerate(indices):
                __loss_obj.append(self.kld_loss((outputs['obj'][batch_id])[map_id].log(),torch.ones_like((outputs['obj'][batch_id])[map_id])))

            _loss_obj = sum(__loss_obj)
            
            losses = {"_loss_obj": _loss_obj}
        
        return losses

    
    def loss_labels(self, outputs, targets, indices, num_masks, mask_type):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=253)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, mask_type):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id][mask_type][target_id]
            
            
            if self.num_points != -1:
                point_idx = torch.randperm(target_mask.shape[1], 
                                        device=target_mask.device)[:int(self.num_points*target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(target_mask.shape[1], device=target_mask.device)
            map = map[:, point_idx]
            device = map.device
            target_mask = target_mask[:, point_idx].float()
            if self.train_is_true:
                if self.train_oracle:
                    ignore_masks = targets[batch_id]['labels'][target_id] != 253 
                else:
                    ignore_masks = (targets[batch_id]['labels'][target_id] != 253)*(targets[batch_id]['labels'][target_id] != 200)
                    
                target_mask = target_mask[ignore_masks]
                map = map[ignore_masks]
                num_masks = target_mask.shape[0]
                
            if num_masks!=0: #only for scenes which have the common class
                loss_masks.append(sigmoid_ce_loss_jit(map,
                                                    target_mask,
                                                    num_masks))
                loss_dices.append(dice_loss_jit(map,
                                                target_mask,
                                                num_masks))
        if loss_masks: 
        # del target_mask
            return {
                "loss_mask": torch.sum(torch.stack(loss_masks)),
                "loss_dice": torch.sum(torch.stack(loss_dices))
            }
        else: #if no scene in the batch has common or unknown
            return {
                "loss_mask": torch.tensor(0.0).to(device),
                "loss_dice": torch.tensor(0.0).to(device)
            }

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t[mask_type] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks, mask_type),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks, mask_type),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, mask_type):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'objectness': self.loss_obj
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, mask_type)

    def forward(self, outputs, targets, mask_type, iteration):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        
        indices = self.matcher(outputs_without_aux, targets, mask_type)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, mask_type))
            
        if not self.enable_multiple_stage_clustering:
            losses.update(self.get_clustering_loss(outputs["refin_queries"], outputs, targets, indices, iteration))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, mask_type)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, mask_type)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                
                if self.enable_multiple_stage_clustering: 
                    if (i+1)%3==0:   
                        l_dict = {"c_loss" + f"_{i}": self.get_clustering_loss(outputs["mul_queries"][:,i,:,:],outputs, targets, indices, iteration)["c_loss"]}
                        losses.update(l_dict)
                
        return losses
           

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    
    def get_indices(self, outputs, targets, mask_type):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        return self.matcher(outputs_without_aux, targets, mask_type)


class Queue:
    def __init__(self, store_cap, seen_cls, save_dir):
        
        self.seen_cls = seen_cls
        self.num_seen_cls = store_cap[0]
        self.size_per_cls = store_cap[1]
        self.store = [deque(maxlen=self.size_per_cls) for _ in range(self.num_seen_cls)]
        self.save_dir = save_dir
        self.task = save_dir.split('/')[-1]
        finetune_exists = "finetune" in self.task.split('_')
        if not finetune_exists:
            prev_task = self.task.replace(self.task[-1], str(int(self.task[-1])-1))
        else:
            prev_task = self.task.replace("_finetune", "")
        self.path_to_prev_means = save_dir.replace(self.task, prev_task)+"/"+prev_task+"_classes_means.pkl"
        if os.path.exists(self.path_to_prev_means):
            if not finetune_exists:
                self.prev_means = torch.load(self.path_to_prev_means)[:-1]
            else:
                self.prev_means = torch.load(self.path_to_prev_means)
        else:
            self.prev_means = None
            

            
    def update_store(self, outputs, targets , indices):
        self.device = outputs['pred_logits'].device
        for batch_id, (map_id, target_id) in enumerate(indices): 
            #known
            tg_labels = targets[batch_id]['labels'][target_id].detach().cpu()
            ref_qerries = outputs['refin_queries'][batch_id][map_id].detach().cpu()
            all_ref_querries = outputs['refin_queries'][batch_id].detach().cpu()
            label_list = torch.unique(tg_labels).detach().cpu().tolist()
            if 198 in label_list:
                label_list.remove(198)
            if 253 in label_list:
                label_list.remove(253)
                
            for tg in label_list:
                self.store[tg].append(ref_qerries[tg_labels==tg])

    def get_store(self):
        return self.store

    def get_means(self):
        means = [torch.mean(torch.cat(list(self.store[i])).float(), dim=0) for i in self.seen_cls]  
        curr_means = torch.stack(means).to(self.device)
        
        if self.prev_means != None:
            torch.save(torch.cat([self.prev_means, curr_means]), self.save_dir+"/"+self.task+"_classes_means.pkl")
        else:
            torch.save(curr_means, self.save_dir+"/"+self.task+"_classes_means.pkl")
        return curr_means
