import pdb 
from overrides import overrides
import torch 
import torch.nn.functional as F
from allennlp.common.registrable import Registrable
from allennlp.nn import util 
class Loss(torch.nn.Module, Registrable):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, debug_answer, label_weights):
        raise NotImplementedError

@Loss.register("ce")
class CELoss(Loss):
    def __init__(self):
        super().__init__()
        self.loss_fxn = F.cross_entropy

    @overrides 
    def forward(self, logits, labels, debug_answer, label_weights=None):
        return self.loss_fxn(logits, labels), label_weights, None

@Loss.register("bce")
class BCELoss(Loss):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab 
        self.loss_fxn = F.binary_cross_entropy_with_logits

    def get_label_weights(self, logits, labels, label_weights):
        """
        get label weights for BCE loss from per-answer counts 
        """

        # pdb.set_trace() 
        # try with 1 here, don't compute loss for UNKs 
        # pdb.set_trace() 
        label_mask = labels > 1  # 1 is OOV, -1 is pad
        # label_weights[label_weights > 0] = 1
        weighted_labels = util.masked_index_replace(
            logits.new_zeros(logits.size() + (1,)),
            labels.clamp(min=0).long(),
            label_mask,
            label_weights.unsqueeze(-1),
        ).squeeze(-1)

        binary_label_mask = weighted_labels.new_ones(logits.size())
        # don't compute loss for padding tokens 
        # binary_label_mask[weighted_labels > 0] = 1
        return weighted_labels, binary_label_mask 

    @overrides 
    def forward(self, logits, labels, debug_answer, label_weights=None):
        labels = labels.squeeze(1).float()
        label_weight_tensor, binary_mask = self.get_label_weights(logits, labels, label_weights)
        batch_size = labels.shape[0]
        # loss = self.loss_fxn(logits, label_weight_tensor, weight=binary_mask, reduction="sum")/batch_size
        loss = self.loss_fxn(logits, label_weight_tensor, weight=binary_mask, reduction="mean") * label_weight_tensor.size(1)
                # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19
                # https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L316
        return loss, label_weight_tensor, binary_mask

@Loss.register("bce_ce")
class CEAndBCELoss(BCELoss):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.bce_loss_fxn = F.binary_cross_entropy_with_logits
        self.ce_loss_fxn = F.cross_entropy


    def get_ce_labels(self, bce_labels, label_weights): 
        mc_answer_idx = label_weights.argmax(dim=-1)
        mc_answer_idx = mc_answer_idx.unsqueeze(-1)
        ce_labels = torch.gather(bce_labels, dim=1, index=mc_answer_idx).squeeze(-1)
        return ce_labels.long()

    @overrides 
    def forward(self, logits, bce_labels, debug_answer, label_weights=None):
        bce_labels = bce_labels.squeeze(1).float()
        bce_label_weight_tensor, binary_mask = self.get_label_weights(logits, bce_labels, label_weights)
        batch_size = bce_labels.size(0)
        # in min_gen
        if len(bce_labels.size()) == 1:
            bce_labels = bce_labels.unsqueeze(-1)

        # loss = self.loss_fxn(logits, label_weight_tensor, weight=binary_mask, reduction="sum")/batch_size
        bce_loss = self.bce_loss_fxn(logits, bce_label_weight_tensor, weight=binary_mask, reduction="mean") * bce_label_weight_tensor.size(1)
                # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19
                # https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L316

        ce_labels = self.get_ce_labels(bce_labels, label_weights)
        try:
            ce_loss = self.ce_loss_fxn(logits, ce_labels, ignore_index=-1, reduction="mean")
        except RuntimeError:
            pdb.set_trace() 
        loss = bce_loss + ce_loss
        return loss, bce_label_weight_tensor, binary_mask


@Loss.register("wbce")
class WeightedBCELoss(BCELoss):
    def __init__(self,
                vocab, 
                temperature: float = 1.0): 
        super().__init__(vocab)
        self.eps = 1e-10
        self.temperature = temperature
        # temperature [0, 1]
        # temperature of 1 means BCE, temperature of 0 means full reweighting 

    @overrides
    def get_label_weights(self, logits, labels, label_weights):
        """
        get label weights for W-BCE loss from per-answer counts 
        """

        label_mask = labels > 1  # 1 is OOV, -1 is pad
        weighted_labels = util.masked_index_replace(
            logits.new_zeros(logits.size() + (1,)),
            labels.clamp(min=0).long(),
            label_mask,
            label_weights.unsqueeze(-1),
        ).squeeze(-1)

        pos_binary_label_mask = weighted_labels.new_zeros(logits.size())
        neg_binary_label_mask = weighted_labels.new_zeros(logits.size())

        pos_binary_label_mask[weighted_labels > 0] = 1
        neg_binary_label_mask[weighted_labels == 0] = 1

        return weighted_labels, pos_binary_label_mask, neg_binary_label_mask

    @overrides 
    def forward(self, logits, labels, debug_answer, label_weights=None):
        labels = labels.squeeze(1).float()
        binary_labels = labels > 0
        temp = 1 - self.temperature + self.eps
        count_pos = torch.sum(binary_labels)+self.eps
        count_neg = torch.sum(~binary_labels)
        ratio = min(1.0, count_pos / (temp * (count_pos + count_neg))) 
        # the lower the ratio, the higher the weight 
        weight_pos = 1.0 / ((1-ratio)+self.eps)
        # weight_pos = 1.0
        # the lower the ratio, the lower the weight 
        weight_neg = 1.0
        print(f"\ntemp: {temp}\ncount_pos: {count_pos}\ncount_neg: {count_neg}\nbeta: {ratio}\nweight_pos: {weight_pos}\nweight_neg: {weight_neg}")

        label_weight_tensor, pos_binary_mask, neg_binary_mask = self.get_label_weights(logits, labels, label_weights) 
        batch_size = labels.shape[0]



        loss_pos =  self.loss_fxn(logits, label_weight_tensor, weight=pos_binary_mask, reduction="sum")/batch_size
        loss_neg =  self.loss_fxn(logits, label_weight_tensor, weight=neg_binary_mask, reduction="sum")/batch_size
        # print(f"loss_pos: {loss_pos}, loss_neg: {loss_neg}, weight_pos: {weight_pos}") 
        loss = weight_pos * loss_pos + weight_neg * loss_neg
        print(f"loss: {loss}") 
        return loss, label_weight_tensor, pos_binary_mask + neg_binary_mask

@Loss.register("dro")
class DROBCELoss(WeightedBCELoss):
    def __init__(self,
                vocab): 
        super().__init__(vocab)

    @overrides 
    def forward(self, logits, labels, debug_answer, label_weights=None):
        labels = labels.squeeze(1).float()

        label_weight_tensor, pos_binary_mask, neg_binary_mask = self.get_label_weights(logits, labels, label_weights) 
        batch_size = labels.shape[0]

        loss_pos =  self.loss_fxn(logits, label_weight_tensor, weight=pos_binary_mask, reduction="sum")/batch_size
        loss_neg =  self.loss_fxn(logits, label_weight_tensor, weight=neg_binary_mask, reduction="sum")/batch_size
        print(f"loss_pos: {loss_pos}, loss_neg: {loss_neg}") 
        loss = torch.max(loss_pos, loss_neg)
        print(f"loss: {loss}") 
        return loss, label_weight_tensor, pos_binary_mask + neg_binary_mask

@Loss.register("multilabel_ce")
class MultilabelCELoss(Loss):
    def __init__(self, vocab):
        """manually compute CE loss against multi-hot dist"""
        super().__init__() 
        self.vocab = vocab

    def get_soft_labels(self, labels, debug_answers): 
        weights = torch.zeros_like(labels)
        for row in range(weights.shape[0]):
            for answer, count in debug_answers[row].items():
                try:
                    ans_idx = self.vocab._token_to_index['answers'][answer]
                    weights[row, ans_idx] = count
                except KeyError:
                    # for now, just uniformly smooth this example 
                    print(f"cannot find answer in vocab: {answer}")
                    weights[row] = 1.0
        # normalize weights 
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights
    
    @overrides
    def forward(self, logits, labels, debug_answer, label_weights=None): 
        labels = labels.squeeze(1)
        labels = self.get_soft_labels(labels, debug_answer)
        logits = logits.log_softmax(dim=1)        
        loss = torch.mean(torch.sum(-labels * logits, dim=1))
        return loss, label_weights, None

@Loss.register("asym")
class AsymmetricLossMultiLabel(Loss):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, label_weights=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum(), label_weights, None
