import copy

import torch
import torch.nn as nn
import torch.optim as optim

from .second_net import SECONDNet


class SECONDNet_TENT(SECONDNet):
    """SECONDNet with TENT (Test Entropy miNimization) adaptation.

    This reuses the same TENT approach as `PointPillar_TENT`:
    - Keep the model in eval mode globally.
    - Enable BN layers (in selected sub-modules) in train mode so they use batch
      statistics.
    - Freeze all parameters except BN affine params (gamma/beta).
    - Optimize BN affine params to minimize prediction entropy on-the-fly.

    Defaults are intentionally conservative: adapt only `backbone_2d` unless
    configured otherwise.
    """

    _MODULE_NAME_MAP = {
        'vfe': 'vfe',
        'backbone_3d': 'backbone_3d',
        'map_to_bev_module': 'map_to_bev_module',
        'backbone_2d': 'backbone_2d',
        'dense_head': 'dense_head',
    }

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        tent_cfg = self.model_cfg.get('TENT', None)
        if tent_cfg is not None:
            self.tent_lr = tent_cfg.get('LR', 0.001)
            self.tent_steps = tent_cfg.get('STEPS', 10)
            self.tent_episodic = tent_cfg.get('EPISODIC', True)
            self.tent_enabled = tent_cfg.get('ENABLED', True)
            self.tent_adapt_modules = tent_cfg.get('ADAPT_MODULES', ['backbone_2d'])
        else:
            self.tent_lr = 0.001
            self.tent_steps = 10
            self.tent_episodic = True
            self.tent_enabled = True
            self.tent_adapt_modules = ['backbone_2d']

        self._tent_target_modules = []
        for mod_name in self.tent_adapt_modules:
            attr_name = self._MODULE_NAME_MAP.get(mod_name, mod_name)
            mod = getattr(self, attr_name, None)
            if mod is not None:
                self._tent_target_modules.append(mod)

        if self.tent_episodic:
            self._tent_anchor_state = copy.deepcopy(self._collect_bn_state())

    # ------------------------------------------------------------------
    # BN utilities (scoped to target modules only)
    # ------------------------------------------------------------------

    def _iter_target_bn_modules(self):
        for target_mod in self._tent_target_modules:
            for module in target_mod.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    yield module

    def _iter_target_bn_named_modules(self):
        for target_mod in self._tent_target_modules:
            for name, module in target_mod.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    yield name, module

    def _collect_bn_params(self):
        params = []
        for module in self._iter_target_bn_modules():
            if module.affine:
                params.append(module.weight)
                params.append(module.bias)
        return params

    def _collect_bn_state(self):
        bn_state = {}
        for name, module in self._iter_target_bn_named_modules():
            bn_state[name] = copy.deepcopy(module.state_dict())
        return bn_state

    def _restore_bn_state(self, bn_state):
        for name, module in self._iter_target_bn_named_modules():
            if name in bn_state:
                module.load_state_dict(bn_state[name])

    def _configure_tent_mode(self):
        self.eval()

        for param in self.parameters():
            param.requires_grad_(False)

        for module in self._iter_target_bn_modules():
            module.train()
            module.track_running_stats = False
            if module.affine:
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

    # ------------------------------------------------------------------
    # Entropy loss (sigmoid/Bernoulli, consistent with OpenPCDet dense heads)
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid_entropy(cls_logits: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        probs = torch.sigmoid(cls_logits)
        ent = -(probs * torch.log(probs + eps) + (1.0 - probs) * torch.log(1.0 - probs + eps))
        return ent.sum(dim=-1)

    @staticmethod
    def _select_entropy_anchors(
        entropy_per_anchor: torch.Tensor,
        cls_logits: torch.Tensor,
        topk: int = 0,
        score_thresh: float = -1.0,
    ) -> torch.Tensor:
        B, N = entropy_per_anchor.shape
        with torch.no_grad():
            max_score = torch.sigmoid(cls_logits).amax(dim=-1)

            if score_thresh is not None and score_thresh > 0:
                mask = max_score > float(score_thresh)
                if mask.sum() == 0:
                    return torch.ones_like(mask, dtype=torch.bool)
                return mask

            if topk is not None and int(topk) > 0 and int(topk) < N:
                k = int(topk)
                _, idx = torch.topk(max_score, k=k, dim=1, largest=True, sorted=False)
                mask = torch.zeros((B, N), device=entropy_per_anchor.device, dtype=torch.bool)
                mask.scatter_(1, idx, True)
                return mask

            return torch.ones((B, N), device=entropy_per_anchor.device, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_dict):
        if self.training:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)

            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict

        if not self.tent_enabled:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

        if self.tent_episodic and hasattr(self, '_tent_anchor_state'):
            self._restore_bn_state(self._tent_anchor_state)

        self._configure_tent_mode()

        bn_params = self._collect_bn_params()
        optimizer = optim.Adam(bn_params, lr=self.tent_lr)

        tent_cfg = self.model_cfg.get('TENT', {})
        entropy_topk = tent_cfg.get('ENTROPY_TOPK', 0)
        entropy_score_thresh = tent_cfg.get('ENTROPY_SCORE_THRESH', -1.0)
        entropy_weight = tent_cfg.get('ENTROPY_WEIGHT', 1.0)

        # Override the outer torch.no_grad() context from eval_utils
        with torch.enable_grad():
            for _step in range(self.tent_steps):
                batch_dict_copy = {k: v for k, v in batch_dict.items()}
                for cur_module in self.module_list:
                    batch_dict_copy = cur_module(batch_dict_copy)

                if 'batch_cls_preds' not in batch_dict_copy:
                    raise KeyError(
                        'TENT expected `batch_cls_preds` in batch_dict (dense head logits), '
                        'but it was not found. Check your dense head implementation.'
                    )

                cls_preds = batch_dict_copy['batch_cls_preds']
                entropy_per_anchor = self._sigmoid_entropy(cls_preds)
                mask = self._select_entropy_anchors(
                    entropy_per_anchor,
                    cls_preds,
                    topk=entropy_topk,
                    score_thresh=entropy_score_thresh,
                )
                entropy_loss = entropy_per_anchor[mask].mean() * float(entropy_weight)

                optimizer.zero_grad()
                entropy_loss.backward()
                optimizer.step()

        with torch.no_grad():
            batch_dict_final = {k: v for k, v in batch_dict.items()}
            for cur_module in self.module_list:
                batch_dict_final = cur_module(batch_dict_final)

        pred_dicts, recall_dicts = self.post_processing(batch_dict_final)
        return pred_dicts, recall_dicts
