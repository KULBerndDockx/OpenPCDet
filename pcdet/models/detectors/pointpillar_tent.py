import torch
import torch.nn as nn
import torch.optim as optim
import copy

from .pointpillar import PointPillar


class PointPillar_TENT(PointPillar):
    """
    PointPillar with TENT (Test Entropy miNimization) adaptation.
    
    At test time, TENT:
    1. Sets BN layers (in selected modules) to train mode so they use batch
       statistics instead of stale running statistics from the source domain.
    2. Freezes all parameters except BN affine parameters (gamma, beta).
    3. Optimizes BN affine parameters to minimize the entropy of model predictions,
       adapting the model to the target distribution on-the-fly.
    
    By default, only the BACKBONE_2D BN layers are adapted. The VFE's BN1d operates
    on per-voxel point features with highly variable counts — switching it to batch
    statistics with small test batch sizes causes catastrophic feature degradation.
    
    During training, this behaves identically to the standard PointPillar.
    """

    # Map from config names to the attribute names on Detector3DTemplate
    _MODULE_NAME_MAP = {
        'vfe': 'vfe',
        'backbone_3d': 'backbone_3d',
        'map_to_bev_module': 'map_to_bev_module',
        'backbone_2d': 'backbone_2d',
        'dense_head': 'dense_head',
    }

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        # TENT hyperparameters from config (with sensible defaults)
        tent_cfg = self.model_cfg.get('TENT', None)
        if tent_cfg is not None:
            self.tent_lr = tent_cfg.get('LR', 0.00001)
            self.tent_steps = tent_cfg.get('STEPS', 1)
            self.tent_episodic = tent_cfg.get('EPISODIC', True)
            self.tent_enabled = tent_cfg.get('ENABLED', True)
            # Which sub-modules to adapt. Default: only backbone_2d
            self.tent_adapt_modules = tent_cfg.get('ADAPT_MODULES', ['backbone_2d'])
        else:
            self.tent_lr = 0.00001
            self.tent_steps = 1
            self.tent_episodic = True
            self.tent_enabled = True
            self.tent_adapt_modules = ['backbone_2d']

        # Resolve the actual nn.Module references for the modules to adapt
        self._tent_target_modules = []
        for mod_name in self.tent_adapt_modules:
            attr_name = self._MODULE_NAME_MAP.get(mod_name, mod_name)
            mod = getattr(self, attr_name, None)
            if mod is not None:
                self._tent_target_modules.append(mod)

        # Store a copy of the original BN state for episodic resets
        # (only BN params in target modules, to avoid large memory overhead)
        if self.tent_episodic:
            self._tent_anchor_state = copy.deepcopy(self._collect_bn_state())

    # ------------------------------------------------------------------
    # BN utility methods (scoped to target modules only)
    # ------------------------------------------------------------------

    def _iter_target_bn_modules(self):
        """Iterate over BN layers only within the modules selected for adaptation."""
        for target_mod in self._tent_target_modules:
            for module in target_mod.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    yield module

    def _iter_target_bn_named_modules(self):
        """Iterate over (name, BN module) within the modules selected for adaptation."""
        for target_mod in self._tent_target_modules:
            for name, module in target_mod.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    yield name, module

    def _collect_bn_params(self):
        """Collect BN affine parameters (weight=gamma, bias=beta) from target modules."""
        params = []
        for module in self._iter_target_bn_modules():
            if module.affine:
                params.append(module.weight)  # gamma
                params.append(module.bias)    # beta
        return params

    def _collect_bn_state(self):
        """Snapshot the state of BN layers in target modules (for episodic reset)."""
        bn_state = {}
        for name, module in self._iter_target_bn_named_modules():
            bn_state[name] = copy.deepcopy(module.state_dict())
        return bn_state

    def _restore_bn_state(self, bn_state):
        """Restore BN layers in target modules to a previous snapshot."""
        for name, module in self._iter_target_bn_named_modules():
            if name in bn_state:
                module.load_state_dict(bn_state[name])

    def _configure_tent_mode(self):
        """
        Configure model for TENT:
        - Set entire model to eval (freezes dropout, uses running stats everywhere)
        - Freeze all parameters
        - Then selectively set BN layers in target modules to train mode
          and unfreeze their affine parameters
        """
        # Start from eval mode globally (disables dropout, BN uses running stats)
        self.eval()

        # Freeze everything
        for param in self.parameters():
            param.requires_grad_(False)

        # Only adapt BN layers in the selected target modules
        for module in self._iter_target_bn_modules():
            module.train()  # Use batch statistics
            module.track_running_stats = False  # Don't update running stats
            if module.affine:
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

    # ------------------------------------------------------------------
    # Entropy loss
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax_entropy(cls_preds):
        """
        Compute entropy of softmax predictions.
        
        Args:
            cls_preds: (B, num_anchors, num_classes) raw logits
        Returns:
            mean entropy (scalar)
        """
        probs = torch.softmax(cls_preds, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropy.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_dict):
        if self.training:
            # Standard training — identical to PointPillar
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)

            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict

        else:
            # ---------- TEST-TIME ADAPTATION (TENT) ----------
            if not self.tent_enabled:
                # If TENT is disabled, fall back to standard inference
                for cur_module in self.module_list:
                    batch_dict = cur_module(batch_dict)
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts

            # Episodic reset: restore BN params to anchor state before each batch
            if self.tent_episodic and hasattr(self, '_tent_anchor_state'):
                self._restore_bn_state(self._tent_anchor_state)

            # Configure: eval globally, then selectively enable TENT on target modules
            self._configure_tent_mode()

            # Create optimizer for BN affine params in target modules only
            bn_params = self._collect_bn_params()
            optimizer = optim.Adam(bn_params, lr=self.tent_lr)

            # Override the outer torch.no_grad() context from eval_utils
            with torch.enable_grad():
                for step in range(self.tent_steps):
                    # Forward pass through all modules
                    batch_dict_copy = {k: v for k, v in batch_dict.items()}
                    for cur_module in self.module_list:
                        batch_dict_copy = cur_module(batch_dict_copy)

                    # Compute entropy on the class predictions
                    cls_preds = batch_dict_copy['batch_cls_preds']
                    entropy_loss = self._softmax_entropy(cls_preds)

                    # Backward + update BN affine params
                    optimizer.zero_grad()
                    entropy_loss.backward()
                    optimizer.step()

            # Final forward pass with adapted BN params (no grad needed)
            with torch.no_grad():
                batch_dict_final = {k: v for k, v in batch_dict.items()}
                for cur_module in self.module_list:
                    batch_dict_final = cur_module(batch_dict_final)

            pred_dicts, recall_dicts = self.post_processing(batch_dict_final)
            return pred_dicts, recall_dicts
