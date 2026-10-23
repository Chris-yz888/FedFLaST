import logging

import torch
from torch import nn
from ...pruning.init_scheme import f_decay

try:
    from core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedPruning.core.trainer.model_trainer import ModelTrainer

class MyModelTrainer(ModelTrainer):

    def get_model(self):
        return self.model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def _has_client_sam_prune_budget(self, model, round_idx, args):
        """Return whether this adjustment call will prune at least one connection."""
        if round_idx is None:
            return False
        for name, mask in model.mask_dict.items():
            active_num = (mask == 1).int().sum().item()
            k = int(f_decay(round_idx, args.adjust_alpha, args.T_end) * active_num)
            if k > 0:
                return True
        return False

    def _compute_client_sam_gradients(self, model, criterion, probe_batch, gradients, args):
        """Compute g+ for FedFLaST client-side SAM-guided pruning.

        This probe is only used for mask pruning.  It temporarily perturbs active
        sparse weights, computes one gradient on the same minibatch used for g0,
        and then restores the original weights exactly.
        """
        if probe_batch is None:
            logging.warning("[FedFLaST-ClientSAMPrune] No probe batch is available; fallback to FedDST pruning.")
            return None

        sparse_params = []
        for name, param in model.named_parameters():
            if name in model.mask_dict and name in gradients and param.requires_grad:
                sparse_params.append((name, param))

        if not sparse_params:
            logging.warning("[FedFLaST-ClientSAMPrune] No sparse gradients are available; fallback to FedDST pruning.")
            return None

        rho = getattr(args, "flast_local_rho", 0.05)
        eps = getattr(args, "flast_local_eps", 1e-12)
        norm_type = getattr(args, "flast_local_sam_norm", "l2")

        grad_norm = None
        if norm_type == "l2":
            norm_sq = None
            for name, param in sparse_params:
                masked_grad = gradients[name].to(param.device, dtype=param.dtype) * model.mask_dict[name].to(param.device, dtype=param.dtype)
                if not torch.isfinite(masked_grad).all():
                    logging.warning("[FedFLaST-ClientSAMPrune] Non-finite g0 detected; fallback to FedDST pruning.")
                    return None
                layer_norm_sq = torch.sum(masked_grad * masked_grad)
                norm_sq = layer_norm_sq if norm_sq is None else norm_sq + layer_norm_sq
            grad_norm = torch.sqrt(norm_sq)
            if (not torch.isfinite(grad_norm)) or grad_norm.item() <= eps:
                logging.warning("[FedFLaST-ClientSAMPrune] Zero SAM gradient norm; fallback to FedDST pruning.")
                return None

        original_params = {}
        original_buffers = {
            name: buf.detach().clone()
            for name, buf in model.named_buffers()
        }
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = None
        if torch.cuda.is_available():
            try:
                cuda_rng_states = torch.cuda.get_rng_state_all()
            except Exception:
                cuda_rng_states = None

        try:
            with torch.no_grad():
                for name, param in sparse_params:
                    original_params[name] = param.data.detach().clone()
                    mask = model.mask_dict[name].to(param.device, dtype=param.dtype)
                    masked_grad = gradients[name].to(param.device, dtype=param.dtype) * mask
                    if norm_type == "sign":
                        perturb = rho * masked_grad / (torch.abs(masked_grad) + eps)
                    else:
                        perturb = rho * masked_grad / (grad_norm + eps)
                    perturb = perturb * mask
                    if not torch.isfinite(perturb).all():
                        logging.warning("[FedFLaST-ClientSAMPrune] Non-finite epsilon detected; fallback to FedDST pruning.")
                        return None
                    param.data.add_(perturb)

            model.zero_grad()
            x, labels = probe_batch
            log_probs = model(x)
            loss = criterion(log_probs, labels)
            loss.backward()
            sam_gradients = {
                name: param.grad.data.cpu().clone()
                for name, param in model.named_parameters()
                if param.requires_grad and param.grad is not None
            }
        finally:
            with torch.no_grad():
                for name, param in sparse_params:
                    if name in original_params:
                        param.data.copy_(original_params[name].to(param.device))
                for name, buf in model.named_buffers():
                    if name in original_buffers:
                        buf.data.copy_(original_buffers[name].to(buf.device))
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                try:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
                except Exception:
                    pass
            model.zero_grad()
            model.apply_mask()

        return sam_gradients

    def train(self, train_data, device, args, mode, round_idx = None):

        # mode 0 :  training with mask 
        # mode 1 : training with mask 
        # mode 2 : training with mask, calculate mask
        # mode 3 : training with mask, calculate mask
        model = self.model
        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
            
        epoch_loss = []
        
        if mode in [2, 3]:
            local_epochs = args.adjustment_epochs if args.adjustment_epochs is not None else args.epochs
        else:
            local_epochs = args.epochs

        if mode in [2, 3]:
            A_epochs = local_epochs // 2 if args.A_epochs is None else args.A_epochs
            first_epochs = min(local_epochs, A_epochs)
        else:
            first_epochs = local_epochs

        for epoch in range(first_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                #self.model.apply_mask_gradients()  # apply pruning mask
                
                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        if mode in [2, 3]:
            model.zero_grad()
            sam_probe_batch = None
            if args.growth_data_mode == "random":
                gradients = {name: torch.randn_like(param, device='cpu').clone() for name, param in model.named_parameters() if param.requires_grad}

            elif args.growth_data_mode == "single":
                x, labels = next(iter(train_data))
                x, labels = x[0].unsqueeze(0).repeat(2, 1, 1, 1).to(device), labels[0].unsqueeze(0).repeat(2).to(device)  # Duplicate the sample to create a pseudo-batch
                sam_probe_batch = (x, labels)
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
                model.zero_grad()
            else:
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    if sam_probe_batch is None:
                        sam_probe_batch = (x, labels)
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    if args.growth_data_mode == "batch":
                        break
                gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
                model.zero_grad()

            # pruning and growing
            sam_gradients = None
            if getattr(args, "use_flast_client_sam_prune", False):
                if args.growth_data_mode not in ["batch", "single"]:
                    logging.warning(
                        "[FedFLaST-ClientSAMPrune] growth_data_mode=%s is not a same-minibatch probe mode; fallback to FedDST pruning.",
                        args.growth_data_mode,
                    )
                elif not self._has_client_sam_prune_budget(model, round_idx, args):
                    logging.info(
                        "[FedFLaST-ClientSAMPrune] client=%s round=%s has no positive prune budget; skip SAM probe.",
                        self.id,
                        round_idx,
                    )
                else:
                    sam_gradients = self._compute_client_sam_gradients(
                        model, criterion, sam_probe_batch, gradients, args
                    )
                    if sam_gradients is not None:
                        logging.info(
                            "[FedFLaST-ClientSAMPrune] client=%s round=%s rho=%s norm=%s candidate_multiplier=%s",
                            self.id,
                            round_idx,
                            getattr(args, "flast_local_rho", None),
                            getattr(args, "flast_local_sam_norm", None),
                            getattr(args, "flast_candidate_multiplier", None),
                        )

            model.adjust_mask_dict(
                gradients,
                t=round_idx,
                T_end=args.T_end,
                alpha=args.adjust_alpha,
                sam_gradients=sam_gradients,
                sam_prune_candidate_multiplier=getattr(args, "flast_candidate_multiplier", 2.0),
                sam_prune_score_mode="sam_max",
                sam_prune_local_rho=getattr(args, "flast_local_rho", 0.05),
                sam_prune_local_eps=getattr(args, "flast_local_eps", 1e-12),
                sam_prune_local_norm=getattr(args, "flast_local_sam_norm", "l2"),
            )
            model.apply_mask()

        for epoch in range(first_epochs, local_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        return model.mask_dict

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'Accuracy': 0,
            'Loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['Accuracy'] += correct.item()
                metrics['Loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        
        metrics['Accuracy'] /= metrics['test_total'] 
        metrics['Loss'] /= metrics['test_total'] 
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
