import logging
import os, signal
import sys
import copy

from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process
from api.pruning.init_scheme import generate_layer_density_dict, pruning
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
try:
    from core.distributed.communication.message import Message
    from core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedPruning.core.distributed.communication.message import Message
    from FedPruning.core.distributed.server.server_manager import ServerManager

class FedFLaSTServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.mode = 0 

        # === FedFLaST Module 1: Mask-Projected Global Momentum Perturbation ===
        self._flast_enabled = getattr(args, 'use_fedflast_perturbation', False)
        self._flast_momentum_buffer = {}   # Dict[str, Tensor]: per-param momentum
        self._flast_prev_w = None          # w_t: original model before last perturbation
        self._flast_prev_w_tilde = None    # w_tilde: perturbed model that was broadcast
        self._flast_p_t = None             # p_t: pseudo-gradient for momentum update
        self._flast_initialized = False    # whether momentum buffer has been initialized
        self._flast_topology_enabled = getattr(args, 'use_flast_topology', False)
        logging.info(
            "[FedFLaST] perturbation=%s, rho=%s, momentum=%s, server_lr=%s, eps=%s, "
            "topology=%s, topology_lambda=%s, topology_score=%s",
            self._flast_enabled,
            getattr(args, 'flast_rho', None),
            getattr(args, 'flast_momentum', None),
            getattr(args, 'flast_server_lr', None),
            getattr(args, 'flast_eps', None),
            self._flast_topology_enabled,
            getattr(args, 'flast_topology_lambda', None),
            getattr(args, 'flast_topology_score', None),
        )

        # mode 0, the server send both weight and mask to clients, received the weight, perform weight aggregation, if t % \delta t == 0 and t <= t_end, go to mode 2, else, go to mode 1
        # mode 1, the server send weights, received the weights, perform weights aggregation, if t % \delta t == 0 and t <= t_end, go to mode 2, else, go to mode 1
        # mode 2, the server send weights, received the weights and masks, perform weights and masks aggregation, pruning and growing to produce new mask,  go to mode 0
        # TODO (special mode, only for \delta t == 1) mode 3, the server send both weight and mask to clients, received the weights and masks, perform weights and masks aggregation, pruning and growing to produce new mask,  if t < t_end, go to mode 3 ,else , go to mode 0.

    def run(self):
        super().run()

    def _save_final_checkpoint(self):
        """Save the recovered final server model together with its sparse mask."""
        if not getattr(self.args, "save_final_checkpoint", False):
            return

        checkpoint_path = getattr(self.args, "checkpoint_path", None)
        if not checkpoint_path:
            checkpoint_path = os.path.join(
                "checkpoints",
                "fedflast_{}_{}_den{:03d}_seed{}.pt".format(
                    self.args.dataset,
                    self.args.model,
                    int(round(self.args.target_density * 100)),
                    getattr(self.args, "seed", 0),
                ),
            )

        checkpoint_path = os.path.abspath(checkpoint_path)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        model = self.aggregator.trainer.model
        model.apply_mask()
        payload = {
            "format_version": 1,
            "method": "FedFLaST",
            "round": self.round_num,
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "mask_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.mask_dict.items()
            },
            "layer_density_dict": copy.deepcopy(model.layer_density_dict),
            "args": dict(vars(self.args)),
        }
        torch.save(payload, checkpoint_path)
        logging.info("[checkpoint] saved final FedFLaST model to %s", checkpoint_path)

    def send_init_msg(self):
        # FedFLaST: initialize momentum buffer on first call
        if (self._flast_enabled or self._flast_topology_enabled) and not self._flast_initialized:
            self._flast_init_momentum()

        # sampling clients
        logging.info(f"current step is {self.round_idx} and the current mode is {self.mode}")
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)

        # FedFLaST: first round, momentum is 0, so perturbation is 0, no need to modify
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1], self.mode, self.round_idx)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                self.handle_message_receive_model_from_client)

    # ==============================================================
    # FedFLaST Module 1: Helper Methods
    # ==============================================================

    def _flast_init_momentum(self):
        """Initialize server-side momentum buffer for prunable (masked) parameters.
        
        NOTE: mask_dict keys are like 'conv1.weight' (inner model keys),
        but state_dict keys are like 'model.conv1.weight' (SparseModel wraps inner model).
        We use state_dict format for momentum buffer keys to match w_t/w_tilde/w_aggregated.
        """
        mask_dict = self.aggregator.trainer.model.mask_dict
        device = self.aggregator.device

        for name, mask in mask_dict.items():
            state_key = f"model.{name}"  # Match state_dict key format
            self._flast_momentum_buffer[state_key] = torch.zeros(
                mask.shape, dtype=mask.dtype, device=device
            )
        self._flast_initialized = True
        logging.info(f"[FedFLaST] Momentum buffer initialized for {len(self._flast_momentum_buffer)} prunable parameters.")
        logging.info(f"[FedFLaST] Device: {device}")

    @torch.no_grad()
    def _flast_compute_perturbation(self, mask_dict):
        """
        Compute mask-projected perturbation delta.
        
        d_t = m_t ⊙ u_{t-1}
        delta_t = rho * d_t / (||d_t||_2 + eps)   if ||d_t||_2 >= eps else 0
        
        Returns:
            delta: Dict[str, Tensor] perturbation per parameter (state_dict keys)
            active_norm: float, ||d_t||_2
        """
        rho = self.args.flast_rho
        eps = self.args.flast_eps
        delta = {}
        squared_norm = 0.0

        for name, mask in mask_dict.items():
            state_key = f"model.{name}"  # Convert mask key to state_dict key
            if state_key not in self._flast_momentum_buffer:
                continue
            mom = self._flast_momentum_buffer[state_key]
            mask_dev = mask.to(mom.device, dtype=mom.dtype)
            projected = mom * mask_dev
            squared_norm += torch.sum(projected * projected).item()
            delta[state_key] = projected

        active_norm = squared_norm ** 0.5

        if active_norm < eps:
            for name in delta:
                delta[name] = torch.zeros_like(delta[name])
        else:
            scale = rho / (active_norm + eps)
            for name in delta:
                delta[name] = delta[name] * scale

        return delta, active_norm

    def _flast_build_broadcast_model(self, w_current, delta):
        """Build w_tilde = w_current + delta (without modifying w_current)."""
        w_tilde = {}
        for k, v in w_current.items():
            if k in delta:
                w_tilde[k] = v + delta[k].to(v.device)
            else:
                w_tilde[k] = v.clone().detach()
        return w_tilde

    @torch.no_grad()
    def _flast_recover_original_model(self, w_t, w_tilde, w_aggregated):
        """
        Recover w_{t+1} = w_t + server_lr * (w_aggregated - w_tilde).
        Returns w_{t+1} state dict.
        """
        server_lr = self.args.flast_server_lr
        w_t1 = {}
        for k in w_t:
            if k in w_tilde and k in w_aggregated and k in self._flast_momentum_buffer:
                client_update = w_aggregated[k] - w_tilde[k]
                w_t1[k] = w_t[k] + server_lr * client_update.to(w_t[k].device)
            elif k in w_aggregated:
                w_t1[k] = w_aggregated[k]
            else:
                w_t1[k] = w_t[k]
        return w_t1

    @torch.no_grad()
    def _flast_compute_pseudo_gradient(self, w_t, w_t1):
        """p_t = (w_t - w_{t+1}) / (server_lr + eps)."""
        server_lr = self.args.flast_server_lr
        eps = self.args.flast_eps
        p_t = {}
        for k in w_t:
            if k in w_t1 and k in self._flast_momentum_buffer:
                p_t[k] = (w_t[k] - w_t1[k]) / (server_lr + eps)
        return p_t

    @torch.no_grad()
    def _flast_update_momentum(self, p_t, m_t1):
        """
        u_t = m_{t+1} ⊙ (mu * u_{t-1} + p_t).
        m_t1 keys are mask_dict format (e.g. 'conv1.weight'),
        momentum buffer keys are state_dict format (e.g. 'model.conv1.weight').
        """
        mu = self.args.flast_momentum
        for k, mom in list(self._flast_momentum_buffer.items()):
            # Strip 'model.' prefix to look up in mask_dict
            mask_key = k[6:] if k.startswith("model.") else k
            if mask_key in m_t1:
                mask_k = m_t1[mask_key].to(mom.device, dtype=mom.dtype)
                p_k = p_t.get(k, torch.zeros_like(mom)).to(mom.device)
                self._flast_momentum_buffer[k] = mask_k * (mu * mom + p_k)

    def _flast_log_metrics(self, active_norm, delta, p_t, round_idx):
        """Log flast diagnostics to the text logger only.

        Keep W&B logging identical to FedDST: only Test/* metrics call wandb.log.
        FedFLaST hyperparameters are still captured by wandb.init(config=args)
        and the run name in main_fedflast.py.
        """
        pert_norm = 0.0
        if delta is not None:
            for v in delta.values():
                pert_norm += torch.sum(v * v).item()
            pert_norm = pert_norm ** 0.5

        mom_norm = 0.0
        for v in self._flast_momentum_buffer.values():
            mom_norm += torch.sum(v * v).item()
        mom_norm = mom_norm ** 0.5

        pg_norm = 0.0
        if p_t is not None:
            for v in p_t.values():
                pg_norm += torch.sum(v * v).item()
            pg_norm = pg_norm ** 0.5

        def _dict_cosine(a_dict, b_dict):
            if a_dict is None or b_dict is None:
                return 0.0
            dot = 0.0
            a_sq = 0.0
            b_sq = 0.0
            for k, a in a_dict.items():
                if k not in b_dict:
                    continue
                b = b_dict[k].to(a.device)
                dot += torch.sum(a * b).item()
                a_sq += torch.sum(a * a).item()
                b_sq += torch.sum(b * b).item()
            denom = (a_sq ** 0.5) * (b_sq ** 0.5)
            if denom < 1e-12:
                return 0.0
            return dot / denom

        momentum_pg_cos = _dict_cosine(self._flast_momentum_buffer, p_t)
        delta_pg_cos = _dict_cosine(delta, p_t)
        perturb_over_pg = pert_norm / (pg_norm + 1e-12)
        momentum_over_pg = mom_norm / (pg_norm + 1e-12)

        logging.info(
            "[FedFLaST] Round %s: active_direction_norm=%.6f, perturbation_norm=%.6f, "
            "momentum_norm=%.6f, pseudo_gradient_norm=%.6f, "
            "perturb_over_pg=%.6f, momentum_over_pg=%.6f, "
            "cos(momentum,pseudo_gradient)=%.6f, cos(delta,pseudo_gradient)=%.6f",
            round_idx,
            active_norm,
            pert_norm,
            mom_norm,
            pg_norm,
            perturb_over_pg,
            momentum_over_pg,
            momentum_pg_cos,
            delta_pg_cos,
        )

    @torch.no_grad()
    def _flast_build_topology_score_dict(self, model):
        """Build per-layer topology guidance scores for FedFLaST Module 2.

        Returned keys match mask_dict / model.named_parameters() keys such as
        'conv1.weight'.  Momentum and pseudo-gradient buffers use state_dict keys
        such as 'model.conv1.weight', so we translate them here.
        """
        score_mode = getattr(self.args, 'flast_topology_score', 'momentum')
        score_dict = {}

        for name, weight in model.named_parameters():
            state_key = f"model.{name}"
            score = None

            if score_mode in ["momentum", "mom", "u"]:
                score = self._flast_momentum_buffer.get(state_key, None)
            elif score_mode in ["pseudo_gradient", "pg", "gradient"]:
                if self._flast_p_t is not None:
                    score = self._flast_p_t.get(state_key, None)
            elif score_mode in ["w_pseudo_gradient", "w_pg", "weight_pg"]:
                if self._flast_p_t is not None and state_key in self._flast_p_t:
                    score = weight.data.to(self._flast_p_t[state_key].device) * self._flast_p_t[state_key]
            elif score_mode in ["momentum_plus_pg", "mom_pg"]:
                mom = self._flast_momentum_buffer.get(state_key, None)
                pg = self._flast_p_t.get(state_key, None) if self._flast_p_t is not None else None
                if mom is not None and pg is not None:
                    score = mom + pg.to(mom.device)
                elif mom is not None:
                    score = mom
                else:
                    score = pg
            else:
                raise Exception(f"flast topology score {score_mode} is not supported")

            if score is not None and score.shape == weight.shape:
                score_dict[name] = score.detach()

        return score_dict

    # ==============================================================

    def mode_convert(self,):
        if self.mode == 0:
            if self.round_idx % self.args.delta_T == 0 and self.round_idx <= self.args.T_end :
                self.mode = 2
            else:
                self.mode = 1
        elif self.mode == 1:
            if self.round_idx % self.args.delta_T == 0 and self.round_idx <= self.args.T_end :
                self.mode = 2
            else:
                self.mode = 1
        elif self.mode == 2:
            self.mode = 0
        elif self.mode == 3:
            if self.round_idx < self.args.T_end :
                self.mode = 3
            else:
                self.mode = 0

        return self.mode
    
    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        if self.mode in [2, 3]:
            masks = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_MASKS)
            self.aggregator.add_local_trained_mask(sender_id - 1, masks)
            
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()

            # ==================== FedFLaST Recovery Phase ====================
            # Recover w_{t+1} from saved w_t, w_tilde, and aggregated_perturbed
            if self._flast_enabled and self._flast_prev_w is not None:
                w_t1 = self._flast_recover_original_model(
                    self._flast_prev_w, self._flast_prev_w_tilde, global_model_params
                )
                # Compute pseudo-gradient before mask update
                self._flast_p_t = self._flast_compute_pseudo_gradient(self._flast_prev_w, w_t1)
                # Set model to w_{t+1}
                self.aggregator.set_global_model_params(w_t1)
                global_model_params = w_t1
                logging.info(f"[FedFLaST] Recovered w_{{{self.round_idx + 1}}} from perturbed aggregation.")
            # =================================================================

            logging.info(f"current mode for server is {self.mode}, the round is {self.round_idx}")
            if self.mode in [2, 3]:
                model = self.aggregator.trainer.model
                global_mask = self.aggregator.aggregate_mask()
                # # prune to reach density
                layer_density_strategy, pruning_strategy = model.strategy.split("_")
                topology_score_dict = None
                topology_lambda = 0.0
                if self._flast_topology_enabled:
                    topology_score_dict = self._flast_build_topology_score_dict(model)
                    topology_lambda = getattr(self.args, 'flast_topology_lambda', 0.0)
                    logging.info(
                        "[FedFLaST-Topology] Guided pruning enabled: score=%s, lambda=%s, layers=%s",
                        getattr(self.args, 'flast_topology_score', None),
                        topology_lambda,
                        len(topology_score_dict),
                    )
                new_global_mask = pruning(
                    model,
                    model.layer_density_dict,
                    pruning_strategy,
                    mask_dict=global_mask,
                    guidance_score_dict=topology_score_dict,
                    guidance_lambda=topology_lambda,
                )
                model.mask_dict = new_global_mask
                model.to(self.aggregator.device)
                model.apply_mask()

            # ==================== FedFLaST Momentum Update ====================
            # Update momentum EVERY round (not just mode 2/3).
            # When no mask readjustment: m_{t+1} = m_t (mask unchanged).
            # When mask readjustment: m_{t+1} is the new mask after prune-grow.
            if self._flast_enabled and self._flast_p_t is not None:
                current_mask = self.aggregator.trainer.model.mask_dict
                self._flast_update_momentum(self._flast_p_t, current_mask)
                logging.info(f"[FedFLaST] Momentum updated with m_{{{self.round_idx + 1}}}.")
            # ================================================================
                
            # logging.info("mask_dict after pruning and growing = " +str(mask_dict))
            self.aggregator.test_on_server_for_all_clients(self.round_idx)

            # Save current round_idx for flast logging BEFORE increment
            current_round = self.round_idx

            # start the next round
            self.round_idx += 1

            # convert the mode
            self.mode = self.mode_convert()

            if self.round_idx == self.round_num + 1:
                # post_complete_message_to_sweep_process(self.args)
                self._save_final_checkpoint()
                self.finish()
                print('here')
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                    self.args.client_num_per_round)

            print('indexes of clients: ' + str(client_indexes))
            print("size = %d" % self.size)
            logging.info(f"current step is {self.round_idx} and the current mode is {self.mode}")
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)

            # ==================== FedFLaST Perturbation Phase ====================
            # Compute perturbation on current weights and build w_tilde for broadcasting
            flast_delta = None
            flast_active_norm = 0.0
            if self._flast_enabled:
                w_current = self.aggregator.get_global_model_params()
                mask_dict = self.aggregator.trainer.model.mask_dict

                # Save w_t for next round's recovery
                self._flast_prev_w = {k: v.clone().detach() for k, v in w_current.items()}

                # Compute mask-projected perturbation
                flast_delta, flast_active_norm = self._flast_compute_perturbation(mask_dict)

                # Build w_tilde = w_current + delta
                global_model_params = self._flast_build_broadcast_model(w_current, flast_delta)
                self._flast_prev_w_tilde = global_model_params

                # Log FedFLaST diagnostics to the text logger without advancing W&B step.
                self._flast_log_metrics(flast_active_norm, flast_delta, self._flast_p_t, current_round)
            # =====================================================================
            
            if self.mode in [0, 3]:
                mask_dict = self.aggregator.trainer.model.mask_dict
                for k in mask_dict:
                    mask_dict[k] = mask_dict[k].cpu()
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                        client_indexes[receiver_id - 1], self.mode, self.round_idx, mask_dict)
            else:
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                        client_indexes[receiver_id - 1], self.mode, self.round_idx)


    def send_message_init_config(self, receive_id, global_model_params, client_index, mode_code, round_idx):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, mode_code, round_idx, mask_dict=None):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_MASKS, mask_dict)
        self.send_message(message)
