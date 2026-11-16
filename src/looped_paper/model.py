# Core looped model extracted without changing its forward computation.
import torch
from torch import nn
from transformers import GPT2Config
from .gpt2 import GPT2Model

class GeneralTransformerModel(nn.Module):

    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, linear_embedding=False, path_indep=False):
        super(GeneralTransformerModel, self).__init__()
        configuration = GPT2Config(n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False)
        self.name = f'gpt2_embd={n_embd}_layer={n_layer}_head={n_head}'
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        if linear_embedding:
            self._read_in = nn.Linear(n_dims, n_embd)
        else:
            self._read_in = nn.Embedding(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims)
        self._addition_pe = None
        self.addition_pe_kwargs = {}
        self.use_addition_pe = False
        self._abacus = None
        self.abacus_kwargs = {}
        self.use_abacus = False
        self.path_indep = path_indep
        self.exit_gate = nn.Linear(n_embd, 1)
        nn.init.zeros_(self.exit_gate.weight)
        nn.init.constant_(self.exit_gate.bias, 0.0)

    def _compute_loop_gate(self, hidden_states, attention_mask=None):
        """Compute loop exit gate λ_t using masked average pooling.

        hidden_states: [B, L, D] (top-layer, current loop state)
        attention_mask: [B, L] with 1 for valid tokens, 0 for padding (or None)
        returns λ_t: [B] in (0,1)
        """
        (B, L, D) = hidden_states.shape
        h = hidden_states.mean(dim=1)
        logits = self.exit_gate(h).squeeze(-1)
        lam = torch.sigmoid(logits)
        return lam

    def looped_forward_with_exit_prob(self, zs, max_horizon, attention_mask=None, output_attentions=False):
        zs = self._read_in(zs)
        if self.path_indep:
            output = torch.randn_like(zs).to(zs.device)
        else:
            output = torch.zeros_like(zs).to(zs.device)
        output_list = []
        attention_list = []
        exit_prob_list = []
        survival = torch.ones(zs.size(0), device=zs.device)
        for i in range(max_horizon):
            if output_attentions:
                (output, attentions) = self.forward_single(output + zs, attention_mask, output_attentions=True)
                attention_list.append(attentions)
            else:
                output = self.forward_single(output + zs, attention_mask, output_attentions=False)
            output_list.append(self._read_out(output).clone())
            lam = self._compute_loop_gate(output.clone(), attention_mask)
            if i == max_horizon - 1:
                lam = torch.ones_like(lam)
            exit_prob = survival * lam
            exit_prob_list.append(exit_prob.clone())
            survival = survival * (1.0 - lam)
        if output_attentions:
            return (output_list, attention_list, exit_prob_list)
        return (output_list, exit_prob_list)

    def forward_single(self, embeds, attention_mask=None, output_attentions=False):
        output = self._backbone.forward_no_position(inputs_embeds=embeds, attention_mask=attention_mask, output_attentions=output_attentions)
        if output_attentions:
            return (output.last_hidden_state, output.attentions)
        return output.last_hidden_state

    def looped_forward(self, zs, horizon, attention_mask=None, output_attentions=False):
        zs = self._read_in(zs)
        if self.path_indep:
            output = torch.randn_like(zs).to(zs.device)
        else:
            output = torch.zeros_like(zs).to(zs.device)
        output_list = []
        attention_list = []
        for i in range(horizon):
            if output_attentions:
                (output, attentions) = self.forward_single(output + zs, attention_mask, output_attentions=True)
                attention_list.append(attentions)
            else:
                output = self.forward_single(output + zs, attention_mask, output_attentions=False)
            output_list.append(self._read_out(output).clone())
        if output_attentions:
            return (output_list, attention_list)
        return output_list
