import torch
import snntorch as snn
from magicpig import LSHLayer 
from quantize import nf4_quantize
class SpikeGate(snn.Leaky):

    def __init__(self, beta, threshold_skip=0.1):
        super().__init__(beta=beta, threshold=1.0, reset_mechanism="zero")
        self.threshold_skip = threshold_skip

    def forward(self, x):
        mem = self.init_leaky() 
        spk_out = []
        skip_flags = []
        for step in range(x.size(0)):  
            spk, mem = super().forward(x[step], mem)
            spk_rate = spk.mean().item()  
            
            skip = spk_rate < self.threshold_skip
            skip_flags.append(skip)
            spk_out.append(spk if not skip else torch.zeros_like(spk))
        return torch.stack(spk_out), torch.tensor(skip_flags)
    
class SNNLLM(torch.nn.Module):

    def __init__(self, vocab_size, hidden_dim, beta=0.95):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.spike_gate = SpikeGate(beta=beta, threshold_skip=0.05)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):

        x = self.embed(input_ids)  
        x_poisson = torch.rand_like(x) < x.abs()  

        spk_seq, skip_flags = self.spike_gate(x_poisson)
        active_steps = [spk_seq[i] for i, skip in enumerate(skip_flags) if not skip]
        if active_steps:
            active_out = self.fc(torch.stack(active_steps))
            logits = torch.softmax(active_out, dim=-1)
        else:
            logits = torch.zeros(1, vocab_size)  
            return logits, skip_flags
    
    def generate_zero_skip(model, prompt_ids, max_len=50):

    output_ids = list(prompt_ids)
    current_input = torch.tensor(prompt_ids).unsqueeze(0)
    
    for _ in range(max_len):
        logits, skip_flags = model(current_input)
        next_token = logits.argmax(-1)[-1].item()
        if skip_flags[-1]:
            next_token = 0 
        output_ids.append(next_token)
        current_input = torch.tensor(output_ids[-1:]).unsqueeze(0)
    return output_ids
 
    self.lsh_attn = LSHLayer(k=10, l=150) 
    model.to('loihi')