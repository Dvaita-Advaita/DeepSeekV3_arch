import math
import torch

from torchfeather.model.model_args import DeepSeekV3ModelArgs

def precompute_frequency_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:
    d_head = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow

    # Base is 10000
    base = args.rope_theta
    factor = args.rope_factor
    
    # Find the RoPE pair index i which have completed r rotation
    # i = (dim * log(L/2pi * r))/2log(base)
    def find_correction_dim(
            num_rotations: float, dim: int, base: float,max_seq_len: int
    ) -> float:
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )
    # This will tell three boundary region
    # r > beta_fast, beta_slow <= r <= beta_fast,r < beta_slow
    def find_correction_range(
            low_rot: float,high_rot: float,dim: int,base: float,max_seq_len: int
    ) -> tuple[int,int]:
        low = math.floor(find_correction_dim(low_rot,dim,base,max_seq_len))
        high = math.ceil(find_correction_dim(high_rot,dim,base,max_seq_len))
        return max(low,0), min(high,dim - 1)
    
    def linear_ramp_factor(min: float, max: float,dim: int) -> torch.Tensor:
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim,dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func,0,1)
        return ramp_func

    # Basic RoPE frequency calculation
    # theta_i = 10000^(-2i/d_head)
    freqs = 1.0/base **(torch.arange(0,d_head,2,dtype=torch.float32)/d_head)

    # YaRN scaling for extended context. YaRN is used to extend the context length after pre-training
    if seqlen > args.original_seq_len:
        low,high = find_correction_range(
            beta_fast,beta_slow,d_head,base,args.original_seq_len
            )
        smooth = 1 - linear_ramp_factor(low,high,d_head // 2)
        freqs = freqs/factor * (1 - smooth) + freqs * smooth

    # Create position indices 
    t = torch.arange(seqlen)

    # angle = position * freq
    # angle is also called outer product: [positional] * [indices]
    # angle has shape [max_seq_len,D/2]
    angle = torch.outer(t,freqs)

    # Convert to complex exponentials: e^(iota * angle)
    angle_cis = torch.polar(torch.ones_like(angle),angle)

    return angle_cis

def apply_rotary_emb(x: torch.Tensor, angle_cis: torch.Tensor) -> torch.Tensor:
    # x: [B,N,H,D] 
    # angle_cis = [N,D/2]
    # original dtype of x which can be float16 or bfloat16
    dtype = x.dtype
    # As x is a real number we want to view it as complex number because angle_cis already complex
    # [x1,x2] -> torch,view_as_complex([x1,x2]) --> [x1 + ix2] 
    # Since last dim of x is D we need to change it to D/2
    # *x.shape[:-1] means take every dim expect last one so [B,N,H],so x.float().view(B,N,H,-1,2)
    # Suppose D = 8 then (B,N,H,8) -> x.float().view(B,N,H,-1,2) --> (B,N,H,4,2)
    # torch.view_as_complex([B,N,H,4,2]) --> [B,N,H,4],view_as_complex convert 2 real elements into one single complex number
    # [1,2,3,4,5,6,7,8] to [[1,2],[3,4],[5,6],[7,8]] and then view_as_complex make them
    # [[1+2i],[3+4i],[5+6i],[7+8i]]
    x = torch.view_as_complex(x.float().view(*x.shape[:-1],-1,2))
    # As angle_cis has dim of [N,D/2] we need to change it to [1,N,1,D/2] so elementwise multiplication can take place, so broadcasting is done here
    angle_cis = angle_cis.view(1,x.size(1),1,x.size(-1))
    # [x1+ix2] -> torch,real([x1+ix2]) --. [x1,x2]
    # x:[B,N,H,D/2,2] -> x.flatten(3) --> x:[X,N,H,D]
    # So starting from dim 3 flatten all remaining dim,D/2 * 2 = (8/2) * 2 = 4, so [B,N,H,4,2] --> [B,N,H,8]
    y = torch.view_as_real(x * angle_cis).flatten(3)
    return y.to(dtype)
