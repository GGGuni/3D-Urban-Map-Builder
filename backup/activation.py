import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        print("trunc_exp forward")
        ctx.save_for_backward(x)
        #sigma 계산
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        print("trunc_exp backward")
        # print("g: ",g)
        # print("trunc_exp ctx: ", ctx.next_functions)
        x = ctx.saved_tensors[0]
        torch.save(g * torch.exp(x.clamp(-15, 15)), 'grad_trunc.pth')

        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply