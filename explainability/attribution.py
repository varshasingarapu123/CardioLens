import torch

def gradient_attribution(model,signal,target_class=None):
    x=torch.tensor(signal,dtype=torch.float32).reshape(1,1,-1); x.requires_grad_(True); model.eval(); logits=model(x)
    if target_class is None: target_class=int(logits.argmax(1).item())
    model.zero_grad(set_to_none=True); logits[0,target_class].backward(); attribution=x.grad.detach().abs().squeeze().numpy(); maximum=attribution.max()
    if maximum>0: attribution/=maximum
    return target_class,attribution
