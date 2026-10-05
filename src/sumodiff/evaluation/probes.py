"""Small reproducible numerical probes; no diffusion model or training."""
from dataclasses import replace
import time
import torch
from sumodiff.decoding import decode_future,DecoderConfig


def numeric_probe(seed, cuda_smoke=False, config=None):
    torch.manual_seed(seed)
    config = config or DecoderConfig()
    raw = torch.randn(1,3,6,dtype=torch.float64)*.2
    raw[...,5] += 1
    raw.requires_grad_(True)
    start,heading,mask = torch.zeros(1,2,dtype=torch.float64),torch.tensor([[0.,1.]],dtype=torch.float64),torch.ones(1,dtype=torch.bool)
    passed = torch.autograd.gradcheck(lambda value:decode_future(value,start,heading,mask,replace(config,solve_dtype='float64')).states,(raw,),atol=2e-5,rtol=1e-3)
    decoded = decode_future(raw,start,heading,mask,config)
    decoded.states.square().sum().backward()
    result = dict(gradcheck_passed=bool(passed),cpu_gradient_finite=bool(torch.isfinite(raw.grad).all()),
        solve_dtype=config.solve_dtype,gradcheck_dtype='float64',cuda=None)
    if cuda_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA smoke explicitly requested but CUDA is unavailable')
        device = torch.device('cuda:0')
        generator = torch.Generator(device=device).manual_seed(seed)
        future = torch.randn((2,12,40,6),dtype=torch.float32,device=device,generator=generator)*.2
        future[...,5] += 1
        future.requires_grad_(True)
        start = torch.zeros((2,12,2),device=device)
        heading = torch.zeros_like(start); heading[...,1] = 1
        mask = torch.ones((2,12),dtype=torch.bool,device=device)
        # Only this decoder is measured. Warm up solver/library/context separately.
        for _ in range(2):
            out = decode_future(future,start,heading,mask,config)
            out.states.square().mean().backward(); future.grad = None
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()
        for _ in range(5):
            out = decode_future(future,start,heading,mask,config)
            out.states.square().mean().backward(); future.grad = None
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start_time
        out = decode_future(future,start,heading,mask,config)
        out.states.square().mean().backward(); torch.cuda.synchronize()
        result['cuda'] = dict(device=torch.cuda.get_device_name(device),shape=[2,12,40,6],
            output_dtype=str(out.states.dtype),gradient_finite=bool(torch.isfinite(future.grad).all()),
            forward_backward_mean_seconds=elapsed/5,measured_iterations=5,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            scope='decoder-only after warmup; excludes network, sampling and process/driver memory')
    return result
