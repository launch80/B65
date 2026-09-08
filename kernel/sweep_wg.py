import time, os, sys, torch, vllm  # noqa
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w3_format import pack
H = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(H, "libw3_l80.so"))
dev, G = "xpu", 128
def pack_planes(c):
    K, N = c.shape; nb = K // 32
    w = pack(c.T.reshape(N, nb, 32).reshape(-1, 32)).reshape(N, nb, 3)
    return torch.from_numpy(np.concatenate([w[:,:,0],w[:,:,1],w[:,:,2]],1).astype(np.int32)).to(dev).contiguous()
print(f"device: {torch.xpu.get_device_name(0)}")
for K, N in [(5120, 17408), (17408, 5120)]:
    b4=(K//8)*N*4+(K//G)*N*2; b3=(K//32)*3*N*4+(K//G)*N*2
    nbuf=max(2,int(512e6/b4+0.5)); rng=np.random.default_rng(608)
    vend=[];w3=[]
    for _ in range(nbuf):
        vend.append((torch.randint(-2**31,2**31-1,(N,K//8),dtype=torch.int32,device=dev).t(),
                     (torch.randn(N,K//G,dtype=torch.float16,device=dev).abs()+.01).t().contiguous()))
        w3.append((pack_planes(rng.integers(0,8,size=(K,N)).astype(np.uint8)),
                   (torch.randn(N,K//G,dtype=torch.float16,device=dev).abs()*.02+.005).contiguous()))
    qz=torch.tensor([8],dtype=torch.int8,device=dev); x=torch.randn(1,K,dtype=torch.float16,device=dev)*0.05
    def bench(fn):
        for i in range(5): fn(i)
        torch.xpu.synchronize(); best=1e30
        for _ in range(8):
            torch.xpu.synchronize(); t0=time.perf_counter()
            for i in range(50): fn(i)
            torch.xpu.synchronize(); best=min(best,(time.perf_counter()-t0)/50)
        return best
    tv=bench(lambda i: torch.ops._xpu_C.int4_gemm_w4a16(x,vend[i%nbuf][0],None,vend[i%nbuf][1],qz,G,None))
    print(f"\n  {K}x{N}   vendor {b4/tv/1e9:.1f} GB/s ({tv*1e6:.1f} us)")
    print(f"  {'SG':>4s} {'SGS':>4s} {'NCOL':>5s} {'us':>8s} {'GB/s':>8s} {'speedup':>8s}")
    best=(0,None)
    for sg,sgs in [(32,1),(32,2),(32,4),(32,8),(32,16),(16,2),(16,4),(16,8)]:
        for nc in (1,2,4):
            try:
                t=bench(lambda i: torch.ops.p608w3.gemv_w3(x,w3[i%nbuf][0],w3[i%nbuf][1],False,nc,sg,sgs))
            except Exception: continue
            sp=tv/t
            if sp>best[0]: best=(sp,(sg,sgs,nc))
            if sp > 1.10:
                print(f"  {sg:>4d} {sgs:>4d} {nc:>5d} {t*1e6:>8.1f} {b3/t/1e9:>8.1f} {sp:>7.2f}x")
    print(f"  BEST: SG={best[1][0]} SGS={best[1][1]} NCOL={best[1][2]} -> {best[0]:.2f}x")
