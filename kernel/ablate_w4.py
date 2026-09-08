import time, os, torch, vllm  # noqa
H=os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(H,"libw4_l80.so"))
dev,G="xpu",128
PERM=[0,16,4,20,8,24,12,28]
def repack(q):
    src=q.t().contiguous(); o=torch.zeros_like(src)
    for j,p in enumerate(PERM): o |= ((src>>(4*j))&0xF)<<p
    return o
print(f"device: {torch.xpu.get_device_name(0)}")
print(f"\n  {'K x N':>14s} {'vendor us':>10s} {'full us':>9s} {'no-deq us':>10s} "
      f"{'ALU cost':>9s} {'read ceil':>10s} {'max poss':>9s}")
print("  " + "-"*78)
for K,N in [(5120,17408),(17408,5120),(6144,5120)]:
    nb=(K//8)*N*4+(K//G)*N*2; nbuf=max(2,int(512e6/nb+0.5))
    vend=[];ours=[]
    for _ in range(nbuf):
        qw=torch.randint(-2**31,2**31-1,(N,K//8),dtype=torch.int32,device=dev).t()
        ws=(torch.randn(N,K//G,dtype=torch.float16,device=dev).abs()+.01).t().contiguous()
        vend.append((qw,ws)); ours.append((repack(qw),ws.t().contiguous()))
    qz=torch.tensor([8],dtype=torch.int8,device=dev)
    x=torch.randn(1,K,dtype=torch.float16,device=dev)*0.05
    def bench(fn):
        for i in range(5): fn(i)
        torch.xpu.synchronize(); b=1e30
        for _ in range(8):
            torch.xpu.synchronize(); t0=time.perf_counter()
            for i in range(50): fn(i)
            torch.xpu.synchronize(); b=min(b,(time.perf_counter()-t0)/50)
        return b
    tv=bench(lambda i: torch.ops._xpu_C.int4_gemm_w4a16(x,vend[i%nbuf][0],None,vend[i%nbuf][1],qz,G,None))
    tf=bench(lambda i: torch.ops.p608.gemv_w4(x,ours[i%nbuf][0],ours[i%nbuf][1],2,0))
    tn=bench(lambda i: torch.ops.p608.gemv_w4(x,ours[i%nbuf][0],ours[i%nbuf][1],2,1))
    ceil=nb/579.4e9
    print(f"  {K:>5d}x{N:<7d} {tv*1e6:>10.1f} {tf*1e6:>9.1f} {tn*1e6:>10.1f} "
          f"{(tf-tn)/tf*100:>8.1f}% {ceil*1e6:>10.1f} {tv/ceil:>8.2f}x")
    del vend,ours
print("\n  ALU cost = fraction of runtime that disappears when dequant is removed")
print("  max poss = vendor time / pure read time = the best any 4-bit kernel could do")
