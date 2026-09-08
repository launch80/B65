// Stage 0b - can a hand-written SYCL kernel live inside vLLM's XPU graph capture?
//
// VLLM_XPU_ENABLE_XPU_GRAPH=1 makes vLLM shim torch.cuda.graph -> torch.xpu.graph
// and CUDAGraph -> torch.xpu.XPUGraph (vllm/v1/worker/xpu_model_runner.py:59-63),
// so decode runs under real STREAM CAPTURE, not torch.compile.
//
// The failure mode that matters: a SYCL op that creates its own sycl::queue
// submits outside the captured stream. Capture then records nothing for it, and
// on replay the op silently does not run - the graph replays stale output.
// Nothing errors. So the op must submit on torch's CURRENT XPU stream.
//
// This defines two ops that differ ONLY in which queue they use, so the test can
// tell "capture works" from "capture silently dropped my kernel".
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

static void launch(sycl::queue &q, const float *xp, float *yp, float s, int64_t n) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { yp[i] = xp[i] * s; });
    });
}

// CORRECT: submit on the stream torch is capturing.
at::Tensor scale_stream(const at::Tensor &x, double s) {
    auto y = at::empty_like(x);
    sycl::queue &q = c10::xpu::getCurrentXPUStream().queue();
    launch(q, x.data_ptr<float>(), y.data_ptr<float>(), (float)s, x.numel());
    return y;
}

// WRONG ON PURPOSE: a private queue, to prove the probe can detect the failure.
at::Tensor scale_ownqueue(const at::Tensor &x, double s) {
    auto y = at::empty_like(x);
    static sycl::queue own{sycl::gpu_selector_v};
    launch(own, x.data_ptr<float>(), y.data_ptr<float>(), (float)s, x.numel());
    own.wait();
    return y;
}

TORCH_LIBRARY(p608probe, m) {
    m.def("scale_stream(Tensor x, float s) -> Tensor");
    m.def("scale_ownqueue(Tensor x, float s) -> Tensor");
}
TORCH_LIBRARY_IMPL(p608probe, XPU, m) {
    m.impl("scale_stream", scale_stream);
    m.impl("scale_ownqueue", scale_ownqueue);
}
