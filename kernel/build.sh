set -e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
T=/opt/venv/lib/python3.12/site-packages/torch
icpx -fsycl -O3 -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I$T/include -I$T/include/torch/csrc/api/include -I/opt/venv/include -I/opt/venv/include/sycl \
  -L$T/lib -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu \
  /w/w4_l80.cpp -o /w/libw4_l80.so
icpx -fsycl -O3 -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I$T/include -I$T/include/torch/csrc/api/include -I/opt/venv/include -I/opt/venv/include/sycl \
  -L$T/lib -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu \
  /w/w3_l80.cpp -o /w/libw3_l80.so
echo BUILD_OK
icpx -fsycl -O3 -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I$T/include -I$T/include/torch/csrc/api/include -I/opt/venv/include -I/opt/venv/include/sycl \
  -L$T/lib -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu \
  /w/w4a8_l80.cpp -o /w/libw4a8_l80.so
echo A8_BUILD_OK
