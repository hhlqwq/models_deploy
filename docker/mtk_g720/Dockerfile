# 使用 Docker 内置前端, 避免额外拉取 docker/dockerfile 镜像.
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_SOURCE_URL=https://mirrors.aliyun.com/python-release/source/Python-3.11.11.tgz
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# 在 Ubuntu 内编译 Python, 保持 SDK 的 cp311 ABI.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++-aarch64-linux-gnu rsync ca-certificates curl git patch unzip openssh-client procps time ffmpeg \
    libgl1 libglib2.0-0 libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libffi-dev liblzma-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /tmp/hailongcodex/python-build \
    && cd /tmp/hailongcodex/python-build \
    && curl --fail --location --retry 3 "${PYTHON_SOURCE_URL}" -o python.tgz \
    && tar -xzf python.tgz && cd Python-3.11.11 \
    && ./configure --prefix=/usr/local --with-ensurepip=install \
    && make -j8 && make install && ln -s python3.11 /usr/local/bin/python \
    && rm -rf /tmp/hailongcodex/python-build
RUN python -m pip config --global set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && python -m pip install pip==25.2 setuptools==80.9.0 wheel==0.45.1
COPY requirements.txt yolov5-requirements.txt /opt/mtk-build/
RUN python -m pip install --no-cache-dir torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118 \
    && python -m pip install --no-cache-dir -r /opt/mtk-build/requirements.txt -r /opt/mtk-build/yolov5-requirements.txt
ARG MTK_SDK_ROOT=/opt/mtk/neuropilot-sdk
ENV NCC_ROOT=${MTK_SDK_ROOT}/neuron_sdk/host
ENV PATH="${NCC_ROOT}/bin:${PATH}"
# sdk 是构建脚本传入的服务器本地 SDK, 无远程下载.
COPY --from=sdk / ${MTK_SDK_ROOT}/
RUN python -m pip install --no-cache-dir \
    ${MTK_SDK_ROOT}/offline_tool/mtk_converter-8.16.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    ${MTK_SDK_ROOT}/offline_tool/mtk_quantization-8.2.1-py3-none-any.whl \
    && ln -s ${MTK_SDK_ROOT}/neuron_sdk/host/lib/libc++.so.1 /usr/local/lib/libc++.so.1 \
    && nvrtc_path="$(find /usr/local/lib/python3.11/site-packages/torch/lib -maxdepth 1 -name 'libnvrtc-*.so.*' -print -quit)" \
    && test -n "${nvrtc_path}" \
    && ln -s "$(basename "${nvrtc_path}")" /usr/local/lib/python3.11/site-packages/torch/lib/libnvrtc.so \
    && ldconfig
# GAI 使用已有本地 SDK,只复制 wheel,不复制模型资源.
COPY --from=gai /mtk_llm_sdk-2.7.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /opt/mtk-build/
RUN python -m pip install --no-cache-dir /opt/mtk-build/mtk_llm_sdk-2.7.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# 补齐 ONNX Runtime CUDA EP 需要的 cu11 库并统一挂到系统库路径.
RUN python -m pip install --no-cache-dir nvidia-curand-cu11==10.3.0.86 nvidia-cufft-cu11==10.9.0.58 nvidia-cusolver-cu11==11.4.1.48 nvidia-cusparse-cu11==11.7.4.91 \
    && for lib in /usr/local/lib/python3.11/site-packages/torch/lib/*.so* \
                  /usr/local/lib/python3.11/site-packages/nvidia/*/lib/*.so*; do \
           ln -sf "${lib}" /usr/local/lib/; \
       done \
    && ldconfig
COPY setup_container.sh verify_environment.py /opt/mtk-build/
RUN install -d -m 1777 /tmp/hailongcodex "/tmp/hailongcodex/$(date +%F)" \
    && bash /opt/mtk-build/setup_container.sh --build \
    && python -m pip freeze > /opt/mtk-build/installed-requirements.txt
ARG MTK_RECIPE_SHA256
LABEL mtk.recipe_sha256="${MTK_RECIPE_SHA256}"
LABEL mtk.environment="ubuntu22.04-np8.0.11-py311-cu118" mtk.llm_sdk="2.7.5"
WORKDIR /workspace/models_deploy
CMD ["sleep", "infinity"]
