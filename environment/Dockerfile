# hash:sha256:41a85ba7479bf765762042ff8a5d0f7d004a963694d7f2738751ef7b630db63c
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    control==0.9.2 \
    gym==0.25.0 \
    matplotlib==3.5.2 \
    numpy==1.23.1 \
    opencv-python-headless==4.6.0.66 \
    pillow==9.2.0 \
    scikit-fuzzy==0.4.2 \
    scipy==1.8.1 \
    stable-baselines3==1.6.0
