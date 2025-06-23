# 使用 Ubuntu 基础镜像
FROM ccr.ccs.tencentyun.com/library/ubuntu:22.04

# 更新软件包列表并安装必要的工具
RUN sed -i 's@archive.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y build-essential libopenmpi-dev openmpi-bin openmpi-common gcc g++ && \
    rm -rf /var/lib/apt/lists/* &&\
    mkdir -p /data/students && chown -R root:root /data/students
# 设置工作目录
WORKDIR /data/students

# 启动容器时的默认命令
CMD ["/bin/bash"]

# {
#     "registry-mirrors": [
#      "https://docker-0.unsee.tech",
#         "https://docker-cf.registry.cyou",
#         "https://docker.1panel.live"
#     ]
# }