#!/bin/bash
wheel_path=
venv_dir=/home/ubuntu/XMIGCS_DEPLOY/xmigcs_venv

for arg in "$@";do
  key=${arg%=*}
  val=${arg#*=}
  echo "${key}: ${val}"
  if [ "${key}" = "--wheel" ];then
    wheel_path=${val}
  fi
done

if [[ -z ${wheel_path} ]] || [[ "${wheel_path}" == "--wheel" ]]; then
  echo "need param --wheel=<path to wheel file>"
  exit 1
fi

echo "create virtual environment"
# 检查是否已安装python3.10-venv，如果未安装则安装
if ! dpkg -l | grep -q python3.10-venv; then
    echo "python3.10-venv未安装，正在安装..."
    sudo apt update
    sudo apt install -y python3.10-venv
else
    echo "python3.10-venv已经安装，跳过安装步骤"
fi

# 检查虚拟环境目录是否存在，如果不存在则创建并安装依赖
if [ ! -d "${venv_dir}" ]; then
  echo "虚拟环境目录不存在，正在创建: ${venv_dir}"
  python3 -m venv ${venv_dir}
else
  echo "虚拟环境已存在: ${venv_dir}"
fi

echo "依赖包安装完成, 安装xmigcs"
${venv_dir}/bin/pip uninstall xmigcs -y
${venv_dir}/bin/pip install "${wheel_path}" -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo "安装完成"
echo "= = = INSTALL OTA PACKAGE SUCCESS = = = "
echo "To use xMIGCS, please activate the virtual environment with the following command:"
echo "source ${venv_dir}/bin/activate"
echo "Then you can run xMIGCS with the command 'xmigcs' for real robot"