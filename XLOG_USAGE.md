### 1. 解压二进制包

```bash
cd /home/ubuntu
mkdir -p xlogs_depends/xlog_v1.0.1
# 将二进制包放入该文件夹
# 解压
tar -C /home/ubuntu/xlogs_depends/xlog_v1.0.1 -xf /home/ubuntu/xlogs_depends/xlog_v1.0.1/xlog_v1.0.1_standalone_humble_linux_x86_64.tar.gz
```


### 2. 设置环境变量

```bash
#每次打开终端均需要设置环境变量，可写入bashrc或zshrc
cat <<'EOF' >> ~/.bashrc
export XLOG_PREFIX=/home/ubuntu/xlogs_depends/xlog_v1.0.1
export CMAKE_PREFIX_PATH=$XLOG_PREFIX:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$XLOG_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$XLOG_PREFIX/lib/python3.10/site-packages:$PYTHONPATH
EOF

cat <<'EOF' >> ~/.zshrc
export XLOG_PREFIX=/home/ubuntu/xlogs_depends/xlog_v1.0.1
export CMAKE_PREFIX_PATH=$XLOG_PREFIX:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$XLOG_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$XLOG_PREFIX/lib/python3.10/site-packages:$PYTHONPATH
EOF

source ~/.zshrc
source ~/.bashrc
```

### 3. xmigcs运行

状态机已接入xlogs， 运行xmigs第一行出现下面输出说明xlog配置完成。

```bash
[XLOG][FSM] initialized successfully: logger_id=xmigcs_fsm, sub_dir=xmigcs
```

记录的状态机log信息将在ctrl c 后落盘到/home/ubuntu/logs/glogs/xmigcs目录下。


