# Remote Docker Server

远程Docker服务端，用于将Docker操作代理到远程服务器。

## 架构说明

```
┌──────────────────┐         HTTP API         ┌──────────────────┐
│   Training       │ ──────────────────────►  │   Docker Server  │
│   Server         │                          │   (Remote)       │
│                  │                          │                  │
│  OHEnv           │  containers.list()       │  FastAPI Server  │
│    └── Runtime   │  containers.run()        │    └── Docker    │
│        Client    │  container.exec_run()    │        SDK       │
│        (Proxy)   │  container.put_archive() │                  │
└──────────────────┘ ◄────────────────────────└──────────────────┘
```

## 快速开始

### 1. 远程服务器部署

将 `remote/` 目录复制到远程Docker服务器：

```bash
# 在远程服务器上
cd /path/to/remote
chmod +x setup.sh
./setup.sh
```

启动服务：

```bash
./run.sh
# 或指定端口
./run.sh --port 9000
# 或启用API认证
API_KEY=your_secret_key ./run.sh
```

### 2. 客户端使用

在训练/推理服务器上：

```python
from rllm.environments.openhands.oh_env import OHEnv

# 使用远程Docker服务器
env = OHEnv(
    entry=your_task_entry,
    use_remote=True,
    remote_server_url="http://192.168.1.100:8000",
    remote_api_key="your_secret_key",  # 可选
)

# 正常使用环境
obs, info = env.reset()
obs, reward, done, info = env.step(action)
```

或者直接使用 RuntimeClient：

```python
from rllm.environments.openhands.runtime_client import RuntimeClient

client = RuntimeClient(
    backend="docker",
    use_remote=True,
    remote_server_url="http://192.168.1.100:8000",
)

client.connect(ds_entry)
result = client.run("echo hello")
print(result.output)
```

## API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/ping` | 测试连接 |
| GET | `/containers` | 列出容器 |
| POST | `/containers/run` | 创建并运行容器 |
| GET | `/containers/{id}/status` | 获取容器状态 |
| POST | `/containers/{id}/start` | 启动容器 |
| POST | `/containers/{id}/stop` | 停止容器 |
| DELETE | `/containers/{id}` | 删除容器 |
| POST | `/containers/{id}/exec` | 执行命令 |
| POST | `/containers/{id}/put_archive` | 上传文件 |

完整API文档: `http://your-server:8000/docs`

## 配置选项

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `API_KEY` | API认证密钥 | 无 (禁用认证) |
| `HOST` | 绑定地址 | `0.0.0.0` |
| `PORT` | 监听端口 | `8000` |

### 命令行参数

```bash
python server.py --host 0.0.0.0 --port 8000 --workers 4
```

## 安全建议

1. **设置API密钥**: 生产环境务必设置 `API_KEY`
2. **使用HTTPS**: 配合Nginx/Caddy等反向代理启用TLS
3. **防火墙**: 限制端口访问来源IP
4. **Docker权限**: 服务用户需要docker组权限

## systemd 服务

安装为系统服务以实现开机自启：

```bash
sudo cp remote-docker-server.service /etc/systemd/system/
# 编辑服务文件设置API_KEY
sudo systemctl daemon-reload
sudo systemctl enable remote-docker-server
sudo systemctl start remote-docker-server
```

查看日志：

```bash
sudo journalctl -u remote-docker-server -f
```

## 故障排查

### 连接被拒绝

1. 检查服务是否启动: `curl http://localhost:8000/ping`
2. 检查防火墙: `sudo ufw status`
3. 检查端口绑定: `netstat -tlnp | grep 8000`

### Docker权限错误

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 镜像拉取失败

确保远程服务器能访问Docker镜像源：

```bash
docker pull swebench/sweb.eval.x86_64.xxx
```

