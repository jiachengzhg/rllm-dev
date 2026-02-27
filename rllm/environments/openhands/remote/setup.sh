#!/bin/bash
#
# Remote Docker Server 环境安装脚本
#
# 使用方式:
#   chmod +x setup.sh
#   ./setup.sh
#
# 安装完成后启动服务:
#   source venv/bin/activate
#   python server.py --host 0.0.0.0 --port 8000
#
# 或使用run.sh一键启动

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  Remote Docker Server 环境安装"
echo "================================================"

# 检查Python版本
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ 错误: 未找到Python，请先安装Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ 使用Python: $PYTHON_CMD (版本: $PYTHON_VERSION)"

# 检查Docker是否安装并运行
echo ""
echo "检查Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker未安装，请先安装Docker"
    echo "  安装方式: https://docs.docker.com/engine/install/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ 错误: Docker daemon未运行或当前用户无权限"
    echo "  解决方案:"
    echo "    1. 启动Docker: sudo systemctl start docker"
    echo "    2. 添加用户到docker组: sudo usermod -aG docker $USER"
    echo "    3. 重新登录或运行: newgrp docker"
    exit 1
fi

DOCKER_VERSION=$(docker --version | awk '{print $3}')
echo "✓ Docker已安装并运行 (版本: $DOCKER_VERSION)"

# 创建虚拟环境
echo ""
echo "创建Python虚拟环境..."
if [ -d "venv" ]; then
    echo "  虚拟环境已存在，跳过创建"
else
    $PYTHON_CMD -m venv venv
    echo "✓ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source venv/bin/activate
echo "✓ 虚拟环境已激活"

# 升级pip
echo ""
echo "升级pip..."
pip install --upgrade pip -q

# 创建requirements.txt
echo ""
echo "安装依赖..."
cat > requirements.txt << 'EOF'
# Remote Docker Server 依赖
# FastAPI框架
fastapi>=0.100.0
uvicorn[standard]>=0.22.0

# Docker SDK
docker>=6.0.0

# Pydantic数据验证
pydantic>=2.0.0
EOF

# 安装依赖
pip install -r requirements.txt -q
echo "✓ 依赖安装完成"

# 创建run.sh启动脚本
echo ""
echo "创建启动脚本..."
cat > run.sh << 'EOF'
#!/bin/bash
#
# Remote Docker Server 启动脚本
#
# 使用方式:
#   ./run.sh                        # 默认启动 (0.0.0.0:8000)
#   ./run.sh --port 9000            # 指定端口
#   API_KEY=secret ./run.sh         # 启用API认证
#   ./run.sh --workers 4            # 多进程模式
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ 错误: 虚拟环境不存在，请先运行 ./setup.sh"
    exit 1
fi

# 默认参数
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "================================================"
echo "  启动 Remote Docker Server"
echo "  地址: http://${HOST}:${PORT}"
if [ -n "$API_KEY" ]; then
    echo "  认证: 已启用 (API_KEY已设置)"
else
    echo "  认证: 未启用 (设置API_KEY环境变量启用)"
fi
echo "================================================"

# 启动服务
exec python server.py --host "$HOST" --port "$PORT" "$@"
EOF

chmod +x run.sh
echo "✓ 启动脚本创建完成"

# 创建systemd服务文件(可选)
cat > remote-docker-server.service << EOF
[Unit]
Description=Remote Docker Server
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$SCRIPT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
# Environment="API_KEY=your_secret_key"  # 取消注释并设置API密钥
ExecStart=$SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/server.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "================================================"
echo "  安装完成!"
echo "================================================"
echo ""
echo "启动服务:"
echo "  ./run.sh"
echo ""
echo "或手动启动:"
echo "  source venv/bin/activate"
echo "  python server.py --host 0.0.0.0 --port 8000"
echo ""
echo "启用API认证:"
echo "  export API_KEY=your_secret_key"
echo "  ./run.sh"
echo ""
echo "安装为系统服务(可选):"
echo "  sudo cp remote-docker-server.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable remote-docker-server"
echo "  sudo systemctl start remote-docker-server"
echo ""
echo "查看API文档:"
echo "  http://localhost:8000/docs"
echo ""

