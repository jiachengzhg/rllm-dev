"""
Remote Docker Server - 远程Docker服务端

这个服务提供REST API来代理Docker操作，允许客户端远程控制Docker容器。

功能:
- 列出容器
- 创建并运行容器
- 获取容器状态
- 启动/停止/删除容器
- 在容器中执行命令
- 上传文件到容器

使用方式:
    python server.py --host 0.0.0.0 --port 8000

API认证(可选):
    设置环境变量 API_KEY 来启用Bearer Token认证
    export API_KEY=your_secret_key
"""

import base64
import json
import logging
import os
from typing import Any

import docker
from fastapi import Depends, FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RemoteDockerServer")

# 创建FastAPI应用
app = FastAPI(
    title="Remote Docker Server",
    description="远程Docker操作代理服务",
    version="1.0.0",
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Docker客户端
docker_client: docker.DockerClient | None = None


def get_docker_client() -> docker.DockerClient:
    """获取Docker客户端实例"""
    global docker_client
    if docker_client is None:
        docker_client = docker.from_env(timeout=120)
    return docker_client


# API Key认证
API_KEY = os.environ.get("API_KEY")


async def verify_api_key(authorization: str | None = Header(default=None)):
    """验证API Key"""
    if API_KEY is None:
        # 未设置API_KEY时不需要认证
        return
    
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization[7:]  # 去掉 "Bearer " 前缀
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ==================== 请求/响应模型 ====================

class ContainerInfo(BaseModel):
    """容器信息"""
    id: str
    name: str
    status: str


class ContainersListResponse(BaseModel):
    """容器列表响应"""
    containers: list[ContainerInfo]


class ContainerRunRequest(BaseModel):
    """运行容器请求"""
    image: str
    command: str | None = None
    name: str | None = None
    detach: bool = True
    tty: bool = False
    stdin_open: bool = False
    environment: dict | None = None
    # 其他常用参数
    working_dir: str | None = None
    volumes: dict | None = None
    network_mode: str | None = None
    ports: dict | None = None
    mem_limit: str | None = None
    cpu_count: int | None = None


class ContainerRunResponse(BaseModel):
    """运行容器响应"""
    id: str
    name: str
    status: str


class ContainerStatusResponse(BaseModel):
    """容器状态响应"""
    status: str


class ContainerStopRequest(BaseModel):
    """停止容器请求"""
    timeout: int = 10


class ContainerRemoveRequest(BaseModel):
    """删除容器请求"""
    force: bool = False


class ExecRequest(BaseModel):
    """执行命令请求"""
    cmd: str
    workdir: str | None = None
    environment: dict | None = None
    stdout: bool = True
    stderr: bool = True


class ExecResponse(BaseModel):
    """执行命令响应"""
    exit_code: int
    output: str


class PutArchiveRequest(BaseModel):
    """上传归档请求"""
    path: str
    data: str  # base64 encoded


class PutArchiveResponse(BaseModel):
    """上传归档响应"""
    success: bool


class PingResponse(BaseModel):
    """Ping响应"""
    ok: bool
    message: str


# ==================== API端点 ====================

@app.get("/ping", response_model=PingResponse)
async def ping():
    """测试连接"""
    try:
        client = get_docker_client()
        client.ping()
        return PingResponse(ok=True, message="Docker server is running")
    except Exception as e:
        return PingResponse(ok=False, message=str(e))


@app.get("/containers", response_model=ContainersListResponse, dependencies=[Depends(verify_api_key)])
async def list_containers(
    all: bool = Query(default=False, description="包括已停止的容器"),
    filters: str | None = Query(default=None, description="过滤条件(JSON格式)"),
):
    """
    列出容器
    
    Args:
        all: 是否包括已停止的容器
        filters: JSON格式的过滤条件，如 {"name": "my-container"}
    """
    try:
        client = get_docker_client()
        
        filter_dict = None
        if filters:
            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid filters JSON")
        
        containers = client.containers.list(all=all, filters=filter_dict)
        
        result = []
        for c in containers:
            result.append(ContainerInfo(
                id=c.id,
                name=c.name,
                status=c.status,
            ))
        
        return ContainersListResponse(containers=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing containers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/containers/run", response_model=ContainerRunResponse, dependencies=[Depends(verify_api_key)])
async def run_container(request: ContainerRunRequest):
    """
    创建并运行容器
    """
    try:
        client = get_docker_client()
        
        # 构建运行参数
        run_kwargs = {
            "image": request.image,
            "detach": request.detach,
            "tty": request.tty,
            "stdin_open": request.stdin_open,
        }
        
        if request.command:
            run_kwargs["command"] = request.command
        if request.name:
            run_kwargs["name"] = request.name
        if request.environment:
            run_kwargs["environment"] = request.environment
        if request.working_dir:
            run_kwargs["working_dir"] = request.working_dir
        if request.volumes:
            run_kwargs["volumes"] = request.volumes
        if request.network_mode:
            run_kwargs["network_mode"] = request.network_mode
        if request.ports:
            run_kwargs["ports"] = request.ports
        if request.mem_limit:
            run_kwargs["mem_limit"] = request.mem_limit
        if request.cpu_count:
            run_kwargs["cpu_count"] = request.cpu_count
        
        container = client.containers.run(**run_kwargs)
        
        logger.info(f"Container created: {container.name} ({container.id[:12]})")
        
        return ContainerRunResponse(
            id=container.id,
            name=container.name,
            status=container.status,
        )
    
    except docker.errors.ImageNotFound as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image}")
    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error running container: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/containers/{container_id}/status", response_model=ContainerStatusResponse, dependencies=[Depends(verify_api_key)])
async def get_container_status(container_id: str):
    """
    获取容器状态
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        container.reload()  # 刷新状态
        return ContainerStatusResponse(status=container.status)
    
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Container not found: {container_id}")
    except Exception as e:
        logger.error(f"Error getting container status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/containers/{container_id}/start", dependencies=[Depends(verify_api_key)])
async def start_container(container_id: str):
    """
    启动容器
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        container.start()
        logger.info(f"Container started: {container_id[:12]}")
        return {"success": True}
    
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Container not found: {container_id}")
    except Exception as e:
        logger.error(f"Error starting container: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/containers/{container_id}/stop", dependencies=[Depends(verify_api_key)])
async def stop_container(container_id: str, request: ContainerStopRequest | None = None):
    """
    停止容器
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        
        timeout = 10
        if request:
            timeout = request.timeout
        
        container.stop(timeout=timeout)
        logger.info(f"Container stopped: {container_id[:12]}")
        return {"success": True}
    
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Container not found: {container_id}")
    except Exception as e:
        logger.error(f"Error stopping container: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/containers/{container_id}", dependencies=[Depends(verify_api_key)])
async def remove_container(container_id: str, request: ContainerRemoveRequest | None = None):
    """
    删除容器
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        
        force = False
        if request:
            force = request.force
        
        container.remove(force=force)
        logger.info(f"Container removed: {container_id[:12]}")
        return {"success": True}
    
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Container not found: {container_id}")
    except Exception as e:
        logger.error(f"Error removing container: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/containers/{container_id}/exec", response_model=ExecResponse, dependencies=[Depends(verify_api_key)])
async def exec_in_container(container_id: str, request: ExecRequest):
    """
    在容器中执行命令
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        
        # 构建执行命令
        cmd = ["/bin/sh", "-c", request.cmd]
        
        exec_kwargs = {
            "cmd": cmd,
            "stdout": request.stdout,
            "stderr": request.stderr,
        }
        
        if request.workdir:
            exec_kwargs["workdir"] = request.workdir
        if request.environment:
            exec_kwargs["environment"] = request.environment
        
        result = container.exec_run(**exec_kwargs)
        
        output = result.output
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        
        return ExecResponse(
            exit_code=result.exit_code if result.exit_code is not None else -1,
            output=output,
        )
    
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Container not found: {container_id}")
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/containers/{container_id}/put_archive", response_model=PutArchiveResponse, dependencies=[Depends(verify_api_key)])
async def put_archive(container_id: str, request: PutArchiveRequest):
    """
    上传tar归档到容器
    """
    try:
        client = get_docker_client()
        container = client.containers.get(container_id)
        
        # 解码base64数据
        try:
            data = base64.b64decode(request.data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")
        
        success = container.put_archive(request.path, data)
        
        logger.info(f"Archive uploaded to container {container_id[:12]}:{request.path}")
        return PutArchiveResponse(success=success)
    
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Container not found: {container_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading archive: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 服务器启动 ====================

def main():
    """启动服务器"""
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Remote Docker Server")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载(开发模式)")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Remote Docker Server on {args.host}:{args.port}")
    if API_KEY:
        logger.info("API key authentication is ENABLED")
    else:
        logger.warning("API key authentication is DISABLED (set API_KEY env var to enable)")
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

