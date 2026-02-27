"""
Remote Docker Proxy - 模拟docker SDK行为，将操作转发到远程Docker服务器。

这个模块提供了:
- RemoteDockerClient: 模拟docker.Client的行为
- RemoteContainer: 模拟docker.Container的行为

远程服务器只需要实现简单的REST API即可支持:
- containers.list
- containers.run  
- container.status
- container.start()
- container.stop()
- container.remove()
- container.exec_run()
- container.put_archive()
"""

import base64
import io
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecResult:
    """模拟docker exec_run的返回结果"""
    exit_code: int
    output: bytes


class RemoteContainer:
    """
    模拟docker.Container对象，将操作转发到远程服务器。
    """
    
    def __init__(
        self, 
        client: "RemoteDockerClient",
        container_id: str,
        name: str,
        status: str = "running",
    ):
        self._client = client
        self.id = container_id
        self.name = name
        self._status = status
    
    @property
    def status(self) -> str:
        """获取容器状态"""
        try:
            response = self._client._request(
                "GET", 
                f"/containers/{self.id}/status"
            )
            self._status = response.get("status", "unknown")
        except Exception:
            pass
        return self._status
    
    def start(self) -> None:
        """启动容器"""
        self._client._request("POST", f"/containers/{self.id}/start")
        self._status = "running"
    
    def stop(self, timeout: int = 10) -> None:
        """停止容器"""
        self._client._request(
            "POST", 
            f"/containers/{self.id}/stop",
            json={"timeout": timeout}
        )
        self._status = "exited"
    
    def remove(self, force: bool = False) -> None:
        """删除容器"""
        self._client._request(
            "DELETE",
            f"/containers/{self.id}",
            json={"force": force}
        )
    
    def exec_run(
        self,
        cmd: str | list[str],
        workdir: str | None = None,
        environment: dict | None = None,
        stdout: bool = True,
        stderr: bool = True,
        **kwargs,
    ) -> ExecResult:
        """在容器中执行命令"""
        if isinstance(cmd, list):
            cmd_str = " ".join(cmd) if cmd[0] != "/bin/sh" else cmd[-1]
        else:
            cmd_str = cmd
            
        response = self._client._request(
            "POST",
            f"/containers/{self.id}/exec",
            json={
                "cmd": cmd_str,
                "workdir": workdir,
                "environment": environment,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        
        output = response.get("output", "")
        if isinstance(output, str):
            output = output.encode("utf-8")
        
        return ExecResult(
            exit_code=response.get("exit_code", -1),
            output=output,
        )
    
    def put_archive(self, path: str, data: bytes) -> bool:
        """上传tar归档到容器"""
        # 将bytes转为base64以便JSON传输
        encoded_data = base64.b64encode(data).decode("utf-8")
        
        response = self._client._request(
            "POST",
            f"/containers/{self.id}/put_archive",
            json={
                "path": path,
                "data": encoded_data,
            }
        )
        return response.get("success", False)


class RemoteContainersManager:
    """
    模拟docker.client.containers管理器
    """
    
    def __init__(self, client: "RemoteDockerClient"):
        self._client = client
    
    def list(self, all: bool = False, filters: dict | None = None) -> list[RemoteContainer]:
        """列出容器"""
        response = self._client._request(
            "GET",
            "/containers",
            params={
                "all": all,
                "filters": filters,
            }
        )
        
        containers = []
        for c in response.get("containers", []):
            containers.append(RemoteContainer(
                client=self._client,
                container_id=c["id"],
                name=c["name"],
                status=c.get("status", "unknown"),
            ))
        return containers
    
    def run(
        self,
        image: str,
        command: str | None = None,
        name: str | None = None,
        detach: bool = False,
        tty: bool = False,
        stdin_open: bool = False,
        environment: dict | None = None,
        **kwargs,
    ) -> RemoteContainer:
        """创建并运行容器"""
        response = self._client._request(
            "POST",
            "/containers/run",
            json={
                "image": image,
                "command": command,
                "name": name,
                "detach": detach,
                "tty": tty,
                "stdin_open": stdin_open,
                "environment": environment,
                **kwargs,
            }
        )
        
        return RemoteContainer(
            client=self._client,
            container_id=response["id"],
            name=response.get("name", name or ""),
            status=response.get("status", "running"),
        )


class RemoteDockerClient:
    """
    模拟docker.Client，将Docker操作转发到远程服务器。
    
    使用方式与docker.from_env()返回的client完全一致:
        client = RemoteDockerClient(server_url="http://remote:8000")
        containers = client.containers.list()
        container = client.containers.run(...)
        
    远程服务器需要实现相应的REST API endpoints。
    """
    
    def __init__(
        self,
        server_url: str,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        """
        初始化远程Docker客户端。
        
        Args:
            server_url: 远程服务器URL，如 http://192.168.1.100:8000
            api_key: 可选的API密钥用于认证
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http_client = None
        
        # 初始化containers管理器
        self.containers = RemoteContainersManager(self)
    
    def _get_http_client(self):
        """获取或创建HTTP客户端"""
        if self._http_client is None:
            try:
                import httpx
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                self._http_client = httpx.Client(
                    base_url=self.server_url,
                    headers=headers,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "httpx is required for RemoteDockerClient. "
                    "Install with: pip install httpx"
                )
        return self._http_client
    
    def _request(
        self, 
        method: str, 
        path: str, 
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """发送HTTP请求到远程服务器"""
        client = self._get_http_client()
        
        # 清理params中的None值
        if params:
            params = {k: v for k, v in params.items() if v is not None}
            # 将复杂类型转为JSON字符串
            for k, v in params.items():
                if isinstance(v, (dict, list)):
                    import json as json_module
                    params[k] = json_module.dumps(v)
        
        response = client.request(
            method=method,
            url=path,
            json=json,
            params=params,
        )
        response.raise_for_status()
        return response.json()
    
    def close(self) -> None:
        """关闭客户端连接"""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
    
    def ping(self) -> bool:
        """测试与远程服务器的连接"""
        try:
            response = self._request("GET", "/ping")
            return response.get("ok", False)
        except Exception:
            return False


def from_remote(
    server_url: str,
    api_key: str | None = None,
    timeout: int = 120,
) -> RemoteDockerClient:
    """
    创建远程Docker客户端的便捷函数。
    
    类似于docker.from_env()的使用方式。
    
    Args:
        server_url: 远程服务器URL
        api_key: 可选的API密钥
        timeout: 请求超时时间
        
    Returns:
        RemoteDockerClient实例
    """
    return RemoteDockerClient(
        server_url=server_url,
        api_key=api_key,
        timeout=timeout,
    )

