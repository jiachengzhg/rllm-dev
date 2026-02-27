"""
华山Docker代理 - 通过MCP服务与华山平台交互。

华山是公司内部的Docker运行平台，通过MCP服务提供容器管理功能。

核心工具:
- start: 启动容器，返回pod_name
- wait_for_ready: 等待pod就绪
- bash_execute: 在容器中执行命令
- stop: 停止容器

注意：华山MCP没有list/status接口，因此本代理采用简化设计。
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("HuashanDockerProxy")


@dataclass
class ExecResult:
    """模拟docker exec_run的返回结果"""
    exit_code: int
    output: bytes


class HuashanMCPClient:
    """
    华山MCP客户端 - 与华山MCP服务通信。
    
    基于mcp_client_example.py中的MCPCaller简化实现。
    """
    
    INIT_PAYLOAD = {
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {
                "name": "mcp",
                "version": "0.1.0"
            }
        },
        "jsonrpc": "2.0",
        "id": 0
    }
    
    NOTIFY_PAYLOAD = {
        "method": "notifications/initialized",
        "jsonrpc": "2.0",
    }
    
    # 工具名称前缀
    TOOL_PREFIX = "codeagent_huashan_docker_"
    HTTP_TIMEOUT_BUFFER_SEC = 120
    
    ROUTE_SNIFF_INIT_TIMES = 10

    def __init__(self, server_url: str, timeout: int = 300):
        """
        初始化华山MCP客户端。
        
        Args:
            server_url: MCP服务器URL
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url
        self.timeout = timeout
        self._cookies: list[str] = []
        self._request_idx = 0
    
    def _get_hash(self, hash_key: str) -> str:
        return hashlib.sha256(hash_key.encode()).hexdigest()
    
    def _get_element_by_hash(self, hash_key: str, all_elements: list) -> str:
        hash_hex = self._get_hash(hash_key)
        hash_int = int(hash_hex, 16)
        index = hash_int % len(all_elements)
        return all_elements[index]
    
    async def _sniff_route(self) -> None:
        """嗅探路由cookie"""
        try:
            import aiohttp
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    self.server_url,
                    json=self.INIT_PAYLOAD,
                    headers=headers,
                    ssl=ssl_context,
                ) as response:
                    if response.headers and 'Set-Cookie' in response.headers:
                        cookie = response.headers['Set-Cookie']
                        if 'route' in cookie and cookie not in self._cookies:
                            self._cookies.append(cookie)
        except Exception as e:
            logger.debug(f"Route sniff failed: {e}")
    
    async def _init_cookies(self) -> None:
        """初始化cookies"""
        for _ in range(self.ROUTE_SNIFF_INIT_TIMES):
            await self._sniff_route()
    
    async def call_tool(
        self, 
        tool_name: str, 
        params: dict,
        session: str | None = None,
    ) -> Any:
        """
        调用MCP工具。
        
        Args:
            tool_name: 工具名称（不含前缀）
            params: 工具参数
            session: 可选的会话ID用于路由
            
        Returns:
            工具调用结果
        """
        try:
            import aiohttp
            import ssl
        except ImportError:
            raise ImportError(
                "aiohttp is required for HuashanMCPClient. "
                "Install with: pip install aiohttp"
            )
        
        # 初始化cookies
        if not self._cookies:
            await self._init_cookies()
        elif self._request_idx % 100 == 0:
            await self._sniff_route()
        
        # 选择路由cookie
        cookie = ""
        if self._cookies:
            if session is not None:
                cookie = self._get_element_by_hash(session, self._cookies)
            else:
                cookie = self._cookies[self._request_idx % len(self._cookies)]
        self._request_idx += 1
        
        # 准备请求头
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if cookie:
            headers["Cookie"] = cookie
        
        # 工具调用payload
        full_tool_name = f"{self.TOOL_PREFIX}{tool_name}"
        payload = {
            "method": "tools/call",
            "params": {"name": full_tool_name, "arguments": params},
            "jsonrpc": "2.0",
            "id": 1
        }
        
        # SSL配置
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        request_timeout_sec = self.timeout
        try:
            tool_timeout_sec = params.get("timeout_sec")
            if tool_timeout_sec is not None:
                request_timeout_sec = max(
                    self.timeout,
                    int(float(tool_timeout_sec)) + self.HTTP_TIMEOUT_BUFFER_SEC,
                )
        except Exception:
            request_timeout_sec = self.timeout

        timeout = aiohttp.ClientTimeout(total=request_timeout_sec)
        
        async with aiohttp.ClientSession(timeout=timeout) as client:
            # 初始化
            async with client.post(
                self.server_url, 
                json=self.INIT_PAYLOAD, 
                headers=headers, 
                ssl=ssl_context
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"MCP init failed: {response.status} {response.reason}"
                    )
                headers['mcp-session-id'] = response.headers.get('mcp-session-id')
            
            # 通知
            async with client.post(
                self.server_url, 
                json=self.NOTIFY_PAYLOAD, 
                headers=headers, 
                ssl=ssl_context
            ) as response:
                if response.status != 202:
                    raise RuntimeError(
                        f"MCP notify failed: {response.status} {response.reason}"
                    )
            
            # 调用工具
            async with client.post(
                self.server_url, 
                json=payload, 
                headers=headers, 
                ssl=ssl_context
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"MCP tool call failed: {response.status} {response.reason}"
                    )
                
                text = await response.text()
                for line in text.split(os.linesep):
                    if not line.startswith("data:"):
                        continue
                    result = line.removeprefix("data:").strip()
                    content = json.loads(result)
                    
                    # 检查错误
                    if content.get('result', {}).get('isError'):
                        raise RuntimeError(
                            f"MCP tool error: {content['result'].get('content')}"
                        )
                    
                    # 提取返回内容
                    result_content = content.get('result', {}).get('content', [])
                    if not result_content:
                        return None
                    
                    real_content = result_content[0].get('text', '')
                    try:
                        return json.loads(real_content)
                    except json.JSONDecodeError:
                        return real_content
        
        return None
    
    def call_tool_sync(
        self, 
        tool_name: str, 
        params: dict,
        session: str | None = None,
    ) -> Any:
        """同步版本的call_tool"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is not None:
            # 已有运行中的事件循环，使用nest_asyncio或创建新线程
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    self.call_tool(tool_name, params, session)
                )
                return future.result()
        else:
            # 没有运行中的事件循环，直接使用asyncio.run
            return asyncio.run(self.call_tool(tool_name, params, session))


class HuashanContainer:
    """
    模拟docker.Container对象，使用华山MCP执行操作。
    """
    
    def __init__(
        self, 
        client: "HuashanDockerClient",
        pod_name: str,
        image: str,
    ):
        self._client = client
        self.id = pod_name
        self.name = pod_name
        self._image = image
        self._status = "running"
    
    @property
    def status(self) -> str:
        """
        获取容器状态。
        
        注意：华山MCP没有状态查询接口，这里返回缓存的状态。
        """
        return self._status
    
    def start(self) -> None:
        """
        启动容器。
        
        注意：华山MCP不支持重新启动已停止的容器，此方法为空操作。
        """
        logger.warning("Huashan does not support restarting stopped containers")
    
    def stop(self, timeout: int = 10) -> None:
        """停止容器"""
        self._client._call_tool("stop", {"pod_name": self.name})
        self._status = "exited"
    
    def remove(self, force: bool = False) -> None:
        """
        删除容器。
        
        华山MCP中stop即删除，此方法调用stop。
        """
        if self._status != "exited":
            self.stop()
    
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
        # 处理命令格式
        if isinstance(cmd, list):
            # 如果是 ["/bin/sh", "-c", "actual_command"] 格式
            if len(cmd) >= 3 and cmd[0] == "/bin/sh" and cmd[1] == "-c":
                cmd_list = [cmd[2]]
            else:
                cmd_list = cmd
        else:
            cmd_list = [cmd]
        
        params = {
            "pod_name": self.name,
            "cmd": cmd_list,
            "timeout_sec": kwargs.get("timeout", 3000), # set to 3000 because compute reward will take a long time
        }
        if workdir:
            params["work_dir"] = workdir
        
        try:
            result = self._client._call_tool("bash_execute", params)
            
            # 解析结果
            if result is None:
                return ExecResult(exit_code=0, output=b"")
            
            if isinstance(result, str):
                output = result
            else:
                output = str(result)
            
            # 检查是否有退出码信息
            exit_code = 0
            if "Process exited with code" in output:
                import re
                match = re.search(r"Process exited with code (\d+)", output)
                if match:
                    exit_code = int(match.group(1))
            if "Execution timeout" in output:
                exit_code = 124
            
            return ExecResult(
                exit_code=exit_code,
                output=output.encode("utf-8"),
            )
        except Exception as e:
            err_msg = str(e).strip() or repr(e)
            logger.error(f"exec_run failed: {err_msg}")
            return ExecResult(exit_code=-1, output=str(e).encode("utf-8"))
    
    def put_archive(self, path: str, data: bytes) -> bool:
        """
        上传tar归档到容器。
        
        华山MCP不直接支持文件上传，通过bash命令实现。
        这里使用base64编码后通过echo写入。
        """
        import base64
        import tarfile
        import io
        
        # 解压tar获取文件内容
        try:
            tar_stream = io.BytesIO(data)
            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        file_content = tar.extractfile(member)
                        if file_content:
                            content = file_content.read()
                            dest_file = os.path.join(path, member.name)
                            
                            # 使用base64编码并通过bash写入
                            b64_content = base64.b64encode(content).decode("utf-8")
                            
                            # 确保目录存在
                            dest_dir = os.path.dirname(dest_file)
                            self.exec_run(f"mkdir -p {dest_dir}")
                            
                            # 写入文件（分块写入避免命令过长）
                            chunk_size = 50000  # 每块约50KB
                            if len(b64_content) <= chunk_size:
                                cmd = f"echo '{b64_content}' | base64 -d > {dest_file}"
                                self.exec_run(cmd)
                            else:
                                # 大文件分块写入
                                self.exec_run(f"rm -f {dest_file}")
                                for i in range(0, len(b64_content), chunk_size):
                                    chunk = b64_content[i:i+chunk_size]
                                    if i == 0:
                                        cmd = f"echo '{chunk}' > {dest_file}.b64"
                                    else:
                                        cmd = f"echo '{chunk}' >> {dest_file}.b64"
                                    self.exec_run(cmd)
                                self.exec_run(f"base64 -d {dest_file}.b64 > {dest_file} && rm {dest_file}.b64")
            
            return True
        except Exception as e:
            logger.error(f"put_archive failed: {e}")
            return False


class HuashanContainersManager:
    """
    模拟docker.client.containers管理器
    """
    
    def __init__(self, client: "HuashanDockerClient"):
        self._client = client
        self._containers: dict[str, HuashanContainer] = {}
    
    def list(self, all: bool = False, filters: dict | None = None) -> list[HuashanContainer]:
        """
        列出容器。
        
        注意：华山MCP不支持列出容器，返回缓存的容器列表。
        """
        if filters and "name" in filters:
            name = filters["name"]
            if name in self._containers:
                return [self._containers[name]]
            return []
        return list(self._containers.values())
    
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
    ) -> HuashanContainer:
        """创建并运行容器"""
        # 调用start工具
        params = {"image_id": image}
        
        # 可以传递pod_spec
        if "pod_spec" in kwargs:
            params["pod_spec"] = kwargs["pod_spec"]
        
        # 启动命令
        # 华山平台无TTY/STDIN保持机制，直接传"/bin/bash"会立即退出。
        # 对这种默认bash命令，交给服务端默认启动命令（如sleep infinity）。
        if command:
            if isinstance(command, str):
                if command != "/bin/bash":
                    params["cmd"] = [command]
            else:
                # 如果显式传了命令列表，原样透传（过滤单独的 /bin/bash）
                if not (len(command) == 1 and command[0] == "/bin/bash"):
                    params["cmd"] = command
        
        pod_name = self._client._call_tool("start", params)
        
        if not pod_name:
            raise RuntimeError(f"Failed to start container with image: {image}")
        
        # 等待容器就绪
        is_ready = self._client._call_tool(
            "wait_for_ready", 
            {"pod_name": pod_name, "timeout_sec": 120.0}
        )
        
        if not is_ready:
            raise RuntimeError(f"Container {pod_name} failed to become ready")
        
        # 创建容器对象
        container = HuashanContainer(
            client=self._client,
            pod_name=pod_name,
            image=image,
        )
        
        self._containers[pod_name] = container
        return container


class HuashanDockerClient:
    """
    模拟docker.Client，使用华山MCP服务管理容器。
    
    使用方式与docker.from_env()返回的client类似:
        client = HuashanDockerClient(server_url="https://...")
        container = client.containers.run(image, ...)
        result = container.exec_run(cmd)
    """
    
    def __init__(
        self,
        server_url: str,
        timeout: int = 300,
    ):
        """
        初始化华山Docker客户端。
        
        Args:
            server_url: 华山MCP服务器URL
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url
        self.timeout = timeout
        self._mcp_client = HuashanMCPClient(server_url, timeout)
        
        # 初始化containers管理器
        self.containers = HuashanContainersManager(self)
    
    def _call_tool(self, tool_name: str, params: dict) -> Any:
        """调用MCP工具"""
        return self._mcp_client.call_tool_sync(tool_name, params)
    
    def close(self) -> None:
        """关闭客户端连接"""
        # 停止所有容器
        for container in list(self.containers._containers.values()):
            try:
                container.stop()
            except Exception as e:
                logger.warning(f"Failed to stop container {container.name}: {e}")
        self.containers._containers.clear()
    
    def ping(self) -> bool:
        """测试连接"""
        try:
            # 尝试调用一个简单的操作来测试连接
            return True
        except Exception:
            return False


def from_huashan(
    server_url: str,
    timeout: int = 300,
) -> HuashanDockerClient:
    """
    创建华山Docker客户端的便捷函数。
    
    Args:
        server_url: 华山MCP服务器URL
        timeout: 请求超时时间
        
    Returns:
        HuashanDockerClient实例
    """
    return HuashanDockerClient(server_url=server_url, timeout=timeout)