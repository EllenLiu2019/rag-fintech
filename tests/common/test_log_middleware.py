"""
中间件模块的单元测试
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from common.log_middleware import (
    request_logging_middleware,
    setup_request_logging_middleware
)


class TestRequestLoggingMiddleware:
    """测试 request_logging_middleware 函数"""
    
    @pytest.fixture
    def mock_request(self):
        """创建模拟的 Request 对象"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/test"
        request.query_params = {}
        request.state = Mock()
        return request
    
    @pytest.fixture
    def mock_call_next(self):
        """创建模拟的 call_next 函数"""
        async def call_next(request):
            return JSONResponse(content={"message": "test"}, status_code=200)
        return call_next
    
    @pytest.mark.asyncio
    async def test_request_id_generation(self, mock_request, mock_call_next):
        """测试请求 ID 生成"""
        await request_logging_middleware(mock_request, mock_call_next)
        
        # 验证请求 ID 已设置到 request.state
        assert hasattr(mock_request.state, 'request_id')
        assert mock_request.state.request_id is not None
        assert len(mock_request.state.request_id) == 8  # UUID 前8位
    
    @pytest.mark.asyncio
    async def test_response_headers(self, mock_request, mock_call_next):
        """测试响应头中包含请求 ID"""
        response = await request_logging_middleware(mock_request, mock_call_next)
        
        # 验证响应头包含 X-Request-ID
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] == mock_request.state.request_id
    
    @pytest.mark.asyncio
    async def test_logging_request_info(self, mock_request, mock_call_next):
        """测试请求信息日志记录"""
        with patch('common.log_middleware.logger') as mock_logger:
            await request_logging_middleware(mock_request, mock_call_next)
            
            # 验证记录了请求信息
            assert mock_logger.info.called
            # 检查是否记录了请求方法和路径
            [str(call) for call in mock_logger.info.call_args_list]
            assert any('GET' in str(call) and '/api/test' in str(call) for call in mock_logger.info.call_args_list)
    
    @pytest.mark.asyncio
    async def test_logging_response_info(self, mock_request, mock_call_next):
        """测试响应信息日志记录"""
        with patch('common.log_middleware.logger') as mock_logger:
            await request_logging_middleware(mock_request, mock_call_next)
            
            # 验证记录了响应信息
            assert any('200' in str(call) or '响应' in str(call) for call in mock_logger.info.call_args_list)
    
    @pytest.mark.asyncio
    async def test_query_params_logging(self, mock_request, mock_call_next):
        """测试查询参数日志记录"""
        # 设置查询参数
        mock_query_params = Mock()
        mock_query_params.__iter__ = Mock(return_value=iter([('key', 'value')]))
        mock_query_params.__getitem__ = Mock(return_value='value')
        mock_request.query_params = {'test': 'value'}
        
        with patch('common.log_middleware.logger') as mock_logger:
            await request_logging_middleware(mock_request, mock_call_next)
            
            # 验证记录了查询参数
            assert any('query params' in str(call) for call in mock_logger.info.call_args_list)
    
    @pytest.mark.asyncio
    async def test_no_query_params(self, mock_request, mock_call_next):
        """测试没有查询参数时不记录查询参数"""
        mock_request.query_params = {}
        
        with patch('common.log_middleware.logger') as mock_logger:
            await request_logging_middleware(mock_request, mock_call_next)
            
            # 验证没有记录查询参数
            call_args = [str(call) for call in mock_logger.info.call_args_list]
            query_param_logs = [call for call in call_args if 'query params' in call]
            assert len(query_param_logs) == 0
    
    @pytest.mark.asyncio
    async def test_response_passed_through(self, mock_request, mock_call_next):
        """测试响应正确传递"""
        response = await request_logging_middleware(mock_request, mock_call_next)
        
        # 验证响应对象正确返回
        assert response is not None
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_different_request_methods(self, mock_request, mock_call_next):
        """测试不同的 HTTP 方法"""
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        
        for method in methods:
            mock_request.method = method
            response = await request_logging_middleware(mock_request, mock_call_next)
            
            assert response is not None
            assert mock_request.state.request_id is not None
    
    @pytest.mark.asyncio
    async def test_request_id_uniqueness(self, mock_request, mock_call_next):
        """测试每次请求生成不同的请求 ID"""
        request_ids = set()
        
        for _ in range(10):
            # 创建新的 mock request
            new_request = Mock(spec=Request)
            new_request.method = "GET"
            new_request.url.path = "/api/test"
            new_request.query_params = {}
            new_request.state = Mock()
            
            await request_logging_middleware(new_request, mock_call_next)
            request_ids.add(new_request.state.request_id)
        
        # 验证生成了不同的请求 ID（至少大部分不同）
        assert len(request_ids) >= 8  # 10次请求应该生成至少8个不同的ID


class TestSetupRequestLoggingMiddleware:
    """测试 setup_request_logging_middleware 函数"""
    
    def test_middleware_registration(self):
        """测试中间件是否正确注册到 FastAPI 应用"""
        app = FastAPI()
        
        # 注册中间件
        setup_request_logging_middleware(app)
        
        # 验证中间件已注册
        # FastAPI 的中间件存储在 app.user_middleware 中
        assert len(app.user_middleware) > 0
        
        # 检查是否有 HTTP 中间件（通过装饰器注册的中间件会被包装成 BaseHTTPMiddleware）
        from starlette.middleware.base import BaseHTTPMiddleware
        http_middlewares = [m for m in app.user_middleware if hasattr(m, 'cls') and m.cls == BaseHTTPMiddleware]
        assert len(http_middlewares) > 0
    
    def test_middleware_integration(self):
        """测试中间件在 FastAPI 应用中的集成"""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        # 注册中间件
        setup_request_logging_middleware(app)
        
        # 创建测试客户端
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # 发送请求
        with patch('common.log_middleware.logger') as mock_logger:
            response = client.get("/test")
            
            # 验证响应
            assert response.status_code == 200
            assert "X-Request-ID" in response.headers
            
            # 验证日志被调用
            assert mock_logger.info.called
    
    def test_middleware_with_query_params(self):
        """测试带查询参数的请求"""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        setup_request_logging_middleware(app)
        
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        with patch('common.log_middleware.logger') as mock_logger:
            response = client.get("/test?key=value&foo=bar")
            
            assert response.status_code == 200
            assert "X-Request-ID" in response.headers
            
            # 验证记录了查询参数
            call_args = [str(call) for call in mock_logger.info.call_args_list]
            assert any('query params' in str(call) for call in call_args)


class TestMiddlewareErrorHandling:
    """测试中间件的错误处理"""
    
    @pytest.mark.asyncio
    async def test_call_next_exception(self):
        """测试 call_next 抛出异常时的处理"""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/test"
        mock_request.query_params = {}
        mock_request.state = Mock()
        
        async def failing_call_next(request):
            raise ValueError("Test error")
        
        # 中间件应该让异常传播
        with pytest.raises(ValueError, match="Test error"):
            await request_logging_middleware(mock_request, failing_call_next)
        
        # 但请求 ID 应该已经设置
        assert hasattr(mock_request.state, 'request_id')
    
    def test_different_response_status_codes(self):
        """测试不同的响应状态码"""
        app = FastAPI()
        
        @app.get("/success")
        async def success():
            return {"message": "success"}
        
        @app.get("/not-found")
        async def not_found():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not found")
        
        setup_request_logging_middleware(app)
        
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # 测试成功响应
        response = client.get("/success")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        
        # 测试错误响应
        response = client.get("/not-found")
        assert response.status_code == 404
        assert "X-Request-ID" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

