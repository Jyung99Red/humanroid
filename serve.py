"""回放页面的本地服务: 提供 web/ 下的静态文件。只监听本机(127.0.0.1), 只读, 不接受任何写入。

页面里的 three.js 是 ES 模块, 必须通过 http 打开(file:// 会被浏览器拦), 所以需要这个服务。
用法: python serve.py [端口]     (viewer.bat 会替你启动并打开浏览器)
"""
import http.server
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")      # 改了回放/页面后刷新就能看到
        super().end_headers()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):                # 控制台编码不是 UTF-8 时(Windows 常见)也不要因为打印而崩
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    print(f"Viewer: http://127.0.0.1:{PORT}/index.html   (Ctrl+C to quit)")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
