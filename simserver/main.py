from __future__ import annotations
import os
import uvicorn

def main():
    # 환경에 맞게 host/port 조정
    uvicorn.run(
        "simserver.app:create_app",
        factory=True,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("UVICORN_RELOAD", "0") in ("1", "true", "True"),
        log_level=os.getenv("LOG_LEVEL", "debug"),   # ← 디버그
        access_log=True,                              # ← 액세스 로그
        proxy_headers=True,                           # 프록시/컨테이너에서 유용
    )

if __name__ == "__main__":
    main()
