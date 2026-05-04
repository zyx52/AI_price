#!/usr/bin/env bash
# ================================================================
# AI Pricing Platform —— 服务启动脚本
#
# 启动所有守护进程:
#   1. Redis (如果未运行)
#   2. Weather Daemon   (30分钟轮询)
#   3. Turnstile Daemon (5分钟轮询)
#   4. Pricing Subscriber (实时定价决策)
#   5. API Server       (FastAPI)
#   6. Dashboard        (Streamlit, 可选)
#
# 用法:
#   chmod +x scripts/start_all.sh
#   ./scripts/start_all.sh              # 启动所有服务
#   ./scripts/start_all.sh --no-dashboard  # 跳过看板
#   ./scripts/start_all.sh --mock       # 强制mock模式
# ================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------- 颜色 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------- 参数 ----------
MOCK_MODE=false
NO_DASHBOARD=false

for arg in "$@"; do
    case "$arg" in
        --mock) MOCK_MODE=true ;;
        --no-dashboard) NO_DASHBOARD=true ;;
    esac
done

# ---------- 前置检查 ----------
check_python() {
    if command -v python3 &>/dev/null; then
        PYTHON=python3
    elif command -v python &>/dev/null; then
        PYTHON=python
    else
        err "Python 未安装"
        exit 1
    fi
    log "Python: $($PYTHON --version)"
}

check_redis() {
    if command -v redis-cli &>/dev/null; then
        if redis-cli ping &>/dev/null 2>&1; then
            log "Redis: 已运行"
            return 0
        fi
    fi

    # 尝试启动
    if command -v redis-server &>/dev/null; then
        warn "Redis 未运行, 正在启动..."
        redis-server --daemonize yes --port 6379 2>/dev/null || true
        sleep 1
        if redis-cli ping &>/dev/null 2>&1; then
            log "Redis: 启动成功"
            return 0
        fi
    fi

    warn "Redis 不可用, 守护进程将降级为 no-op 模式"
}

install_project() {
    if ! $PYTHON -c "import config" &>/dev/null 2>&1; then
        log "安装项目..."
        pip install -e . -q
    fi
    log "项目已安装"
}

# ---------- 启动函数 ----------
start_weather_daemon() {
    log "🌤️  启动天气守护进程 (30分钟轮询)..."
    if $MOCK_MODE; then
        $PYTHON services/weather_daemon.py --mock --interval 1800 &
    else
        $PYTHON services/weather_daemon.py --interval 1800 &
    fi
    echo $! > /tmp/ai_pricing_weather.pid
    log "   PID=$(cat /tmp/ai_pricing_weather.pid)"
}

start_turnstile_daemon() {
    log "🚪 启动闸机守护进程 (5分钟轮询)..."
    $PYTHON services/turnstile_daemon.py --source mock --interval 300 &
    echo $! > /tmp/ai_pricing_turnstile.pid
    log "   PID=$(cat /tmp/ai_pricing_turnstile.pid)"
}

start_pricing_subscriber() {
    log "🎯 启动定价引擎订阅者..."
    $PYTHON services/pricing_subscriber.py &
    echo $! > /tmp/ai_pricing_subscriber.pid
    log "   PID=$(cat /tmp/ai_pricing_subscriber.pid)"
}

start_api() {
    log "🌐 启动 FastAPI 服务 (端口8000)..."
    $PYTHON -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
    echo $! > /tmp/ai_pricing_api.pid
    log "   PID=$(cat /tmp/ai_pricing_api.pid)"
    log "   Swagger: http://localhost:8000/docs"
}

start_dashboard() {
    if $NO_DASHBOARD; then
        log "⏭️  跳过看板启动"
        return
    fi
    log "📊 启动 Streamlit 看板 (端口8501)..."
    $PYTHON -m streamlit run dashboard/app.py --server.port 8501 &
    echo $! > /tmp/ai_pricing_dashboard.pid
    log "   PID=$(cat /tmp/ai_pricing_dashboard.pid)"
    log "   看板: http://localhost:8501"
}

# ---------- 清理 ----------
cleanup() {
    log "正在停止所有服务..."
    for pidfile in /tmp/ai_pricing_*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            kill "$pid" 2>/dev/null || true
            rm -f "$pidfile"
        fi
    done
    log "所有服务已停止"
}
trap cleanup EXIT INT TERM

# ================================================================
# 主流程
# ================================================================
echo ""
echo -e "${BLUE}================================================================"
echo "  🎢 AI智能定价票务平台 · 服务启动"
echo -e "================================================================${NC}"
echo ""

check_python
check_redis
install_project

echo ""

start_weather_daemon
sleep 1

start_turnstile_daemon
sleep 1

start_pricing_subscriber
sleep 2

start_api
sleep 1

start_dashboard

echo ""
echo -e "${GREEN}================================================================"
echo "  ✅ 所有服务已启动"
echo -e "================================================================${NC}"
echo ""
echo "  服务列表:"
echo "    🌤️  天气守护进程     - PID $(cat /tmp/ai_pricing_weather.pid 2>/dev/null || echo 'N/A')"
echo "    🚪 闸机守护进程      - PID $(cat /tmp/ai_pricing_turnstile.pid 2>/dev/null || echo 'N/A')"
echo "    🎯 定价引擎订阅者    - PID $(cat /tmp/ai_pricing_subscriber.pid 2>/dev/null || echo 'N/A')"
echo "    🌐 API 服务         - http://localhost:8000"
if ! $NO_DASHBOARD; then
    echo "    📊 运营看板         - http://localhost:8501"
fi
echo ""
echo "  按 Ctrl+C 停止所有服务"
echo ""

# 等待
wait
