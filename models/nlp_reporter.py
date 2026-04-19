"""
NLP 叙事报告生成

场景:
  - 每日运营简报
  - 机会点提示("明日雨天,建议上线雨天特惠")
  - 高管摘要

实现:
  1) 规则模板引擎(无需LLM,可直接运行) —— 默认方式
  2) LLM增强模式(可接入 Claude/OpenAI API,留出接口) —— 需要API Key时启用

真实项目建议走LLM模式获得更好的叙事感。
"""
from __future__ import annotations

import os
import importlib
from typing import Optional

from utils.logger import get_logger

logger = get_logger("NLPReporter")


class NLPReporter:
    def __init__(
        self,
        use_llm: bool = False,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.use_llm = use_llm
        resolved_provider = provider or os.getenv("NLP_API_PROVIDER", "anthropic")
        self.provider = resolved_provider.lower().strip()
        self.model = model or os.getenv("NLP_API_MODEL") or self._default_model(self.provider)
        self.base_url = base_url or os.getenv("NLP_API_BASE_URL")
        self.api_key = api_key or self._resolve_api_key(self.provider)

        if use_llm and not self.api_key:
            logger.warning(f"未提供 {self.provider} API Key,降级为模板模式")
            self.use_llm = False

    @staticmethod
    def _default_model(provider: str) -> str:
        defaults = {
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek-chat",
            "moonshot": "moonshot-v1-8k",
            "qwen": "qwen-plus",
        }
        return defaults.get(provider, "gpt-4o-mini")

    @staticmethod
    def _resolve_api_key(provider: str) -> Optional[str]:
        # 优先统一变量,其次读取各平台常见变量名
        if os.getenv("NLP_API_KEY"):
            return os.getenv("NLP_API_KEY")

        provider_key_env = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
        }
        env_name = provider_key_env.get(provider)
        return os.getenv(env_name) if env_name else None

    @staticmethod
    def _provider_base_url(provider: str) -> Optional[str]:
        # OpenAI SDK可直连一批兼容API
        defaults = {
            "openai": None,
            "deepseek": "https://api.deepseek.com",
            "moonshot": "https://api.moonshot.cn/v1",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        return defaults.get(provider)

    def _build_prompt(self, ctx: dict) -> str:
        return f"""
你是乐园运营AI助理,请根据以下数据,用3-5句话生成专业的运营日报开场白:

日期: {ctx.get('date')}
推荐票价: ¥{ctx.get('recommended_price')}
预计客流: {ctx.get('predicted_visitors'):,.0f}
预计收入: ¥{ctx.get('predicted_revenue'):,.0f}
天气: {ctx.get('weather')}
日期类型: {ctx.get('day_type')}
预警: {ctx.get('alerts', [])}

语气: 简洁专业、重点突出、面向高管。
"""

    def _call_anthropic(self, prompt: str) -> str:
        anthropic = importlib.import_module("anthropic")
        client = anthropic.Anthropic(api_key=self.api_key)
        msg = client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        content = getattr(msg, "content", None) or []
        if content and hasattr(content[0], "text"):
            return str(content[0].text).strip()
        return ""

    def _call_openai_compatible(self, prompt: str) -> str:
        openai_mod = importlib.import_module("openai")
        base_url = self.base_url or self._provider_base_url(self.provider)
        client = openai_mod.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0.7,
            messages=[
                {"role": "system", "content": "你是乐园运营分析助理。"},
                {"role": "user", "content": prompt},
            ],
        )
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        return str(content).strip() if content else ""

    # ---------- 每日简报 ----------
    def daily_brief(self, context: dict) -> str:
        """
        context:
          date, recommended_price, predicted_visitors, predicted_revenue,
          weather, day_type, competitor_avg, alerts(list), bundle_suggestion
        """
        if self.use_llm:
            return self._llm_brief(context)
        return self._template_brief(context)

    # ---------- 模板方式 ----------
    def _template_brief(self, ctx: dict) -> str:
        weather_tips = {
            "晴好": "天气晴好,宜户外项目,推荐正价或小幅上浮。",
            "雨": "有雨,户外客流预计下降30%以上,建议启动【雨天特惠+室内项目优先券】组合。",
            "暴雨": "极端降雨,户外项目受限,强烈建议降价+导流至室内场馆,提升二消占比。",
            "酷热": "酷暑高温,建议推出【夜场票】【室内避暑套餐】,错峰运营。",
            "严寒": "严寒天气,客流下降明显,建议捆绑热饮/温泉等体验提升吸引力。",
        }
        day_type_tips = {
            "golden_week": "黄金周:建议限流+高价+高附加值套餐,客流/单客收入双提升。",
            "holiday":     "法定节假日:预计客流旺盛,价格可上浮15-25%,配合增值服务。",
            "weekend":     "周末客流较高,价格温和上浮,重点保障入园体验。",
            "weekday":     "工作日淡峰,建议促销引流(学生/家庭/早鸟票),提高上座率。",
        }

        alerts = ctx.get("alerts", [])
        alert_lines = "\n".join(f"  ⚠️  {a}" for a in alerts) if alerts else "  ✅  无预警"

        text = f"""
═══════════════════════════════════════════════════════════
  AI定价日报 | {ctx.get('date')}
═══════════════════════════════════════════════════════════

📊 核心建议
  • 推荐票价:     ¥{ctx.get('recommended_price', 0):.0f}
  • 预计客流:     {ctx.get('predicted_visitors', 0):,.0f} 人
  • 预计总收入:   ¥{ctx.get('predicted_revenue', 0):,.0f}
  • 预计负载率:   {ctx.get('load_rate', 0)*100:.1f}%

🌤 外部环境
  • 天气:     {ctx.get('weather', '—')}
  • 日期类型: {ctx.get('day_type', '—')}
  • 竞品均价: ¥{ctx.get('competitor_avg', 0):.0f}

💡 运营洞察
  {day_type_tips.get(ctx.get('day_type', ''), '')}
  {weather_tips.get(ctx.get('weather', ''), '')}

🎁 套餐建议
  {ctx.get('bundle_suggestion', '—')}

🚨 预警
{alert_lines}

═══════════════════════════════════════════════════════════
""".strip()
        return text

    # ---------- LLM模式(可选) ----------
    def _llm_brief(self, ctx: dict) -> str:
        """
        调用 LLM API 生成更自然的叙事
        支持:
          - anthropic
          - openai
          - deepseek/moonshot/qwen (通过 OpenAI 兼容接口)
        """
        prompt = self._build_prompt(ctx)

        try:
            if self.provider == "anthropic":
                narrative = self._call_anthropic(prompt)
            elif self.provider in {"openai", "deepseek", "moonshot", "qwen"}:
                narrative = self._call_openai_compatible(prompt)
            else:
                logger.warning(f"未知provider: {self.provider},降级模板模式")
                return self._template_brief(ctx)

            if not narrative:
                logger.warning("LLM返回空内容,降级模板模式")
                return self._template_brief(ctx)
            return narrative + "\n\n" + self._template_brief(ctx)
        except ImportError as e:
            logger.warning(f"{self.provider} SDK未安装: {e},降级模板模式")
            return self._template_brief(ctx)
        except Exception as e:
            logger.error(f"LLM调用失败: {e},降级模板")
            return self._template_brief(ctx)
