import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from utils.config_handler import agent_conf


class AmapApiError(RuntimeError):
    pass


class AmapClient:
    def __init__(self):
        config_key = str(agent_conf.get("amap_web_service_key", "")).strip()
        self.api_key = os.getenv("AMAP_WEB_SERVICE_KEY", config_key).strip()
        self.timeout_seconds = int(agent_conf.get("amap_timeout_seconds", 8))
        self.base_url = "https://restapi.amap.com"

    def _request_json(self, path: str, params: dict) -> dict:
        if not self.api_key:
            raise AmapApiError(
                "未配置高德API Key。请设置环境变量 AMAP_WEB_SERVICE_KEY 或 config/agent.yml 中 amap_web_service_key。"
            )

        payload = {
            "key": self.api_key,
            "output": "JSON",
            **params,
        }
        url = f"{self.base_url}{path}?{urlencode(payload)}"
        req = Request(url, headers={"User-Agent": "zst-agent/1.0"})

        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                text = resp.read().decode("utf-8")
        except HTTPError as e:
            raise AmapApiError(f"高德API HTTP错误：{e.code}") from e
        except URLError as e:
            raise AmapApiError(f"高德API网络错误：{e.reason}") from e
        except Exception as e:
            raise AmapApiError(f"高德API请求失败：{str(e)}") from e

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise AmapApiError(f"高德API返回非JSON：{text[:120]}") from e

        if str(data.get("status", "")) != "1":
            info = data.get("info", "UNKNOWN")
            infocode = data.get("infocode", "UNKNOWN")
            raise AmapApiError(f"高德API业务错误：{info}（infocode={infocode}）")

        return data

    def ip_location(self) -> dict:
        data = self._request_json("/v3/ip", {})
        return {
            "province": data.get("province", ""),
            "city": data.get("city", ""),
            "adcode": data.get("adcode", ""),
            "rectangle": data.get("rectangle", ""),
        }

    @staticmethod
    def _is_adcode(city: str) -> bool:
        return city.isdigit() and len(city) == 6

    def city_to_adcode(self, city: str) -> str:
        clean_city = city.strip()
        if not clean_city:
            raise AmapApiError("city不能为空")

        if self._is_adcode(clean_city):
            return clean_city

        data = self._request_json(
            "/v3/config/district",
            {
                "keywords": clean_city,
                "subdistrict": 0,
                "extensions": "base",
            },
        )
        districts = data.get("districts", [])
        if not districts:
            raise AmapApiError(f"无法解析城市编码：{clean_city}")

        adcode = str(districts[0].get("adcode", "")).strip()
        if not adcode:
            raise AmapApiError(f"城市缺少adcode：{clean_city}")
        return adcode

    def get_live_weather(self, city: str) -> dict:
        adcode = self.city_to_adcode(city)
        data = self._request_json(
            "/v3/weather/weatherInfo",
            {
                "city": adcode,
                "extensions": "base",
            },
        )
        lives = data.get("lives", [])
        if not lives:
            raise AmapApiError(f"未查询到天气数据，adcode={adcode}")

        live = lives[0]
        live["_query_adcode"] = adcode
        return live


amap_client = AmapClient()
