import json

from agent.tools import agent_tools


def test_build_cleaning_environment_advice_from_weather(monkeypatch):
    def fake_get_live_weather(city: str) -> dict:
        return {
            "city": "南京市",
            "weather": "小雨",
            "temperature": "28",
            "humidity": "82",
            "winddirection": "东",
            "windpower": "5",
            "reporttime": "2026-06-21 10:00:00",
        }

    monkeypatch.setattr(agent_tools.amap_client, "get_live_weather", fake_get_live_weather)

    payload = json.loads(agent_tools.build_cleaning_environment_advice("南京"))

    assert payload["city"] == "南京市"
    assert payload["humidity_percent"] == 82
    assert payload["cleaning_mode"]["mop"] == "低水量"
    assert payload["cleaning_mode"]["window"] == "清扫时建议关窗"
    assert any("晾干拖布" in item for item in payload["maintenance"])
    assert payload["source"] == "高德开放平台天气服务"


def test_user_location_override(monkeypatch):
    monkeypatch.setenv("ZST_USER_LOCATION", "南京市")

    assert agent_tools.resolve_user_location() == "南京市"
