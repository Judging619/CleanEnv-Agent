import json
import os
from datetime import datetime

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.amap_client import AmapApiError, amap_client
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

rag = RagSummarizeService()

DEFAULT_USER_ID = str(agent_conf.get("default_user_id", "1001"))
DEFAULT_USER_LOCATION = str(agent_conf.get("default_user_location", "南京"))

external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


def clear_last_rag_references():
    rag.clear_last_context_docs()


def get_last_rag_references() -> list[dict]:
    return rag.get_last_references()


def reload_rag_service():
    global rag
    rag = RagSummarizeService()


def resolve_user_location() -> str:
    override_location = os.getenv("ZST_USER_LOCATION", "").strip()
    if override_location:
        return override_location

    try:
        location = amap_client.ip_location()
        city = str(location.get("city", "")).strip()
        if city:
            return city

        province = str(location.get("province", "")).strip()
        if province:
            return province
    except AmapApiError as e:
        logger.error(f"[get_user_location]高德定位失败：{str(e)}")
    except Exception as e:
        logger.error(f"[get_user_location]高德定位失败：{str(e)}", exc_info=True)

    return DEFAULT_USER_LOCATION


def _query_weather_live(city: str) -> tuple[str, dict]:
    query_city = city.strip() if city else ""
    if not query_city:
        query_city = resolve_user_location()
    live = amap_client.get_live_weather(query_city)
    return query_city, live


def _safe_int(value, default: int | None = None) -> int | None:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit() or ch == "-")
    if not digits or digits == "-":
        return default
    try:
        return int(digits)
    except ValueError:
        return default


def build_weather_text(city: str) -> str:
    query_city, live = _query_weather_live(city)
    city_name = str(live.get("city", query_city)).strip() or query_city
    weather = str(live.get("weather", "未知")).strip()
    temperature = str(live.get("temperature", "未知")).strip()
    humidity = str(live.get("humidity", "未知")).strip()
    wind_direction = str(live.get("winddirection", "未知")).strip()
    wind_power = str(live.get("windpower", "未知")).strip()
    report_time = str(live.get("reporttime", "未知")).strip()
    return (
        f"城市{city_name}当前天气为{weather}，气温{temperature}摄氏度，空气湿度{humidity}%"
        f"，风向{wind_direction}，风力{wind_power}级，数据更新时间{report_time}。"
        f"\n参考：高德开放平台天气服务"
    )


def build_cleaning_environment_advice(city: str) -> str:
    query_city, live = _query_weather_live(city)
    city_name = str(live.get("city", query_city)).strip() or query_city
    weather = str(live.get("weather", "未知")).strip()
    temperature = _safe_int(live.get("temperature"))
    humidity = _safe_int(live.get("humidity"))
    wind_power = _safe_int(live.get("windpower"))
    wind_direction = str(live.get("winddirection", "未知")).strip()
    report_time = str(live.get("reporttime", "未知")).strip()

    recommendations = []
    maintenance = []
    cleaning_mode = {
        "vacuum": "标准吸力",
        "mop": "标准水量",
        "window": "正常通风",
    }
    suitability = "适合正常清扫"

    rainy = any(token in weather for token in ["雨", "雪", "雷", "雾"])
    if rainy:
        suitability = "适合清扫，但拖地需要保守设置"
        cleaning_mode["mop"] = "低水量或精细拖地"
        recommendations.append("雨雪或潮湿天气容易把泥水带入室内，建议先吸尘再低水量拖地。")
        maintenance.append("清扫后及时清洗并晾干拖布，避免潮湿发味。")

    if humidity is not None:
        if humidity >= 75:
            suitability = "适合清扫，但需要控制湿拖水量"
            cleaning_mode["mop"] = "低水量"
            recommendations.append("当前湿度偏高，湿拖时建议降低出水量，避免地面长时间不干。")
            maintenance.append("尘盒、滤网和拖布要保持干燥，降低发霉和异味风险。")
        elif humidity <= 35:
            cleaning_mode["vacuum"] = "标准或强力吸力"
            recommendations.append("当前空气偏干，室内容易积尘或产生静电，可适当增加吸尘频率。")
            maintenance.append("建议更频繁清理滤网和边刷，减少细灰堆积。")

    if wind_power is not None and wind_power >= 5:
        cleaning_mode["window"] = "清扫时建议关窗"
        recommendations.append("当前风力较大，开窗可能带入灰尘，建议关窗后再启动清扫。")

    if temperature is not None:
        if temperature >= 32:
            recommendations.append("气温较高时避免机器长时间连续高负载运行，清扫后让机器自然散热再充电。")
        elif temperature <= 5:
            recommendations.append("低温环境下电池续航可能下降，建议在室内常温环境中使用和充电。")

    if not recommendations:
        recommendations.append("当前环境对扫地/扫拖机器人较友好，可按日常模式清扫。")
    if not maintenance:
        maintenance.append("按常规频率清理尘盒、滤网、边刷和拖布即可。")

    payload = {
        "city": city_name,
        "weather": weather,
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "wind_direction": wind_direction,
        "wind_power": str(live.get("windpower", "未知")).strip(),
        "report_time": report_time,
        "suitability": suitability,
        "cleaning_mode": cleaning_mode,
        "recommendations": recommendations,
        "maintenance": maintenance,
        "source": "高德开放平台天气服务",
    }
    return json.dumps(payload, ensure_ascii=False)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    try:
        return build_weather_text(city)
    except AmapApiError as e:
        query_city = city.strip() if city else DEFAULT_USER_LOCATION
        logger.error(f"[get_weather]获取城市{query_city}天气失败：{str(e)}")
        return f"暂时无法获取{query_city}的实时天气信息：{str(e)}"
    except Exception as e:
        query_city = city.strip() if city else DEFAULT_USER_LOCATION
        logger.error(f"[get_weather]获取城市{query_city}天气失败：{str(e)}", exc_info=True)
        return f"暂时无法获取{query_city}的实时天气信息，请稍后重试。"


@tool(description="根据指定城市的实时天气，为扫地/扫拖机器人生成清扫模式、拖地水量、开窗和维护建议")
def get_cleaning_environment_advice(city: str = "") -> str:
    try:
        return build_cleaning_environment_advice(city)
    except AmapApiError as e:
        query_city = city.strip() if city else DEFAULT_USER_LOCATION
        logger.error(f"[get_cleaning_environment_advice]生成{query_city}清扫环境建议失败：{str(e)}")
        return json.dumps(
            {
                "city": query_city,
                "available": False,
                "message": f"暂时无法获取{query_city}的实时环境信息：{str(e)}",
                "fallback_advice": "可按常规清扫模式使用；若室内潮湿或刚下雨，建议降低拖地水量并及时晾干拖布。",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        query_city = city.strip() if city else DEFAULT_USER_LOCATION
        logger.error(f"[get_cleaning_environment_advice]生成{query_city}清扫环境建议失败：{str(e)}", exc_info=True)
        return json.dumps(
            {
                "city": query_city,
                "available": False,
                "message": "暂时无法生成实时清扫环境建议，请稍后重试。",
                "fallback_advice": "可按常规清扫模式使用；若室内潮湿或刚下雨，建议降低拖地水量并及时晾干拖布。",
            },
            ensure_ascii=False,
        )


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return resolve_user_location()


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return os.getenv("ZST_USER_ID", DEFAULT_USER_ID)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


def _sort_months(months: list[str]) -> list[str]:
    return sorted(months)


def _latest_month(user_id: str) -> str | None:
    months = list((external_data.get(user_id) or {}).keys())
    if not months:
        return None
    return _sort_months(months)[-1]


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以JSON字符串形式返回")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    user_records = external_data.get(user_id, {})
    month_record = user_records.get(month)

    if month_record is None:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return json.dumps(
            {
                "found": False,
                "user_id": user_id,
                "month": month,
                "record": None,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "found": True,
            "user_id": user_id,
            "month": month,
            "record": month_record,
        },
        ensure_ascii=False,
    )


@tool(description="获取指定用户可查询的月份列表，以JSON字符串形式返回")
def get_available_months(user_id: str) -> str:
    generate_external_data()
    months = _sort_months(list((external_data.get(user_id) or {}).keys()))
    return json.dumps(
        {
            "user_id": user_id,
            "months": months,
            "count": len(months),
        },
        ensure_ascii=False,
    )


@tool(description="获取指定用户最近一期使用记录，以JSON字符串形式返回")
def fetch_latest_external_data(user_id: str) -> str:
    generate_external_data()
    month = _latest_month(user_id)
    if month is None:
        return json.dumps(
            {
                "found": False,
                "user_id": user_id,
                "month": None,
                "record": None,
            },
            ensure_ascii=False,
        )
    record = external_data[user_id][month]
    return json.dumps(
        {
            "found": True,
            "user_id": user_id,
            "month": month,
            "record": record,
        },
        ensure_ascii=False,
    )


@tool(description="获取指定用户画像摘要与趋势信息，以JSON字符串形式返回")
def get_user_profile(user_id: str) -> str:
    generate_external_data()
    user_records = external_data.get(user_id) or {}
    months = _sort_months(list(user_records.keys()))
    if not months:
        return json.dumps(
            {
                "found": False,
                "user_id": user_id,
                "profile": None,
            },
            ensure_ascii=False,
        )

    latest = months[-1]
    latest_record = user_records[latest]
    return json.dumps(
        {
            "found": True,
            "user_id": user_id,
            "profile": {
                "latest_month": latest,
                "months_count": len(months),
                "feature": latest_record.get("特征", ""),
                "efficiency": latest_record.get("效率", ""),
                "consumables": latest_record.get("耗材", ""),
                "comparison": latest_record.get("对比", ""),
            },
        },
        ensure_ascii=False,
    )


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
