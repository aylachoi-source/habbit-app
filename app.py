# app.py
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, Tuple, List

import requests
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


# -----------------------------
# Utilities
# -----------------------------
KST = ZoneInfo("Asia/Seoul")


def today_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def calc_achievement(checked: Dict[str, bool]) -> Tuple[int, int]:
    done = sum(1 for v in checked.values() if v)
    total = len(checked)
    rate = round((done / total) * 100) if total else 0
    return done, rate


def init_sample_data() -> List[Dict[str, Any]]:
    """
    Demo 6-day sample data (yesterday back to -6 days),
    and we'll add/overwrite today separately.
    """
    base = datetime.now(KST).date()
    samples = []
    # 6 days sample: -6 .. -1
    # (가볍게 자연스러운 변동치)
    pattern_done = [2, 4, 3, 5, 1, 4]
    pattern_mood = [5, 7, 6, 8, 4, 7]
    for i in range(6, 0, -1):
        d = base - timedelta(days=i)
        done = pattern_done[6 - i]
        mood = pattern_mood[6 - i]
        rate = round((done / 5) * 100)
        samples.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "done": done,
                "rate": rate,
                "mood": mood,
            }
        )
    return samples


def upsert_today_record(done: int, rate: int, mood: int) -> None:
    d = today_str()
    records = st.session_state["records"]
    for r in records:
        if r["date"] == d:
            r.update({"done": done, "rate": rate, "mood": mood})
            return
    records.append({"date": d, "done": done, "rate": rate, "mood": mood})


# -----------------------------
# External APIs
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_weather(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap current weather
    - Korean language
    - Celsius
    - timeout=10
    Returns None on failure
    """
    if not api_key:
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key.strip(),
        "units": "metric",
        "lang": "kr",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Minimal normalized fields
        weather_desc = (data.get("weather") or [{}])[0].get("description")
        icon = (data.get("weather") or [{}])[0].get("icon")
        temp = data.get("main", {}).get("temp")
        feels_like = data.get("main", {}).get("feels_like")
        humidity = data.get("main", {}).get("humidity")
        wind = data.get("wind", {}).get("speed")
        return {
            "city": city,
            "desc": weather_desc,
            "icon": icon,
            "temp_c": temp,
            "feels_like_c": feels_like,
            "humidity": humidity,
            "wind_mps": wind,
        }
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_dog_image() -> Optional[Dict[str, str]]:
    """
    Dog CEO random image
    - timeout=10
    Returns None on failure
    """
    url = "https://dog.ceo/api/breeds/image/random"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("status") != "success":
            return None
        image_url = data.get("message")
        if not image_url:
            return None

        # Try to parse breed from URL pattern: .../breeds/{breed}/...
        breed = "알 수 없음"
        m = re.search(r"/breeds/([^/]+)/", image_url)
        if m:
            raw = m.group(1)  # e.g., "retriever-golden" or "bulldog-french"
            parts = raw.split("-")
            # Heuristic: "breed-subbreed" => "breed (subbreed)"
            if len(parts) >= 2:
                breed = f"{parts[0]} ({'-'.join(parts[1:])})"
            else:
                breed = raw

        return {"url": image_url, "breed": breed}
    except Exception:
        return None


def generate_report(
    openai_api_key: str,
    coach_style: str,
    habits: Dict[str, bool],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog: Optional[Dict[str, str]],
) -> Optional[str]:
    """
    OpenAI call:
    - model: gpt-5-mini
    - system prompt depends on coach_style
    - returns None on failure
    """
    if not openai_api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        st.error("openai 패키지가 필요합니다. `pip install openai` 후 다시 실행하세요.")
        return None

    style_map = {
        "스파르타 코치": (
            "너는 엄격한 스파르타 코치다. 칭찬은 짧게, 기준은 높게. "
            "애매한 말 금지. 행동 지침을 구체적으로 제시해라."
        ),
        "따뜻한 멘토": (
            "너는 따뜻하고 현실적인 멘토다. 공감과 격려를 하되, 실행 가능한 작은 다음 행동을 제시해라."
        ),
        "게임 마스터": (
            "너는 RPG 게임 마스터다. 사용자의 하루를 퀘스트/스탯/보상 관점으로 해석하고, 내일 미션을 퀘스트처럼 제시해라."
        ),
    }
    system_prompt = style_map.get(coach_style, style_map["따뜻한 멘토"])

    # Build compact context
    habit_lines = "\n".join([f"- {k}: {'완료' if v else '미완료'}" for k, v in habits.items()])
    weather_line = "날씨 정보 없음"
    if weather:
        weather_line = (
            f"{weather.get('city')} / {weather.get('desc')} / "
            f"{weather.get('temp_c')}°C(체감 {weather.get('feels_like_c')}°C) / "
            f"습도 {weather.get('humidity')}% / 바람 {weather.get('wind_mps')}m/s"
        )
    dog_line = "강아지 정보 없음"
    if dog:
        dog_line = f"품종 추정: {dog.get('breed')}"

    user_prompt = f"""
[오늘 기록]
날짜: {today_str()}
기분(1~10): {mood}

[습관 체크]
{habit_lines}

[날씨]
{weather_line}

[랜덤 강아지]
{dog_line}

아래 형식으로만 답해:
컨디션 등급: (S/A/B/C/D 중 1개)
습관 분석: (핵심 3~5줄)
날씨 코멘트: (1~2줄)
내일 미션: (체크리스트 3개, 매우 구체적으로)
오늘의 한마디: (한 문장)
""".strip()

    try:
        client = OpenAI(api_key=openai_api_key.strip())
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else None
    except Exception:
        return None


# -----------------------------
# Sidebar: API keys
# -----------------------------
st.sidebar.header("🔑 API 설정")
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=st.session_state.get("openai_api_key", ""),
    placeholder="sk-...",
)
owm_api_key = st.sidebar.text_input(
    "OpenWeatherMap API Key",
    type="password",
    value=st.session_state.get("owm_api_key", ""),
    placeholder="OpenWeatherMap Key",
)

st.session_state["openai_api_key"] = openai_api_key
st.session_state["owm_api_key"] = owm_api_key

st.sidebar.divider()
st.sidebar.caption("키는 브라우저 세션(session_state)에만 저장됩니다.")


# -----------------------------
# Main UI: Habit check-in
# -----------------------------
st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관과 컨디션을 기록하고, AI 코치 리포트를 받아보세요.")

if "records" not in st.session_state:
    st.session_state["records"] = init_sample_data()

habit_meta = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

st.subheader("✅ 습관 체크인")

c1, c2 = st.columns(2)
checked: Dict[str, bool] = {}

with c1:
    for emoji, name in habit_meta[:3]:
        checked[name] = st.checkbox(f"{emoji} {name}", key=f"hb_{name}")
with c2:
    for emoji, name in habit_meta[3:]:
        checked[name] = st.checkbox(f"{emoji} {name}", key=f"hb_{name}")

mood = st.slider("🙂 오늘 기분은 어떤가요?", min_value=1, max_value=10, value=7, step=1)

cities = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Ulsan", "Suwon", "Jeju", "Sejong",
]
city = st.selectbox("📍 도시 선택", cities, index=0)

coach_style = st.radio(
    "🧑‍🏫 코치 스타일",
    ["스파르타 코치", "따뜻한 멘토", "게임 마스터"],
    horizontal=True,
)

done, rate = calc_achievement(checked)
upsert_today_record(done=done, rate=rate, mood=mood)


# -----------------------------
# Metrics + Chart
# -----------------------------
st.subheader("📈 오늘 요약")

m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{rate}%")
m2.metric("달성 습관", f"{done}/5")
m3.metric("기분", f"{mood}/10")

# Build 7-day dataframe: last 6 days sample + today
records = st.session_state["records"]
df = pd.DataFrame(records).copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")

# Keep last 7
df7 = df.tail(7).copy()
df7["date_label"] = df7["date"].dt.strftime("%m/%d")
df7 = df7.set_index("date_label")

st.subheader("🗓️ 7일 달성률(%)")
st.bar_chart(df7[["rate"]])


# -----------------------------
# Generate report
# -----------------------------
st.subheader("🧾 AI 코치 리포트")

btn = st.button("컨디션 리포트 생성", type="primary")

weather_data = None
dog_data = None
report_text = None

if btn:
    with st.spinner("날씨/강아지/AI 리포트 생성 중..."):
        weather_data = get_weather(city=city, api_key=owm_api_key)
        dog_data = get_dog_image()
        report_text = generate_report(
            openai_api_key=openai_api_key,
            coach_style=coach_style,
            habits=checked,
            mood=mood,
            weather=weather_data,
            dog=dog_data,
        )

    # Results display
    r1, r2 = st.columns(2)

    with r1:
        st.markdown("### 🌤️ 오늘의 날씨")
        if weather_data:
            st.markdown(
                f"""
**{weather_data.get('city')}**
- 상태: {weather_data.get('desc')}
- 기온: {weather_data.get('temp_c')}°C (체감 {weather_data.get('feels_like_c')}°C)
- 습도: {weather_data.get('humidity')}%
- 바람: {weather_data.get('wind_mps')} m/s
""".strip()
            )
        else:
            st.info("날씨 정보를 불러오지 못했습니다. (API Key/도시/네트워크 확인)")

    with r2:
        st.markdown("### 🐶 오늘의 강아지")
        if dog_data:
            st.caption(f"품종 추정: {dog_data.get('breed')}")
            st.image(dog_data.get("url"), use_container_width=True)
        else:
            st.info("강아지 이미지를 불러오지 못했습니다. (네트워크 확인)")

    st.markdown("### 🤖 리포트")
    if report_text:
        st.markdown(report_text)
    else:
        st.warning("AI 리포트를 생성하지 못했습니다. (OpenAI API Key/네트워크/패키지 확인)")

    # Share text
    st.markdown("### 🔗 공유용 텍스트")
    habit_summary = ", ".join([f"{'✅' if v else '⬜'} {k}" for k, v in checked.items()])
    weather_share = "날씨: 없음"
    if weather_data:
        weather_share = f"날씨: {weather_data.get('desc')} / {weather_data.get('temp_c')}°C"
    dog_share = "강아지: 없음"
    if dog_data:
        dog_share = f"강아지: {dog_data.get('breed')}"
    share = f"""[AI 습관 트래커] {today_str()}
달성률: {rate}% ({done}/5) / 기분: {mood}/10
{habit_summary}
{weather_share}
{dog_share}

리포트:
{report_text or '(리포트 생성 실패)'}
"""
    st.code(share, language="text")


# -----------------------------
# Footer: API 안내
# -----------------------------
with st.expander("📌 API 안내 / 설정 방법"):
    st.markdown(
        """
**OpenAI API Key**
- OpenAI 계정에서 API 키를 발급받아 사이드바에 입력하세요.
- 파이썬 라이브러리 필요: `pip install openai`

**OpenWeatherMap API Key**
- OpenWeatherMap에서 API 키를 발급받아 사이드바에 입력하세요.
- 도시 선택은 기본적으로 영문 표기(Seoul, Busan 등)를 사용합니다.

**네트워크/실패 처리**
- 날씨/강아지 API는 `timeout=10`이며 실패 시 `None`을 반환하도록 구성되어 있습니다.
- 키가 비어 있거나 호출에 실패하면 카드에 안내 메시지가 표시됩니다.
""".strip()
    )
