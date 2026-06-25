import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(
    page_title="LG CODEZERO 청소기 대시보드",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_data():
    chunk_df = pd.read_csv('chunk_umap.csv', encoding='utf-8-sig')
    image_df = pd.read_csv('image_umap.csv', encoding='utf-8-sig')
    return chunk_df, image_df

chunk_df, image_df = load_data()

label_map = dict(zip(
    chunk_df.dropna(subset=['text_cluster_label'])['text_cluster'].astype(int),
    chunk_df.dropna(subset=['text_cluster_label'])['text_cluster_label']
))

CLUSTER_COLORS = {
    0:'#3B8BD4', 1:'#E24B4A', 2:'#1D9E75',
    3:'#EF9F27', 4:'#D4537E', 5:'#7F77DD',
    6:'#5DCAA5', 7:'#F0997B', 8:'#888780', -1:'#cccccc'
}

SUBCLUSTER_NAMES = {
    'T1_I0': '정품 배터리 관리형',
    'T1_I1': '배터리 교체 탐색형',
    'T1_I2': '호환 배터리 구매형 🔴',
    'T3_I0': '공식 AS 경험형',
    'T3_I1': '비공식 수리업체 이용형 🔴',
    'T3_I2': '부품 자가수리형',
}

GITHUB_USER = "juhee-kk"
REPO_NAME = "lg-dashboard"
BRANCH = "main"

def img_url(subcluster, idx):
    return f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/images/{subcluster}/img_{idx}.jpg"

PREV_WEEK = {
    0:68, 1:180, 2:145, 3:95,
    4:92, 5:118, 6:61, 7:103, 8:8
}

@st.cache_data
def calc_volume_change(image_df):
    this_week = image_df[image_df['text_cluster'] != -1].groupby(
        ['text_cluster','text_cluster_label']
    ).size().reset_index(name='this_week')
    this_week['text_cluster'] = this_week['text_cluster'].astype(int)
    this_week['prev_week'] = this_week['text_cluster'].map(PREV_WEEK)
    this_week['change'] = this_week['this_week'] - this_week['prev_week']
    this_week['change_rate'] = ((this_week['change'] / this_week['prev_week']) * 100).round(1)
    return this_week.sort_values('change_rate', ascending=False)

vol_df = calc_volume_change(image_df)
top_surge = vol_df.iloc[0]
top_new = vol_df[vol_df['text_cluster'] == 3].iloc[0]

# ── 세션 상태 ────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None

# ── LLM 함수 ────────────────────────────────────────────
def call_groq(api_key, messages):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.2
        }
    )
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"API 호출 실패: {response.status_code}"

# ════════════════════════════════════════════════════════
# PAGE 1 — 전체 현황
# ════════════════════════════════════════════════════════
def page1():
    st.markdown("## LG 청소기 대시보드")
    st.caption("플랫폼: 블로그 · 멀티모달 CEJ 분석 · 텍스트 군집 9개 · 이미지 서브군집 26개")
    st.divider()

    # ── KPI 카드 ─────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("텍스트 군집", f"{len(vol_df)}개")
    k2.metric("분석 이미지", f"{len(image_df):,}개")
    k3.metric("볼륨 급증", f"T1 +{top_surge['change_rate']}%", delta="즉시 대응", delta_color="inverse")
    k4.metric("주목 군집", "T3 +45%", delta="모니터링", delta_color="inverse")

    st.divider()

    # ── 시그널 요약 ──────────────────────────────────────
    st.markdown("### 이번 주 CEJ 시그널 요약")
    st.caption("2026.06.24 기준 · 이전 주 대비 볼륨 변화율 기준")

    s1, s2 = st.columns(2)

    with s1:
        st.error("🔴 **즉시 대응**")
        st.markdown("""
**T1 배터리 교체 (호환 배터리)**

이번 주 **313건** (지난 주 180건)

변화율: **+74% 급증**

이미지에서 호환 배터리 패턴 집중 감지
        """)

    with s2:
        st.warning("🟡 **모니터링**")
        st.markdown("""
**T3 고장/AS 서비스센터 경험**

이번 주 **138건** (지난 주 95건)

변화율: **+45% 증가**

비공식 수리업체 이미지 반복 등장
        """)

    st.divider()

    # ── CEJ 타임라인 ─────────────────────────────────────
    st.markdown("### CEJ 단계별 소비자 목소리 분포")
    st.caption("소비자 목소리가 집중되는 단계 = 개입 타이밍")

    cej_order = ['구매', '설치', '사용', '관리', '교체']
    cej_counts = {}

    if 'cej_stage' in chunk_df.columns:
        for stage in cej_order:
            count = len(chunk_df[
                (chunk_df['cej_stage'] == stage) &
                (chunk_df['text_cluster'] != -1)
            ])
            cej_counts[stage] = count
    else:
        cej_counts = {'구매': 173, '설치': 0, '사용': 285, '관리': 451, '교체': 9}

    colors = []
    for stage in cej_order:
        count = cej_counts.get(stage, 0)
        if count >= 400:
            colors.append('#E24B4A')
        elif count >= 200:
            colors.append('#EF9F27')
        else:
            colors.append('#3B8BD4')

    fig_cej = go.Figure()

    fig_cej.add_trace(go.Bar(
        x=cej_order,
        y=[cej_counts.get(s, 0) for s in cej_order],
        marker_color=colors,
        text=[f"{cej_counts.get(s,0)}건" for s in cej_order],
        textposition='outside',
    ))

    fig_cej.add_annotation(
        x='관리', y=cej_counts.get('관리', 0) + 30,
        text="⚠️ 집중 구간 — 지금 개입 타이밍",
        showarrow=False,
        font=dict(color='#E24B4A', size=12)
    )

    fig_cej.update_layout(
        height=280,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
        showlegend=False
    )
    st.plotly_chart(fig_cej, use_container_width=True)

    st.divider()

    # ── 군집 선택 ─────────────────────────────────────────
    st.markdown("### 텍스트 군집 탐색")
    options = ['전체 보기'] + [
        f"T{int(row['text_cluster'])}: {row['text_cluster_label']}"
        for _, row in vol_df.iterrows()
    ]
    selected = st.selectbox("군집 선택 (볼륨 변화율 순)", options)
    selected_cluster = None if selected == '전체 보기' else int(
        selected.split(':')[0].replace('T', '')
    )

    # ── UMAP 두 개 ────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**텍스트 군집 UMAP**")
        st.caption("텍스트만 보면 하나의 군집")
        plot_df = chunk_df.copy()
        plot_df['label'] = plot_df['text_cluster'].map(label_map).fillna('Noise')
        fig1 = px.scatter(
            plot_df, x='umap_x', y='umap_y',
            color='label', opacity=0.8, height=350,
            hover_data={'umap_x': False, 'umap_y': False, 'label': True}
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(font=dict(size=9), title='')
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**이미지 서브군집 UMAP**")
        if selected_cluster is not None:
            st.caption(f"T{selected_cluster} → 이미지로 보면 이렇게 분화됩니다")
            img_plot = image_df[image_df['text_cluster'] == selected_cluster]
        else:
            st.caption("이미지로 보면 이렇게 갈립니다")
            img_plot = image_df[image_df['text_cluster'] != -1]

        fig2 = px.scatter(
            img_plot, x='img_umap_x', y='img_umap_y',
            color='image_subcluster_global',
            opacity=0.8, height=350,
            hover_data={'img_umap_x': False, 'img_umap_y': False,
                        'image_subcluster_global': True}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(font=dict(size=9), title='')
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── 볼륨 변화율 차트 ──────────────────────────────────
    st.markdown("### 군집별 볼륨 변화율 (이번 주 vs 지난 주)")
    vol_sorted = vol_df.sort_values('change_rate', ascending=True)
    fig3 = px.bar(
        vol_sorted,
        x='change_rate', y='text_cluster_label',
        orientation='h', height=300,
        color='change_rate',
        color_continuous_scale=['#1D9E75', '#EF9F27', '#E24B4A'],
        labels={'change_rate': '변화율 (%)', 'text_cluster_label': ''}
    )
    fig3.add_vline(x=0, line_color='gray', line_width=1)
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        yaxis=dict(tickfont=dict(size=9))
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── 대표 이미지 (T1, T3) ──────────────────────────────
    if selected_cluster in [1, 3]:
        tc_label = label_map.get(selected_cluster, '')
        st.markdown(f"### T{selected_cluster}: {tc_label} — 이미지 서브군집별 대표 이미지")
        st.caption("텍스트는 하나의 군집 → 이미지로 보면 다른 패턴으로 분화")

        subclusters = sorted(
            image_df[image_df['text_cluster'] == selected_cluster]
            ['image_subcluster_global'].unique()
        )
        sub_counts = image_df[
            image_df['text_cluster'] == selected_cluster
        ].groupby('image_subcluster_global').size()

        for sub_label in subclusters:
            count = sub_counts.get(sub_label, 0)
            sub_name = SUBCLUSTER_NAMES.get(sub_label, sub_label)
            st.markdown(f"**{sub_label} — {sub_name}** ({count}건)")

            img_cols = st.columns(5)
            for j in range(5):
                with img_cols[j]:
                    st.image(img_url(sub_label, j), use_column_width=True)
            st.markdown("---")

    st.divider()

    # ── Page 2 이동 버튼 ──────────────────────────────────
    st.markdown("### 인사이트 도출")
    st.caption("군집을 선택하고 멀티모달 인사이트를 도출합니다")

    if st.button("🔍 군집 분석 & 인사이트 도출하기", type="primary", use_container_width=True):
        st.session_state.page = 2
        st.session_state.selected_cluster = selected_cluster
        st.rerun()

    st.caption("* 이전 주 볼륨은 시뮬레이션 데이터 · 실제 운영 시 주간 파이프라인으로 자동 갱신")


# ════════════════════════════════════════════════════════
# PAGE 2 — 인사이트 도출
# ════════════════════════════════════════════════════════
def page2():
    if st.button("← 대시보드로 돌아가기"):
        st.session_state.page = 1
        st.rerun()

    st.markdown("## 멀티모달 인사이트 도출")
    st.caption("텍스트 군집 + 이미지 서브군집 → LLM 해석 → 인사이트 확정")
    st.divider()

    # ── LLM 추천 ─────────────────────────────────────────
    st.markdown("### LLM 추천 — 볼륨 변화 기준 주목 군집")
    st.info("""
**볼륨 변화율 분석 결과, 아래 이미지 서브군집을 우선 검토하세요.**

- 🔴 **T1_I2 (호환 배터리 구매형)** — T1 군집 내 볼륨 184건으로 압도적. 서드파티 패턴 집중.
- 🔴 **T3_I1 (비공식 수리업체 이용형)** — T3 군집 내 볼륨 84건. 비공식 채널 이탈 신호.
    """)

    st.divider()

    # ── 이미지 서브군집 선택 ──────────────────────────────
    st.markdown("### 이미지 서브군집 선택")
    st.caption("분석할 이미지 서브군집을 선택하세요 (복수 선택 가능)")

    all_subclusters = {
        'T1_I0': ('T1: 배터리 교체', '정품 배터리 관리형', 54),
        'T1_I1': ('T1: 배터리 교체', '배터리 교체 탐색형', 75),
        'T1_I2': ('T1: 배터리 교체', '호환 배터리 구매형 🔴', 184),
        'T3_I0': ('T3: 고장/AS 경험', '공식 AS 경험형', 37),
        'T3_I1': ('T3: 고장/AS 경험', '비공식 수리업체 이용형 🔴', 84),
        'T3_I2': ('T3: 고장/AS 경험', '부품 자가수리형', 17),
    }

    selected_subs = []

    t1_col, t3_col = st.columns(2)

    with t1_col:
        st.markdown("**T1: 배터리 교체 (호환 배터리)**")
        for sub_id in ['T1_I0', 'T1_I1', 'T1_I2']:
            _, name, count = all_subclusters[sub_id]
            default = sub_id in ['T1_I2']
            checked = st.checkbox(
                f"{sub_id} — {name} ({count}건)",
                value=default,
                key=f"check_{sub_id}"
            )
            if checked:
                selected_subs.append(sub_id)

    with t3_col:
        st.markdown("**T3: 고장/AS 서비스센터 경험**")
        for sub_id in ['T3_I0', 'T3_I1', 'T3_I2']:
            _, name, count = all_subclusters[sub_id]
            default = sub_id in ['T3_I1']
            checked = st.checkbox(
                f"{sub_id} — {name} ({count}건)",
                value=default,
                key=f"check_{sub_id}"
            )
            if checked:
                selected_subs.append(sub_id)

    st.divider()

    # ── 선택한 서브군집 대표 이미지 ───────────────────────
    if selected_subs:
        st.markdown("### 선택한 이미지 서브군집 대표 이미지")
        for sub_id in selected_subs:
            _, name, count = all_subclusters[sub_id]
            st.markdown(f"**{sub_id} — {name}** ({count}건)")
            img_cols = st.columns(5)
            for j in range(5):
                with img_cols[j]:
                    st.image(img_url(sub_id, j), use_column_width=True)
            st.markdown("---")

    st.divider()

    # ── LLM 대화형 인사이트 ───────────────────────────────
    st.markdown("### LLM 멀티모달 해석")

    groq_key = st.text_input(
        "Groq API Key (선택)",
        type="password",
        placeholder="gsk_... 입력 시 실제 LLM 호출 / 비워두면 예시 결과 표시"
    )

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # 하드코딩 대화 예시
    HARDCODED_CONVERSATION = [
        {
            "q": "현재 볼륨 변화 기준으로 눈에 띄는 멀티모달 변화를 알려줘",
            "a": """T1_I2 군집에서 **GIGACELL 노란 배터리팩 단독샷**이 압도적으로 증가하고 있습니다.

텍스트만 보면 "배터리 교체했어요"로만 읽히지만, 이미지를 보면 **어떤 브랜드로 이탈했는지 특정**이 가능합니다.

동시에 T3_I1에서는 **에어킹 등 비공식 수리업체 명함/배너**가 반복 등장합니다. 텍스트에는 "수리했어요"만 있고 어디서 수리했는지 나오지 않지만, 이미지에서 채널명이 확인됩니다."""
        },
        {
            "q": "마케팅적 활용 아이디어?",
            "a": """두 가지 방향을 제안합니다.

**1. 정품 구독 선점 전략**
배터리 수명 종료 시점(구매 후 약 11~14개월)에 ThinQ 앱 푸시로 정품 배터리 구독 오퍼를 선제 발송합니다. 호환 배터리 검색이 시작되기 전에 LG가 먼저 접점을 만드는 구조입니다.

**2. 비공식 업체 검색 차단**
네이버 블로그 검색에서 비공식 업체 콘텐츠가 상위 노출되고 있습니다. LG 공식 AS 관련 콘텐츠 SEO 강화 + 블로그 광고 입찰가 상향으로 공식 채널 노출을 선점해야 합니다."""
        },
        {
            "q": "이미지 분석이 없었으면 몰랐을 것은?",
            "a": """텍스트만 분석했다면 아래 세 가지를 놓쳤을 것입니다.

1. **이탈 브랜드 특정 불가** — "호환 배터리 샀어요"는 텍스트에 있지만, GIGACELL이라는 구체적 브랜드는 이미지에서만 확인됩니다.

2. **비공식 업체 채널명 파악 불가** — "수리했어요"만 있고 에어킹, 필터포유 등 업체명은 이미지 속 명함/배너에서만 나옵니다.

3. **이탈 규모 실감 불가** — 텍스트 빈도만으로는 서드파티가 얼마나 깊이 침투했는지 체감이 어렵지만, 이미지 군집 볼륨(T1_I2 184건)으로 규모가 시각화됩니다."""
        }
    ]

    # 대화 표시
    if not st.session_state.chat_history:
        st.session_state.chat_history = HARDCODED_CONVERSATION.copy()

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['q'])
        with st.chat_message("assistant"):
            st.markdown(chat['a'])

    # 실제 LLM 입력창
    user_input = st.chat_input("추가로 물어보고 싶은 것을 입력하세요")
    if user_input:
        if groq_key:
            messages = [
                {"role": "system", "content": "너는 LG전자 HS본부 멀티모달 VOC 분석 보조자다. 텍스트 근거와 이미지 근거를 구분해서 한국어로 답해라."},
            ]
            for chat in st.session_state.chat_history:
                messages.append({"role": "user", "content": chat['q']})
                messages.append({"role": "assistant", "content": chat['a']})
            messages.append({"role": "user", "content": user_input})

            with st.spinner("LLM 답변 생성 중..."):
                answer = call_groq(groq_key, messages)
        else:
            answer = "Groq API Key를 입력하면 실제 LLM이 답변합니다."

        st.session_state.chat_history.append({'q': user_input, 'a': answer})
        st.rerun()

    st.divider()

    # ── 최종 인사이트 카드 ────────────────────────────────
    st.markdown("### 최종 인사이트")

    st.success("""
**LG 청소기 사후관리 생태계를 서드파티가 잠식하고 있다**

텍스트 VOC만 보면 배터리 불만과 AS 불만이 각각 개별 문제로 보입니다.

이미지 분석 결과, T1_I2에서 GIGACELL 호환 배터리, T3_I1에서 에어킹 비공식 수리업체가 반복 등장합니다.

두 군집을 합쳐서 보면 **배터리 · 헤드 · 분해청소 3개 영역 전부 서드파티가 선점**하고 있음이 확인됩니다.

**텍스트만 봤으면 각각의 불만으로만 읽혔을 것입니다.**
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**멀티모달 Delta**")
        st.markdown("""
- 이탈 브랜드 특정 → 이미지에서만 가능
- 비공식 채널명 파악 → 이미지에서만 가능
- 3개 영역 동시 잠식 → 이미지 합산으로 발견
        """)

    with col2:
        st.markdown("**LG 액션 제안**")
        st.markdown("""
- 정품 배터리 구독 서비스 출시
- ThinQ 앱 구매 후 11개월 시점 알림
- 공식 AS 블로그 SEO 강화
- 케어십 서비스 헤드 수리 포함 확대
        """)

    st.divider()

    # ── 사람 최종 판단 ────────────────────────────────────
    st.markdown("### 사람 최종 판단")

    with st.form("decision_form"):
        final_insight = st.text_area(
            "최종 인사이트 제목",
            value="LG 청소기 사후관리 생태계 서드파티 잠식 — 배터리·AS 3영역 동시 이탈"
        )
        action = st.text_area(
            "LG 액션 제안",
            value="정품 배터리 구독 서비스 + ThinQ 앱 알림 + 케어십 서비스 확대"
        )
        priority = st.selectbox("우선순위", ["높음", "중간", "낮음"])
        confidence = st.slider("근거 신뢰도", 1, 5, 4)
        submitted = st.form_submit_button("인사이트 확정 & 저장", type="primary")

        if submitted:
            st.success("✅ 인사이트 저장 완료!")
            st.markdown(f"""
**저장된 인사이트**

- 제목: {final_insight}
- 액션: {action}
- 우선순위: {priority}
- 신뢰도: {confidence}/5
- 분석 군집: {', '.join(selected_subs)}
            """)

# ── 라우터 ────────────────────────────────────────────────
if st.session_state.page == 1:
    page1()
elif st.session_state.page == 2:
    page2()
