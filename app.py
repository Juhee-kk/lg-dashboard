import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(
    page_title="LG 청소기 대시보드",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container { padding-top: 0.5rem; padding-bottom: 0rem; }
div[data-testid="metric-container"] { padding: 4px 8px; }
.stPlotlyChart { margin: 0; padding: 0; }
h3 { margin-bottom: 4px; }
hr { margin: 6px 0; }
.voice-card {
    background: var(--background-color);
    border-left: 3px solid #E24B4A;
    padding: 6px 10px;
    margin-bottom: 6px;
    border-radius: 4px;
    font-size: 11px;
    line-height: 1.5;
}
.signal-red {
    background: #fff1f0;
    border-left: 4px solid #E24B4A;
    padding: 8px 12px;
    border-radius: 4px;
    margin-bottom: 6px;
    font-size: 11px;
}
.signal-yellow {
    background: #fffbe6;
    border-left: 4px solid #EF9F27;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 11px;
}
</style>
""", unsafe_allow_html=True)

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
    this_week['change_rate'] = (
        (this_week['change'] / this_week['prev_week']) * 100
    ).round(1)
    return this_week.sort_values('change_rate', ascending=False)

vol_df = calc_volume_change(image_df)

if 'page' not in st.session_state:
    st.session_state.page = 1
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None

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
    return f"API 호출 실패: {response.status_code}"

# ════════════════════════════════════════════════════════
# PAGE 1
# ════════════════════════════════════════════════════════
def page1():

    # 제목
    st.markdown("### 🔍 LG 청소기 CEJ 대시보드 — 플랫폼: 블로그")
    st.divider()

    # ── 1행: KPI + 시그널 / CEJ 타임라인 ──────────────────
    row1_left, row1_right = st.columns([1.2, 1])

    with row1_left:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("군집 수", "9개")
        k2.metric("분석 이미지", "1,095개")
        k3.metric("🔴 급증", "T1 +74%")
        k4.metric("🟡 주목", "T3 +45%")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='signal-red'>
        🔴 <b>즉시 대응</b> — T1 배터리 교체 (호환 배터리)<br>
        이번 주 <b>313건</b> (지난 주 180건) · <b>+74% 급증</b><br>
        이미지에서 호환 배터리 패턴 집중 감지
        </div>
        <div class='signal-yellow'>
        🟡 <b>모니터링</b> — T3 고장/AS 서비스센터 경험<br>
        이번 주 <b>138건</b> (지난 주 95건) · <b>+45% 증가</b><br>
        비공식 수리업체 이미지 반복 등장
        </div>
        """, unsafe_allow_html=True)

    with row1_right:
        st.markdown("**CEJ 단계별 소비자 목소리 분포**")

        cej_order = ['구매', '설치', '사용', '관리', '교체']
        if 'cej_stage' in chunk_df.columns:
            cej_counts = {
                s: len(chunk_df[
                    (chunk_df['cej_stage']==s) &
                    (chunk_df['text_cluster']!=-1)
                ]) for s in cej_order
            }
        else:
            cej_counts = {'구매':173,'설치':0,'사용':285,'관리':451,'교체':9}

        colors = [
            '#E24B4A' if cej_counts.get(s,0)>=400
            else '#EF9F27' if cej_counts.get(s,0)>=200
            else '#3B8BD4'
            for s in cej_order
        ]

        fig_cej = go.Figure(go.Bar(
            x=cej_order,
            y=[cej_counts.get(s,0) for s in cej_order],
            marker_color=colors,
            text=[f"{cej_counts.get(s,0)}건" for s in cej_order],
            textposition='outside',
            textfont=dict(size=10)
        ))
        fig_cej.add_annotation(
            x='관리', y=cej_counts.get('관리',0)+20,
            text="⚠️ 지금 개입 타이밍",
            showarrow=False,
            font=dict(color='#E24B4A', size=10)
        )
        fig_cej.update_layout(
            height=200,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0,r=0,t=30,b=0),
            xaxis=dict(showgrid=False, tickfont=dict(size=11)),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)',
                      showticklabels=False),
            showlegend=False
        )
        st.plotly_chart(fig_cej, use_container_width=True)

    st.divider()

    # ── 2행: 군집 선택 + UMAP 두 개 ──────────────────────
    sel_col, u1_col, u2_col = st.columns([0.7, 1, 1])

    with sel_col:
        st.markdown("**군집 선택**")
        st.caption("볼륨 변화율 순")
        options = ['전체'] + [
            f"T{int(r['text_cluster'])}: {r['text_cluster_label'][:10]}"
            for _, r in vol_df.iterrows()
        ]
        selected = st.radio("", options, label_visibility='collapsed')
        selected_cluster = None if selected == '전체' else int(
            selected.split(':')[0].replace('T','')
        )
        st.session_state.selected_cluster = selected_cluster

    with u1_col:
        st.markdown("**텍스트 군집 UMAP**")
        st.caption("텍스트만 보면 하나의 군집")
        plot_df = chunk_df.copy()
        plot_df['label'] = plot_df['text_cluster'].map(label_map).fillna('Noise')
        fig1 = px.scatter(
            plot_df, x='umap_x', y='umap_y',
            color='label', opacity=0.7, height=240,
            hover_data={'umap_x':False,'umap_y':False,'label':True}
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            legend=dict(font=dict(size=7),title='')
        )
        st.plotly_chart(fig1, use_container_width=True)

    with u2_col:
        st.markdown("**이미지 서브군집 UMAP**")
        st.caption("이미지로 보면 이렇게 갈림")
        img_plot = (
            image_df[image_df['text_cluster']==selected_cluster]
            if selected_cluster is not None
            else image_df[image_df['text_cluster']!=-1]
        )
        fig2 = px.scatter(
            img_plot, x='img_umap_x', y='img_umap_y',
            color='image_subcluster_global',
            opacity=0.7, height=240,
            hover_data={'img_umap_x':False,'img_umap_y':False,
                       'image_subcluster_global':True}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            legend=dict(font=dict(size=7),title='')
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── 3행: 볼륨 변화율 / 소비자 목소리 / 대표 이미지 ────
    b1, b2, b3 = st.columns([1, 1, 1.5])

    with b1:
        st.markdown("**군집별 볼륨 변화율**")
        vol_sorted = vol_df.sort_values('change_rate', ascending=True)
        fig3 = px.bar(
            vol_sorted,
            x='change_rate', y='text_cluster_label',
            orientation='h', height=220,
            color='change_rate',
            color_continuous_scale=['#1D9E75','#EF9F27','#E24B4A'],
            labels={'change_rate':'변화율(%)','text_cluster_label':''}
        )
        fig3.add_vline(x=0, line_color='gray', line_width=1)
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0,r=0,t=0,b=0),
            coloraxis_showscale=False,
            yaxis=dict(tickfont=dict(size=8))
        )
        st.plotly_chart(fig3, use_container_width=True)

    with b2:
        st.markdown("**소비자가 실제로 한 말**")
        if selected_cluster is not None:
            sub_df = chunk_df[
                chunk_df['text_cluster'] == selected_cluster
            ]['chunk_text'].dropna()
            samples = sub_df.sample(
                min(3, len(sub_df)), random_state=42
            ).tolist()
            for text in samples:
                clean = str(text)[:120].encode(
                    'utf-8', errors='ignore'
                ).decode('utf-8')
                st.markdown(
                    f"<div class='voice-card'>💬 {clean}...</div>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("왼쪽에서 군집을 선택하면")
            st.caption("대표 텍스트가 표시됩니다")

    with b3:
        if selected_cluster in [1, 3]:
            tc_label = label_map.get(selected_cluster,'')
            st.markdown(f"**T{selected_cluster}: {tc_label} — 이미지 서브군집**")

            subclusters = sorted(
                image_df[image_df['text_cluster']==selected_cluster]
                ['image_subcluster_global'].unique()
            )
            sub_counts = image_df[
                image_df['text_cluster']==selected_cluster
            ].groupby('image_subcluster_global').size()

            for sub_label in subclusters:
                count = sub_counts.get(sub_label, 0)
                name = SUBCLUSTER_NAMES.get(sub_label, sub_label)
                st.markdown(
                    f"<span style='font-size:10px;font-weight:600'>"
                    f"{sub_label} — {name} ({count}건)</span>",
                    unsafe_allow_html=True
                )
                cols = st.columns(5)
                for j in range(5):
                    with cols[j]:
                        st.image(
                            img_url(sub_label, j),
                            use_column_width=True
                        )
        else:
            st.caption("T1 또는 T3 선택 시 대표 이미지 표시")

    st.divider()

    if st.button(
        "🔍 인사이트 도출하기 →",
        type="primary",
        use_container_width=True
    ):
        st.session_state.page = 2
        st.rerun()

    st.caption(
        "* 이전 주 볼륨은 시뮬레이션 · "
        "실제 운영 시 주간 파이프라인으로 자동 갱신"
    )


# ════════════════════════════════════════════════════════
# PAGE 2
# ════════════════════════════════════════════════════════
def page2():
    if st.button("← 대시보드로 돌아가기"):
        st.session_state.page = 1
        st.rerun()

    st.markdown("## 멀티모달 인사이트 도출")
    st.caption("텍스트 군집 + 이미지 서브군집 → LLM 해석 → 인사이트 확정")
    st.divider()

    st.markdown("### LLM 추천 — 볼륨 변화 기준 주목 군집")
    st.info("""
**볼륨 변화율 분석 결과, 아래 이미지 서브군집을 우선 검토하세요.**

- 🔴 **T1_I2 (호환 배터리 구매형)** — T1 군집 내 184건으로 압도적. 서드파티 패턴 집중.
- 🔴 **T3_I1 (비공식 수리업체 이용형)** — T3 군집 내 84건. 비공식 채널 이탈 신호.
    """)

    st.divider()

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
        for sub_id in ['T1_I0','T1_I1','T1_I2']:
            _, name, count = all_subclusters[sub_id]
            if st.checkbox(
                f"{sub_id} — {name} ({count}건)",
                value=(sub_id=='T1_I2'),
                key=f"c_{sub_id}"
            ):
                selected_subs.append(sub_id)

    with t3_col:
        st.markdown("**T3: 고장/AS 서비스센터 경험**")
        for sub_id in ['T3_I0','T3_I1','T3_I2']:
            _, name, count = all_subclusters[sub_id]
            if st.checkbox(
                f"{sub_id} — {name} ({count}건)",
                value=(sub_id=='T3_I1'),
                key=f"c_{sub_id}"
            ):
                selected_subs.append(sub_id)

    st.divider()

    if selected_subs:
        st.markdown("### 선택한 서브군집 대표 이미지")
        for sub_id in selected_subs:
            _, name, count = all_subclusters[sub_id]
            st.markdown(f"**{sub_id} — {name}** ({count}건)")
            cols = st.columns(5)
            for j in range(5):
                with cols[j]:
                    st.image(img_url(sub_id, j), use_column_width=True)
            st.markdown("---")

    st.divider()

    st.markdown("### LLM 멀티모달 해석")

    groq_key = st.text_input(
        "Groq API Key (선택)",
        type="password",
        placeholder="gsk_... 입력 시 실제 LLM 호출 / 비워두면 예시 결과 표시"
    )

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    HARDCODED = [
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
배터리 수명 종료 시점(구매 후 약 11~14개월)에 ThinQ 앱 푸시로 정품 배터리 구독 오퍼를 선제 발송합니다.

**2. 비공식 업체 검색 차단**
네이버 블로그 검색에서 비공식 업체 콘텐츠가 상위 노출되고 있습니다. LG 공식 AS 콘텐츠 SEO 강화 + 블로그 광고 입찰가 상향으로 공식 채널 노출을 선점해야 합니다."""
        },
        {
            "q": "이미지 분석이 없었으면 몰랐을 것은?",
            "a": """텍스트만 분석했다면 아래 세 가지를 놓쳤을 것입니다.

1. **이탈 브랜드 특정 불가** — GIGACELL이라는 브랜드는 이미지에서만 확인됩니다.

2. **비공식 업체 채널명 파악 불가** — 에어킹 등 업체명은 이미지 속 명함/배너에서만 나옵니다.

3. **이탈 규모 실감 불가** — T1_I2 184건이라는 볼륨으로 규모가 시각화됩니다."""
        }
    ]

    if not st.session_state.chat_history:
        st.session_state.chat_history = HARDCODED.copy()

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['q'])
        with st.chat_message("assistant"):
            st.markdown(chat['a'])

    user_input = st.chat_input("추가로 물어보고 싶은 것을 입력하세요")
    if user_input:
        if groq_key:
            messages = [{
                "role": "system",
                "content": "너는 LG전자 HS본부 멀티모달 VOC 분석 보조자다. 한국어로 답해라."
            }]
            for chat in st.session_state.chat_history:
                messages.append({"role":"user","content":chat['q']})
                messages.append({"role":"assistant","content":chat['a']})
            messages.append({"role":"user","content":user_input})
            with st.spinner("LLM 답변 생성 중..."):
                answer = call_groq(groq_key, messages)
        else:
            answer = "Groq API Key를 입력하면 실제 LLM이 답변합니다."
        st.session_state.chat_history.append(
            {'q': user_input, 'a': answer}
        )
        st.rerun()

    st.divider()

    st.markdown("### 최종 인사이트")
    st.success("""
**LG 청소기 사후관리 생태계를 서드파티가 잠식하고 있다**

텍스트 VOC만 보면 배터리 불만과 AS 불만이 각각 개별 문제로 보입니다.

이미지 분석 결과, T1_I2에서 GIGACELL 호환 배터리, T3_I1에서 에어킹 비공식 수리업체가 반복 등장합니다.

두 군집을 합쳐서 보면 **배터리 · AS · 분해청소 3개 영역 전부 서드파티가 선점**하고 있음이 확인됩니다.

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
        submitted = st.form_submit_button(
            "인사이트 확정 & 저장", type="primary"
        )
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
