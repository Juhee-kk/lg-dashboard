import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

st.set_page_config(
    page_title="LG 청소기 대시보드",
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

# ── LLM 해석 함수 ────────────────────────────────────────
def call_groq(api_key, cluster_id, cluster_label, cej_stage,
              rep_texts, subclusters_info):

    sub_info_str = "\n".join([
        f"- {s['label']}: {s['count']}건"
        for s in subclusters_info
    ])

    prompt = f"""
너는 LG전자 HS본부의 멀티모달 VOC/CEJ 분석 보조자다.
아래 텍스트 군집과 이미지 서브군집 정보를 보고 인사이트를 분석해라.
반드시 텍스트만 봤을 때와 이미지까지 봤을 때 달라지는 해석을 구분해라.

[텍스트 군집]
- 군집 ID: T{cluster_id}
- 라벨: {cluster_label}
- CEJ 단계: {cej_stage}

[대표 텍스트 VOC]
{chr(10).join([f"{i+1}. {t}" for i, t in enumerate(rep_texts)])}

[이미지 서브군집 분화]
{sub_info_str}

[출력 형식 - 반드시 아래 형식으로]
1. 텍스트만 봤을 때 해석 (2줄 이내):
2. 이미지까지 봤을 때 달라지는 해석 (2줄 이내):
3. 멀티모달 Delta - 이미지 없었으면 몰랐을 것 (1줄):
4. LG 액션 제안 (1줄):
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {"role": "system", "content": "너는 근거 기반 VOC/CEJ 분석 보조자다. 과장하지 말고 텍스트 근거와 이미지 근거를 분리해라. 한국어로 답해라."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.2
        }
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"API 호출 실패: {response.status_code}"

# ── 헤더 ────────────────────────────────────────────────
st.markdown("## LG 청소기 대시보드")
st.caption("플랫폼: 블로그 · 멀티모달 CEJ 분석 · 텍스트 군집 9개 · 이미지 서브군집 26개")
st.divider()

# ── 시그널 카드 3개 ──────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.error(f"""
    📈 **볼륨 급증 군집**
    T{int(top_surge['text_cluster'])}: {top_surge['text_cluster_label']}
    이번 주 {int(top_surge['this_week'])}건 (지난 주 {int(top_surge['prev_week'])}건)
    **+{top_surge['change_rate']}% 급증**
    """)

top_new = vol_df[vol_df['text_cluster'] == 3].iloc[0]
with c2:
    st.warning(f"""
    🔍 **주목할 군집**
    T3: {top_new['text_cluster_label']}
    이번 주 {int(top_new['this_week'])}건
    텍스트 하나 → 이미지 3개로 분화
    """)

with c3:
    st.info(f"""
    📊 **전체 현황**
    분석 텍스트: {len(chunk_df[chunk_df['text_cluster'] != -1])}건
    분석 이미지: {len(image_df)}건
    총 군집: {vol_df.shape[0]}개
    """)

st.divider()

# ── 군집 선택 ────────────────────────────────────────────
options = ['전체 보기'] + [
    f"T{int(row['text_cluster'])}: {row['text_cluster_label']}"
    for _, row in vol_df.iterrows()
]
selected = st.selectbox("텍스트 군집 선택 (볼륨 변화율 순)", options)
selected_cluster = None if selected == '전체 보기' else int(
    selected.split(':')[0].replace('T','')
)

st.divider()

# ── UMAP 두 개 ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("**텍스트 군집 UMAP**")
    st.caption("텍스트만 보면 하나의 군집")
    plot_df = chunk_df.copy()
    plot_df['label'] = plot_df['text_cluster'].map(label_map).fillna('Noise')
    fig1 = px.scatter(
        plot_df, x='umap_x', y='umap_y',
        color='label', opacity=0.8, height=350,
        hover_data={'umap_x':False,'umap_y':False,'label':True}
    )
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        legend=dict(font=dict(size=9),title='')
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
        hover_data={'img_umap_x':False,'img_umap_y':False,
                    'image_subcluster_global':True}
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        legend=dict(font=dict(size=9),title='')
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── 하단 ─────────────────────────────────────────────────
b1, b2 = st.columns(2)

with b1:
    st.markdown("**볼륨 변화율 (이번 주 vs 지난 주)**")
    vol_sorted = vol_df.sort_values('change_rate', ascending=True)
    fig3 = px.bar(
        vol_sorted,
        x='change_rate', y='text_cluster_label',
        orientation='h', height=320,
        color='change_rate',
        color_continuous_scale=['#1D9E75','#EF9F27','#E24B4A'],
        labels={'change_rate':'변화율 (%)','text_cluster_label':''}
    )
    fig3.add_vline(x=0, line_color='gray', line_width=1)
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0),
        coloraxis_showscale=False,
        yaxis=dict(tickfont=dict(size=9))
    )
    st.plotly_chart(fig3, use_container_width=True)

with b2:
    st.markdown("**소비자가 실제로 한 말**")
    if selected_cluster is not None:
        sub_df = chunk_df[chunk_df['text_cluster'] == selected_cluster]
        samples = sub_df['chunk_text'].dropna().sample(
            min(3, len(sub_df)), random_state=42
        )
        for text in samples:
            st.markdown(f"""
            <div style='border-left:3px solid #E24B4A;
                 padding:10px 14px;margin-bottom:8px;
                 border-radius:4px;font-size:13px;
                 line-height:1.6;
                 background:var(--background-color)'>
            {str(text)[:200]}...
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("왼쪽에서 군집을 선택하면 대표 텍스트가 표시됩니다")

st.divider()

# ── 대표 이미지 (T1, T3) ─────────────────────────────────
if selected_cluster in [1, 3]:
    tc_label = label_map.get(selected_cluster, '')
    st.markdown(f"**T{selected_cluster}: {tc_label} — 이미지 서브군집별 대표 이미지**")
    st.caption("텍스트는 하나의 군집 → 이미지로 보면 다른 패턴으로 분화")

    subclusters = sorted(
        image_df[image_df['text_cluster'] == selected_cluster]
        ['image_subcluster_global'].unique()
    )
    sub_counts = image_df[
        image_df['text_cluster'] == selected_cluster
    ].groupby('image_subcluster_global').size()

    cols = st.columns(len(subclusters))
    for col, sub_label in zip(cols, subclusters):
        with col:
            count = sub_counts.get(sub_label, 0)
            st.markdown(f"**{sub_label}** ({count}건)")
            for j in range(3):
                st.image(img_url(sub_label, j), use_column_width=True)

    st.divider()

# ── LLM 해석 ─────────────────────────────────────────────
if selected_cluster is not None:
    st.markdown("### LLM 멀티모달 해석")
    st.caption("텍스트 VOC + 이미지 서브군집 정보를 LLM에게 해석시킵니다")

    with st.expander("Groq API 설정", expanded=False):
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_로 시작하는 API 키 입력"
        )

    if st.button("LLM 해석 생성", type="primary"):
        if not api_key:
            st.warning("Groq API Key를 입력해주세요")
        else:
            # 대표 텍스트 수집
            sub_df = chunk_df[chunk_df['text_cluster'] == selected_cluster]
            rep_texts = sub_df['chunk_text'].dropna().sample(
                min(5, len(sub_df)), random_state=42
            ).tolist()

            # 이미지 서브군집 정보
            sub_counts = image_df[
                image_df['text_cluster'] == selected_cluster
            ].groupby('image_subcluster_global').size()

            subclusters_info = [
                {'label': k, 'count': v}
                for k, v in sub_counts.items()
            ]

            # CEJ 단계
            cej = chunk_df[
                chunk_df['text_cluster'] == selected_cluster
            ]['cej_stage'].iloc[0] if 'cej_stage' in chunk_df.columns else '-'

            with st.spinner("LLM 해석 생성 중..."):
                result = call_groq(
                    api_key=api_key,
                    cluster_id=selected_cluster,
                    cluster_label=label_map.get(selected_cluster, ''),
                    cej_stage=cej,
                    rep_texts=rep_texts,
                    subclusters_info=subclusters_info
                )

            st.markdown("#### 해석 결과")
            st.markdown(result)

            # 의사결정 저장
            st.divider()
            st.markdown("#### 사람 최종 판단")
            with st.form("decision_form"):
                final_insight = st.text_area(
                    "최종 인사이트 제목",
                    placeholder="예: 배터리 이탈 군집에서 서드파티 브랜드 특정 가능"
                )
                action = st.text_area(
                    "LG 액션 제안",
                    placeholder="예: 정품 배터리 구독 서비스 선제 오퍼"
                )
                priority = st.selectbox(
                    "우선순위",
                    ["높음", "중간", "낮음"]
                )
                submitted = st.form_submit_button("판단 저장", type="primary")

                if submitted:
                    st.success("저장 완료!")
                    st.markdown(f"""
                    **군집**: T{selected_cluster} {label_map.get(selected_cluster,'')}
                    **인사이트**: {final_insight}
                    **액션**: {action}
                    **우선순위**: {priority}
                    """)

st.caption("* 이전 주 볼륨은 시뮬레이션 데이터 · 실제 운영 시 주간 파이프라인으로 자동 갱신")
