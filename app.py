
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="LG CodeZero CEJ 대시보드", layout="wide")

@st.cache_data
def load_data():
    chunk_df = pd.read_csv('chunk_umap.csv', encoding='utf-8-sig')
    image_df = pd.read_csv('image_umap.csv', encoding='utf-8-sig')
    return chunk_df, image_df

chunk_df, image_df = load_data()

CLUSTER_COLORS = {
    0:'#3B8BD4', 1:'#E24B4A', 2:'#1D9E75',
    3:'#EF9F27', 4:'#D4537E', 5:'#7F77DD',
    6:'#5DCAA5', 7:'#F0997B', 8:'#888780', -1:'#cccccc'
}

label_map = dict(zip(
    chunk_df.dropna(subset=['text_cluster_label'])['text_cluster'].astype(int),
    chunk_df.dropna(subset=['text_cluster_label'])['text_cluster_label']
))

st.markdown("### LG CodeZero 블로그 CEJ 시그널 대시보드")
st.caption("청소기 블로그 멀티모달 분석")
st.divider()

k1, k2, k3, k4 = st.columns(4)
k1.metric("텍스트 군집", f"{chunk_df['text_cluster'].nunique()}개")
k2.metric("이미지 서브군집", f"{image_df['image_subcluster_global'].nunique()}개")
k3.metric("분석 이미지", f"{len(image_df):,}개")
k4.metric("서드파티 이탈 군집", "3개 ⚠️", delta="즉시 대응", delta_color="inverse")

st.divider()

st.sidebar.title("군집 선택")
options = ['전체 보기'] + [f"T{k}: {v}" for k,v in sorted(label_map.items()) if k != -1]
selected = st.sidebar.radio("텍스트 군집", options)
selected_cluster = None if selected == '전체 보기' else int(selected.split(':')[0].replace('T',''))

col1, col2 = st.columns(2)

with col1:
    st.markdown("**텍스트 군집 UMAP**")
    st.caption("텍스트만 보면 하나의 군집")
    plot_df = chunk_df.copy()
    plot_df['label'] = plot_df['text_cluster'].map(label_map).fillna('Noise')
    plot_df['opacity'] = plot_df['text_cluster'].apply(
        lambda x: 0.9 if selected_cluster is None or x == selected_cluster else 0.2
    )
    fig1 = px.scatter(
        plot_df, x='umap_x', y='umap_y', color='label',
        opacity=0.8, height=380,
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
    st.caption("이미지로 보면 이렇게 갈립니다" if selected_cluster is None else f"T{selected_cluster} → 이미지로 분화")
    img_plot = image_df if selected_cluster is None else image_df[image_df['text_cluster'] == selected_cluster]
    fig2 = px.scatter(
        img_plot, x='img_umap_x', y='img_umap_y',
        color='image_subcluster_global',
        opacity=0.8, height=380,
        hover_data={'img_umap_x':False,'img_umap_y':False,'image_subcluster_global':True}
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

b1, b2, b3 = st.columns(3)

with b1:
    st.markdown("**군집별 볼륨**")
    vol = chunk_df[chunk_df['text_cluster'] != -1].groupby(
        ['text_cluster','text_cluster_label']
    ).size().reset_index(name='count').sort_values('count')
    fig3 = px.bar(vol, x='count', y='text_cluster_label', orientation='h', height=280)
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
        yaxis=dict(tickfont=dict(size=9))
    )
    st.plotly_chart(fig3, use_container_width=True)

with b2:
    st.markdown("**CEJ 단계 분포**")
    cej = chunk_df[chunk_df['text_cluster'] != -1].groupby('cej_stage').size().reset_index(name='count')
    fig4 = px.pie(cej, values='count', names='cej_stage', hole=0.55, height=280)
    fig4.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=10,b=0), legend=dict(font=dict(size=9))
    )
    st.plotly_chart(fig4, use_container_width=True)

with b3:
    st.markdown("**핵심 인사이트**")
    st.error("🔴 배터리 교체 군집 — 서드파티 이탈 집중")
    st.error("🔴 고장/AS 군집 — 비공식 업체 채널 잠식")
    st.warning("🟡 경쟁사 비교 군집 — 구매 단계 개입 필요")
    st.success("🟢 사용 만족 군집 — 긍정 레퍼런스 활용 가능")
