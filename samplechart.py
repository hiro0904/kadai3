import streamlit as st
import plotly.graph_objs as go

# Streamlitアプリケーションのタイトルを設定
st.title("0から100の5つの要素を持つレーダーチャート")

# レーダーチャートのデータを準備
categories = ["要素1", "要素2", "要素3", "要素4", "要素5"]
values = [st.slider(f"{category}の値", 0, 100, 50) for category in categories]

# レーダーチャートを作成
fig = go.Figure()

fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself", name="要素"))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False
)

# レーダーチャートを表示
st.plotly_chart(fig)
