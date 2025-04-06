import streamlit as st
import pandas as pd
import altair as alt

# CSV or Excel ファイルアップロード
df = pd.read_csv("your_data.csv")  # Excelの場合は pd.read_excel に変更

st.set_page_config(page_title="広告分析ダッシュボード", layout="wide")
st.title("📊 スポンサープロダクト広告分析ツール")

# サイドバーでメニュー選択
menu = st.sidebar.radio("表示する分析を選択", (
    "キーワード別の成果一覧",
    "キャンペーン全体の概要",
    "日別推移の可視化",
    "レコメンド機能"
))

# 日付型に変換（必要な場合）
df['Date'] = pd.to_datetime(df['Date'])

if menu == "キーワード別の成果一覧":
    st.header("🔍 キーワード別 成果一覧")
    keyword_summary = df.groupby("Keyword").agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Cost': 'sum',
        'Sales': 'sum',
        'ACOS': 'mean',
        'ROAS': 'mean'
    }).reset_index()

    keyword_summary = keyword_summary.sort_values("Sales", ascending=False)
    st.dataframe(keyword_summary, use_container_width=True)

elif menu == "キャンペーン全体の概要":
    st.header("📈 キャンペーン全体の概要")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("インプレッション", f"{int(df['Impressions'].sum()):,}")
    col2.metric("クリック数", f"{int(df['Clicks'].sum()):,}")
    col3.metric("広告費", f"¥{df['Cost'].sum():,.0f}")
    col4.metric("売上", f"¥{df['Sales'].sum():,.0f}")

    chart = alt.Chart(df).mark_bar().encode(
        x='Keyword:N',
        y='Sales:Q',
        tooltip=['Keyword', 'Sales']
    ).properties(width=800, height=400).interactive()

    st.altair_chart(chart)

elif menu == "日別推移の可視化":
    st.header("📆 日別推移")
    daily = df.groupby("Date").agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Cost': 'sum',
        'Sales': 'sum',
    }).reset_index()

    st.line_chart(daily.set_index("Date"))

elif menu == "レコメンド機能":
    st.header("💡 改善アクション提案")

    # 条件に応じたフィルタとコメント
    bad_roas = df[(df['Cost'] > 0) & (df['ROAS'] < 1)]
    no_sales = df[(df['Clicks'] > 5) & (df['Sales'] == 0)]

    st.subheader("ROAS が低いキーワード")
    if not bad_roas.empty:
        st.dataframe(bad_roas[['Keyword', 'Cost', 'Sales', 'ROAS']], use_container_width=True)
        st.info("ROAS が 1 未満のキーワードは費用に対して効果が出ていません。停止や調整を検討しましょう。")
    else:
        st.success("ROAS が 低いキーワードはありません。")

    st.subheader("クリック多いが売れていないキーワード")
    if not no_sales.empty:
        st.dataframe(no_sales[['Keyword', 'Clicks', 'Sales']], use_container_width=True)
        st.info("クリックは多いが売れていないキーワードがあります。訴求内容の見直しやキーワード除外を検討してください。")
    else:
        st.success("クリックはある程度あり、売上も発生しています。")
