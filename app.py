import streamlit as st
import pandas as pd
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
import io

# タイトル
st.title("Amazon広告分析ダッシュボード")

# ファイルアップロード
uploaded_file = st.file_uploader("Amazon広告レポート（検索語句 or ターゲティング）をアップロード", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("データのプレビュー")
    st.dataframe(df.head())

    # 柔軟なカラム名対応
    rename_dict = {
        'インプレッション': 'Impressions',
        'クリック数': 'Clicks',
        'クリック': 'Clicks',
        '広告費': 'Spend',
        '費用': 'Spend',
        '広告がクリックされてから7日間の総売上高': 'Sales',
        '広告費売上高比率（ACOS）合計': 'ACOS',
        '広告費用対効果（ROAS）合計': 'ROAS',
        'カスタマーの検索キーワード': 'Search Term',
        'ターゲティング': 'Targeting'
    }
    df.rename(columns=rename_dict, inplace=True)

    # 必要なカラムがあるか確認
    required_cols = ['Impressions', 'Clicks', 'Spend', 'Sales']
    if all(col in df.columns for col in required_cols):

        # 指標の計算
        df['CTR'] = df['Clicks'] / df['Impressions']
        df['CPC'] = df['Spend'] / df['Clicks']

        st.subheader("パフォーマンス指標")
        st.write("上位キーワード・ターゲティング")

        top_terms = df.sort_values(by='Sales', ascending=False).head(10)
        expected_cols = ['Search Term', 'Targeting', 'Impressions', 'Clicks', 'Sales', 'ACOS', 'ROAS']
        available_cols = [col for col in expected_cols if col in top_terms.columns]
        st.dataframe(top_terms[available_cols])

        st.subheader("売上 vs ACOS グラフ")
        if 'Sales' in df.columns and 'ACOS' in df.columns:
            chart = alt.Chart(df).mark_circle(size=60).encode(
                x='Sales',
                y='ACOS',
                tooltip=[col for col in ['Search Term', 'Sales', 'ACOS', 'ROAS'] if col in df.columns]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Sales または ACOS カラムが見つからないため、グラフを表示できません。")

        # --- 広告出稿レコメンド ---
        st.subheader("✨ 広告出稿レコメンド（オーガニックで反応あり）")
        recommend_df = df[(df['Sales'] > 0) & (df['Spend'] == 0) & (df['Clicks'] > 0)]

        if not recommend_df.empty:
            recommend_df['RecommendationScore'] = (
                recommend_df['Sales'] * 0.6 +
                (recommend_df['Clicks'] / recommend_df['Impressions']) * 100 * 0.4
            )

            top_recommend = recommend_df.sort_values(by='RecommendationScore', ascending=False)

            st.write("出稿されていないが売上につながっている検索語句の上位候補：")
            st.dataframe(top_recommend[['Search Term', 'Sales', 'Clicks', 'Impressions', 'RecommendationScore']].head(10))

            # CSVエクスポートボタン
            csv_recommend = top_recommend[['Search Term', 'Sales', 'Clicks', 'Impressions', 'RecommendationScore']].to_csv(index=False).encode('utf-8')
            st.download_button("📥 レコメンド結果をCSVでダウンロード", data=csv_recommend, file_name="recommend_keywords.csv", mime='text/csv')
        else:
            st.info("出稿されていないが売上がある検索語句は見つかりませんでした。")

        # --- 自動入札額のAI提案 ---
        st.subheader("💡 自動入札額のAI提案")
        target_acos = st.slider("目標ACOS（％）", 10, 100, 30) / 100

        df = df[df['Clicks'] > 0]  # ゼロ除算防止
        df['CVR'] = df['Sales'] / df['Clicks']
        df['AvgOrderValue'] = df['Sales'] / df['Clicks']
        df['RecommendedBid'] = df['CVR'] * df['AvgOrderValue'] * target_acos

        expected_bid_cols = ['Search Term', 'Sales', 'Clicks', 'CVR', 'AvgOrderValue', 'RecommendedBid']
        available_bid_cols = [col for col in expected_bid_cols if col in df.columns]
        top_bid = df[available_bid_cols]
        st.dataframe(top_bid.sort_values(by='RecommendedBid', ascending=False).head(10))

        # CSVエクスポートボタン（入札提案）
        csv_bid = top_bid.sort_values(by='RecommendedBid', ascending=False).to_csv(index=False).encode('utf-8')
        st.download_button("📥 推奨入札額をCSVでダウンロード", data=csv_bid, file_name="recommended_bids.csv", mime='text/csv')

        # --- AIによる広告診断レポート ---
        st.subheader("🧠 AI診断レポート：広告の問題点と改善提案")
        st.write("AIがあなたの広告データを分析し、改善点とレコメンドを提示します。")

        if st.button("📋 レポートを生成"):
            diagnostic_msgs = []
            for _, row in df.iterrows():
                if row['Spend'] > 1000 and row['Sales'] == 0:
                    diagnostic_msgs.append(f"❌ '{row.get('Search Term', '（キーワード名不明）')}' は広告費が¥{int(row['Spend'])}かかっていますが売上がありません。除外を検討してください。")
                if row['CTR'] < 0.01:
                    diagnostic_msgs.append(f"⚠️ '{row.get('Search Term', '（キーワード名不明）')}' のクリック率は {row['CTR']:.2%} と非常に低く、関連性が薄い可能性があります。")
                if row['CVR'] > 0.1 and row['RecommendedBid'] < 20:
                    diagnostic_msgs.append(f"💡 '{row.get('Search Term', '（キーワード名不明）')}' はCVRが高い（{row['CVR']:.1%}）ですが、入札額が低めです。¥{int(row['RecommendedBid'])} 以上に設定することで売上向上が見込めます。")

            if diagnostic_msgs:
                for msg in diagnostic_msgs:
                    st.markdown(msg)
            else:
                st.info("明確な改善提案が見つかりませんでした。全体的に健全なデータです！")

        # --- 高度AI機能：検索語句のクラスタリング ---
        st.subheader("🧠 検索語句のAIクラスタリング")
        if 'Search Term' in df.columns:
            text_data = df['Search Term'].fillna('')
            vectorizer = TfidfVectorizer(max_features=100)
            X = vectorizer.fit_transform(text_data)

            kmeans = KMeans(n_clusters=4, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)
            st.write("クラスタごとの代表的な検索語句：")
            for i in range(4):
                st.markdown(f"**クラスタ {i}**")
                st.write(df[df['Cluster'] == i]['Search Term'].head(5).tolist())

        # --- 高度AI機能：機械学習による売れる予測 ---
        st.subheader("🔍 機械学習による売れる検索語句の予測")
        df['Target'] = (df['CVR'] > 0.05).astype(int)
        ml_features = ['Impressions', 'Clicks', 'Spend', 'CTR', 'ROAS']
        df_ml = df.dropna(subset=ml_features + ['Target'])

        model = LGBMClassifier()
        model.fit(df_ml[ml_features], df_ml['Target'])

        df_ml['AI_Predicted_Success'] = model.predict_proba(df_ml[ml_features])[:, 1]

        expected_pred_cols = ['Search Term', 'Sales', 'Clicks', 'CVR', 'AI_Predicted_Success']
        available_pred_cols = [col for col in expected_pred_cols if col in df_ml.columns]
        st.dataframe(
            df_ml[available_pred_cols].sort_values(
                by='AI_Predicted_Success' if 'AI_Predicted_Success' in df_ml.columns else available_pred_cols[0],
                ascending=False
            ).head(10)
        )

    else:
        st.error("このファイル形式は対応していないようです。検索語句レポートまたはターゲティングレポートをアップロードしてください。")
else:
    st.info("まずはExcelファイルをアップロードしてください。")
