import streamlit as st
import pandas as pd
import numpy as np

def colPred(model = None, pred = None):
    st.header("数据判断")
    st.divider()

    if "result" not in st.session_state:
        st.session_state.result = None
    
    if st.button("开始分类"):
        x_data = pred.iloc[:,0:2].to_numpy()

        y_pred = model.predict(x_data)
        y_score = model.predict_proba(x_data)
        st.session_state.result = pd.DataFrame(np.concatenate((y_pred.reshape(-1,1),y_score,x_data),axis=1),columns = ["预测","混合源","矿砂源","稻壳源","δ29","δ30"])

        st.dataframe(st.session_state.result, height=350, width=350)

    if st.session_state.result is not None:
        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')
        
        st.download_button(label="保存结果",data=convert_df(st.session_state.result),file_name="output.csv",mime="text/csv")
    
    if st.session_state.result is not None:
        return st.session_state.result
    else:
        return None


