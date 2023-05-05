import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from joblib import load, dump
matplotlib.rcParams['font.family'] = ['Heiti TC'] # Set font for Chinese

def sideBar():
    # Title
    st.header("随机森林分类App")
    st.divider()

    # Choose model
    model_selection = st.selectbox(
        "选择模型", ("重新训练模型", "使用已有模型"), label_visibility="visible")

    # File upload widgets
    with st.expander(model_selection):
        if model_selection == "重新训练模型":
            train_file = st.file_uploader("上传训练数据", type=".csv", accept_multiple_files=False)

            # If uploaded file, read the file
            # TODO: add csv file format check later
            if train_file is not None:
                train_read = pd.read_csv(
                train_file, header=0, encoding = "utf8", index_col=False)
                st.dataframe(train_read, height=200, width=350)
            
            # Make basic scatter plot
            # TODO: implement prettier plot with plotly, or other libraries with native support for Chinese
                with st.container():
                    fig, ax = plt.subplots(figsize = (6,6))
                    sns.scatterplot(data = train_read, x = "x1", y="x2", hue = "y", ax=ax)
                    plt.legend(title= "图例", fontsize=12)
                    plt.xlabel("δ29", fontsize=12)
                    plt.ylabel("δ30", fontsize=12)
                    plt.title("数据分布散点图", fontsize=12)

                    st.pyplot(fig=fig)

        if model_selection == "使用已有模型":
            saved_model = st.file_uploader(
                "上传已有模型", type=".joblib", accept_multiple_files=False)
        
            if saved_model is not None:
                clf = load(saved_model)
                st.success("加载完成",icon="✅")

    # Predict File Upload
    with st.expander("上传预测数据"):
        predict_file = st.file_uploader("上传预测数据", type=".csv", accept_multiple_files=False, label_visibility="collapsed")

        if predict_file is not None:
            predict_read = pd.read_csv(
            predict_file, header=0, encoding="utf8", index_col=False)
            st.dataframe(predict_read, height=200, width=350)

            with st.container():
                    fig, ax = plt.subplots(figsize = (6,6))
                    sns.scatterplot(data = predict_read, x = "x1", y="x2", ax=ax)
                    plt.xlabel("δ29", fontsize=12)
                    plt.ylabel("δ30", fontsize=12)
                    plt.title("数据分布散点图", fontsize=12)

                    st.pyplot(fig=fig)
    
    if ((model_selection == "重新训练模型") and (train_file is not None) and (predict_file is not None)):
        return model_selection, train_read, predict_read
    elif ((model_selection == "使用已有模型") and (saved_model is not None) and (predict_file is not None)):
        return model_selection, clf, predict_read
    else: 
        return None, None, None
    

    