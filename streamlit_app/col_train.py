import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
from joblib import dump
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime
matplotlib.rcParams['font.family'] = ['Heiti TC'] # Set font for Chinese


def colTrain(train=None):
    st.header("模型训练")
    st.divider()
    
    if "clf" not in st.session_state:
        st.session_state.clf = None
    
    if "current" not in st.session_state:
        st.session_state.current = datetime.today()

    X = train.iloc[:, 1:3].to_numpy()
    y = train.iloc[:, 0].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=15)

    train_container = st.container()
    evaluate_container = st.container()

    if train_container.button(label="开始训练", help="使用新数据训练模型"):
        st.session_state.clf = RandomForestClassifier(n_estimators=100, random_state=0)
        st.session_state.clf.fit(X_train, y_train)
        train_container.success("训练完成",icon="✅")
    else:
        train_container.write("点击开始训练")

    if st.session_state.clf is not None:
        filename = "model/model_"+st.session_state.current.strftime("%m-%d-%Y-%H%M%S")+".joblib"

        if train_container.button(label="保存模型"):
            dump(st.session_state.clf,filename)
            train_container.success(f"{filename} 保存完成",icon="✅")

    if st.session_state.clf is not None:
        if evaluate_container.button(label="AUC-ROC分析", help="分析模型表现"):
            y_score = st.session_state.clf.predict_proba(X_test)

            pair_list = list(combinations(np.unique(y_test), 2))

            pair_scores = []
            mean_tpr = dict()

            label_binarizer = LabelBinarizer().fit(y_train)

            # Iterate through all pairs
            # label a: first in the pair, label b: second in the pair
            for ix, (label_a, label_b) in enumerate(pair_list):

                a_mask = y_test == label_a  # All y_tests that are 1 with label a
                b_mask = y_test == label_b  # All y_tests that are 1 with label b
                # All y_tests that are 1 for either a or b; remove the cases when c is true
                ab_mask = np.logical_or(a_mask, b_mask)

                # Remove entries when both a and b are 0
                a_true = a_mask[ab_mask]
                b_true = b_mask[ab_mask]

                idx_a = np.flatnonzero(label_binarizer.classes_ == label_a)[
                    0]  # Index of label a class
                idx_b = np.flatnonzero(label_binarizer.classes_ == label_b)[
                    0]  # Index of label b class

                fpr_a, tpr_a, _ = roc_curve(
                    a_true, y_score[ab_mask, idx_a])  # Get fpr and tpr for a
                fpr_b, tpr_b, _ = roc_curve(
                    b_true, y_score[ab_mask, idx_b])  # Get fpr and tpr for b

                fpr_grid = np.linspace(0.0, 1.0, 1000)

                # Macro averaging for 2 classes; ix: index for each pair
                mean_tpr[ix] = np.zeros_like(fpr_grid)
                mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
                mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
                mean_tpr[ix] /= 2
                mean_score = auc(fpr_grid, mean_tpr[ix])
                pair_scores.append(mean_score)  # Append score for this pair

            macro_roc_auc_ovo = roc_auc_score(
                y_test,
                y_score,
                multi_class="ovo",
                average="macro")

            # To compute average tpr for all pairs
            ovo_tpr = np.zeros_like(fpr_grid)

            fig, ax = plt.subplots(figsize=(6, 6))
            for ix, (label_a, label_b) in enumerate(pair_list):
                ovo_tpr += mean_tpr[ix]  # Add to ovo_tpr
                plt.plot(
                    fpr_grid,
                    mean_tpr[ix],
                    label=f"平均 {label_a} vs {label_b} (AUC = {pair_scores[ix]:.2f})",
                )

            ovo_tpr /= sum(1 for pair in enumerate(pair_list))

            plt.plot(
                fpr_grid,
                ovo_tpr,
                label=f"一对一宏观平均 (AUC = {macro_roc_auc_ovo:.2f})",
                linestyle=":",
                linewidth=4,
            )
            plt.plot([0, 1], [0, 1], "k--",
                    label="随机分类 (AUC = 0.5)")
            plt.axis("square")
            plt.xlabel("假阴性率")
            plt.ylabel("真阳性率")
            plt.title(
                "多类别一对一 (One vs. One)\nROC 分析")
            plt.legend()
                
            evaluate_container.pyplot(fig=fig)
        else:
            evaluate_container.write("点击进行分析")
    
    if st.session_state.clf is not None:
        return st.session_state.clf
    else:
        return None