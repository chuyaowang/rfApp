import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt

def colDisp(model = None, res = None):
    st.header("查看分布")
    st.divider()
    
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    plot_step = 0.002
    n_classes = 3

    X = res.iloc[:,4:6].to_numpy()

    cmap = plt.colormaps["viridis"]
    cmap = truncate_colormap(cmap,0.2,0.8)
    sampleColor = cmap(np.linspace(0,1,n_classes))
    cmap = colors.ListedColormap(sampleColor)

    fig, ax = plt.subplots(figsize=(6, 6))

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )


    # Plot contour
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Make ordinal encoder
    od = OrdinalEncoder().fit(Z.reshape(-1, 1))
    Z = od.transform(Z.reshape(-1,1))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=.85)

    for (cat, group), col in zip(res.groupby("预测"), sampleColor):
        ax.scatter(group["δ29"], group["δ30"], color=col, edgecolors = "black", marker="^", linestyle="", label=cat)

    plt.legend(title= "图例", fontsize=12)
    plt.xlabel("δ29", fontsize=12)
    plt.ylabel("δ30", fontsize=12)
    plt.title("模型分类边界", fontsize=12)
    plt.grid(False)
    
    st.pyplot(fig=fig)
