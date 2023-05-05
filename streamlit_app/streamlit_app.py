import streamlit as st
from sidebar import sideBar
from col_train import colTrain
from col_pred import colPred
from col_disp import colDisp
st.set_page_config(
    layout="wide",
    page_title="Random Forest",
    page_icon="resources/icon.png")

def main():
    with st.sidebar:
        model_selection, train, predict = sideBar()
        # model_selection: the model selected, used or new
        # train: training data or old model
        # predict: data to perform prediction on
    if model_selection == "重新训练模型":
        col_train, col_pred, col_disp = st.columns(3)

        # Trains the model with new data or evaluate existing model
        # TODO: change model selection in the end
        if train is not None:
            with col_train:
                model = colTrain(train = train)

        # Available once training is done
        if (model is not None) and (predict is not None):
            with col_pred:
                result = colPred(model = model, pred = predict)
            
            if result is not None:
                with col_disp:
                    colDisp(model = model, res = result, train=train)
    
    # Prediction can be done right away if using previous model
    if (model_selection == "使用已有模型") and (train is not None) and (predict is not None):
        col_pred, col_disp = st.columns(2)
        
        with col_pred:
            result = colPred(model = train, pred = predict)
        
        if result is not None:
            with col_disp:
                colDisp(model = train, res = result)

if __name__ == '__main__':
	main()
