import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import  plot_roc_curve



def main():

    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushroom edible or poisonous ")
    st.sidebar.markdown("Are your mushroom edible or poisonous")
    

    @st.cache(persist=True)
    def load_data():
        data=pd.read_csv("C:\\Users\\HARSHITA GUPTA\\Documents\\mushrooms.csv") 
        label=LabelEncoder()
        for col in data.columns:
            data[col]=label.fit_transform(data[col])
        return data   
    
    @st.cache(persist=True) 
    def split_d(df):
        y=df.type
        x=df.drop(columns=['type'])
        x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.3, random_state=0)
        return x_train,x_test,y_train,y_test   

    

    def plot_metrics(metrics_list):
        if 'Confusion matrix' in metrics_list:
            st.subheader("Confusion matrix")
            plot_confusion_matrix(model,x_test,y_test,display_labels=class_names) 
            st.pyplot()  

    
    
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model,x_test,y_test) 
            st.pyplot()            

    
        if 'Precision Recall curve' in metrics_list:
            st.subheader("Precision Recall curve")
            plot_precision_recall_curve(model,x_test,y_test) 
            st.pyplot()  
         



    df=load_data()
    x_train,x_test,y_train,y_test=split_d(df)
    class_names=['edible','poisonous']
    st.sidebar.subheader('Chosse Classifier')
    classifier=st.sidebar.selectbox("Classifier",("Support Vector Machine(SVM)","LogisticRegression","RandomForest"))
    



    if classifier=='Support Vector Machine(SVM)':
        st.sidebar.subheader('Model Hyperparameters')
        c=st.sidebar.number_input('c (Regularization parameter)', 0.01,10.0,step=0.01,key='c')
        kernel=st.sidebar.radio('Kernal',['rbf','linear'],key='kernel')
        gamma=st.sidebar.radio('Gamma (Kernel Coefficient)',['scale','auto'],key='gamma')
        metrics=st.sidebar.multiselect('What metrices to plot?',('Confusion matrix','ROC Curve','Precision Recall curve'))
        
        if st.sidebar.button('Classify',key='classify'):
            st.subheader('Support Vector Machine(SVM) Results')
            model=SVC(C=c,kernel=kernel,gamma=gamma)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write('Accuracy:',accuracy.round(2))
            st.write('Precision:',precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write('Recall:',recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)


    
    if classifier=='LogisticRegression':
        st.sidebar.subheader('Model Hyperparameters')
        c=st.sidebar.number_input('c (Regularization parameter)', 0.01,10.0,step=0.01,key='c_LR')
        max_iter=st.sidebar.slider('Maximum no of iteration',100,500,key='max_iter')
        metrics=st.sidebar.multiselect('What metrices to plot?',('Confusion matrix','ROC Curve','Precision Recall curve'))
        
        if st.sidebar.button('Classify',key='classify'):
            st.subheader('LogisticRegression')
            model=LogisticRegression(C=c,max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write('Accuracy:',accuracy.round(2))
            st.write('Precision:',precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write('Recall:',recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier=='RandomForest':
        st.sidebar.subheader('Model Hyperparameters')
        n_estimators=st.sidebar.number_input('The number of trees in the forest' ,100,5000 ,key='n_estimators')
        max_depth=st.sidebar.number_input('The maximum depth of a trees',1,20,key='max_depth')
        bootstrap=st.sidebar.radio('Bootstrap samples when buiding trees',('True','Flase'),key='bootstrap')
        metrics=st.sidebar.multiselect('What metrices to plot?',('Confusion matrix','ROC Curve','Precision Recall curve'))
        
        if st.sidebar.button('Classify',key='classify'):
            st.subheader('RandomForest')
            model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write('Accuracy:',accuracy.round(2))
            st.write('Precision:',precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write('Recall:',recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show your data",False):
        st.subheader("Mushroom Data Set(classification)")
        st.write(df)


if __name__== '__main__':
    main()



