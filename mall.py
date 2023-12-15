import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
matplotlib.use('Agg')


st.header("Welcome to this website")
st.subheader("Choice an option")

hide_streamlit_style = """ 
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

mall = pd.read_csv("Mall_Customers.csv")
mall_con_d = mall.drop(["CustomerID"], axis = 1)
mall_con = pd.get_dummies(mall_con_d)
sc = StandardScaler()
mall_sc = sc.fit_transform(mall_con)

pca = PCA(n_components=2)
mall_transform = pca.fit_transform(mall_sc)

#To find the values of k using Elbow Method
wcss = []
for i in range(1,11):
    model = KMeans(n_clusters= i, init = "k-means++", random_state = 42)
    model.fit(mall_transform)
    wcss.append(model.inertia_)

model = KMeans(n_clusters=4, init = "k-means++", random_state=42)
y_means = model.fit_predict(mall_transform)

# Taking in data
options = ['Prediction', 'Classification']
choice = st.selectbox("Select Options", options)
if choice == 'Prediction':
    st.subheader("Enter the CustomerID, Genre(female/male), Age, Income, and Spending score")
    CustomerID = st.number_input("Enter the CustomerID")
    Genre = st.text_input("Enter the sex(Female or male)")
    Age = st.number_input("Enter Age")
    Income = st.number_input("Enter Individual's income")
    Spending = st.number_input("Enter Individual's spending score")


    All = pd.DataFrame({
        'CustomerID': [CustomerID],
        'Genre': [Genre.lower()],
        'Age': [Age],
        'Annual Income (k$)': [Income],
        'Spending Score (1-100)': [Spending]
    })
    All_drop = All.drop(["CustomerID"], axis = 1)
    All_con = pd.get_dummies(All_drop)
    All_con = All_con.reindex(columns=mall_con.columns, fill_value=0)
    All_data = sc.transform(All_con)
    transformed_data = pca.transform(All_data)
    y_predict = model.predict(transformed_data)[0]

    if y_predict == 3:
        if Genre.lower() == "female":
            st.write("There is a very low chance this customer will visit again; She does not earn much")
        else:
            st.write("There is a very low chance this customer will visit again; He does not earn much")
    elif y_predict == 2:
        if Genre.lower() == "female":
            st.write("She is a young person and has high purchasing power, She will most likely come back so send her more adverts and offer promo")
        else:
            st.write("He is a young person and has high purchasing power, he will most likely come back so send him more adverts and offer promo")
    elif y_predict == 1:
        if Genre.lower() == "female":
            st.write("There is a very low chance this customer will visit again; you do not need to invest much in following up on her")
        else:
            st.write("There is a very low chance this customer will visit again; you do not need to invest much in following up on him") 
    elif y_predict == 0:
        if Genre.lower() == "female":
            st.write("Follow up on this customer, she is most likely going to come back again")
        else:
            st.write("Follow up on this customer, he is most likely going to come back again")  
    else:
        st.write("There is probability the customer will visit again. I cannot say much")
    
        st.write("Predicted cluster labels:", y_predict)

else:
    a = st.file_uploader("Enter a dataset containing the CustomerID, Genre, age, income, and spending score", type=["csv", "xlsx"])
    
    if a is not None:
        st.success("Data uploaded successfully")
        input_a = pd.read_csv(a)

        def func(a):
            # Extract Genre from the input dataset
            a_drop = a.drop(["CustomerID"], axis = 1)
              # Convert 'Genre' to lowercase to handle case-insensitivity
            a_drop['Genre'] = a_drop['Genre'].str.lower()
            input = pd.get_dummies(a_drop, columns=["Genre"])
            input_all_values = input.reindex(columns=mall_con.columns, fill_value=0)
            # Use the existing scaler fitted on the training data
            input_sc = sc.transform(input_all_values)
            # Use the existing PCA model fitted on the training data
            input_transform = pca.transform(input_sc)
            # Use 'predict' instead of 'fit_predict' for new data
            y_predict = model.fit_predict(input_transform)

            from scipy.cluster.hierarchy import fcluster, linkage
            linkage_matrix = linkage(a[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], method='average', metric='euclidean')
            
        
            cluster_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
            
            for i, Genre in zip(cluster_labels, a['Genre'].str.lower()):
                if i == 0:
                    if Genre == "female":
                        st.write("Follow up on this customer, she is most likely going to come back again", i)
                    else:
                        st.write("Follow up on this customer, he is most likely going to come back again", i)
                elif i == 1:
                    if Genre == "female":
                        st.write("There is a very low chance this customer will visit again; you do not need to invest much in following up on her", i)
                    else:
                        st.write("There is a very low chance this customer will visit again; you do not need to invest much in following up on him", i)
                elif i == 2:
                    if Genre == "female":
                        st.write("She is a young person and has high purchasing power, She will most likely come back so send her more adverts and offer promo", i)
                    else:
                        st.write("He is a young person and has high purchasing power, he will most likely come back so her more adverts and offer promo", i)
                elif i == 3:
                    if Genre == "female":
                        st.write("There is a very low chance this customer will visit again; She does not earn much", i)
                    else:
                        st.write("There is a very low chance this customer will visit again; He does not earn much", i)
                else:
                    st.write("There is probability the customer will visit again. I cannot say much", i)

        func(input_a)

        st.write("Thank you for using this app. We hope it was very useful!")
