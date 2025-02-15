import pandas as pd 
import numpy as np 
import tensorflow as tf 
import pickle
import joblib
import streamlit as st 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select page",["About","Prediction","Analysis"])

    
#  About page
if(app_mode=='About'):
    st.title("Predicting Creditworthiness: A Data-Driven Approach to Credit Card Approval  💳")
    st.header("About")
    image_path="cardone.jpg"
    st.image(image_path)
    st.markdown('''
Credit Card Approval AI Model Features:

1. Predictive Modeling: The AI model uses machine learning algorithms to predict the approval status of credit card applications based on various factors such as credit score, income, age, and employment history.
2. Real-time Processing: The model processes applications in real-time, providing instant decisions and reducing the time and effort required to process applications.
3. Automated Decisioning: The model automates the decision-making process, reducing the need for manual review and minimizing the risk of human error.
4. Risk Assessment: The model assesses the creditworthiness of applicants, identifying high-risk individuals and preventing fraudulent activity.
5. Personalization: The model provides personalized recommendations and offers to applicants based on their credit profile and behavior.
6. Scalability: The model can handle large volumes of data and scale to meet the needs of growing banks and financial institutions.
7. Integration: The model can be integrated with existing banking systems and infrastructure, providing a seamless and efficient experience for applicants.

How Credit Cards Work:

1. Application: A customer applies for a credit card by providing personal and financial information, such as income, employment history, and credit score.
2. Approval: The credit card issuer reviews the application and approves or rejects it based on the customer's creditworthiness.
3. Card Issuance: If approved, the customer receives a credit card with a unique account number, expiration date, and security code.
4. Transaction: The customer uses the credit card to make purchases, either online or in-store, by providing the card information and authorizing the transaction.
5. Authorization: The merchant verifies the customer's identity and checks the availability of funds on the credit card account.
6. Settlement: The merchant settles the transaction with the credit card issuer, who then pays the merchant and adds the transaction to the customer's account.
7. Billing: The credit card issuer sends a bill to the customer, detailing the transactions, fees, and interest charges.
8. Payment: The customer makes a payment, either in full or in part, to the credit card issuer, who then updates the account balance and applies any interest charges.

Credit Card Types:

1. Cashback Credit Cards: Offer rewards in the form of cashback on purchases.
2. Rewards Credit Cards: Offer rewards in the form of points, miles, or other benefits.
3. Secured Credit Cards: Require a security deposit and are designed for customers with poor credit.
4. Unsecured Credit Cards: Do not require a security deposit and are designed for customers with good credit.
5. Balance Transfer Credit Cards: Allow customers to transfer existing balances to a new credit card with a lower interest rate.

Credit Card Benefits:

1. Convenience: Credit cards provide a convenient and secure way to make purchases.
2. Rewards: Credit cards offer rewards and benefits, such as cashback, points, and travel miles.
3. Building Credit: Credit cards can help customers build credit by providing a way to demonstrate responsible payment behavior.
4. Emergency Funding: Credit cards can provide emergency funding in case of unexpected expenses or financial difficulties.
5. Purchase Protection: Credit cards often offer purchase protection, such as return and refund policies, and protection against identity theft.
                ''')


#prediction page
elif(app_mode=='Prediction'):
    st.header("Check your chances of getting Credit card approval here!")
    st.markdown('Start now! Enter your details')
    data=pd.read_csv('Application_Data.csv')
    data.head(3)
    data.drop(['Applicant_Gender','Total_Children','Income_Type','Education_Type','Family_Status','Housing_Type','Owned_Mobile_Phone','Owned_Work_Phone','Owned_Phone','Owned_Email','Job_Title','Total_Family_Members','Applicant_Age','Years_of_Working'],axis=1,inplace=True)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X= data[['Owned_Car','Owned_Realty','Total_Income','Total_Bad_Debt','Total_Good_Debt']]  # Features
    y = data['Status']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gbc_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbc_model.fit(X_train, y_train)
    y_pred_gbc = gbc_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_gbc)
    print('Accuracy:', accuracy)
    print('Classification Report:')
    print(classification_report(y_test, y_pred_gbc))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_gbc))

    pickle_out=open("Credit_card.pkl","wb")
    pickle.dump(gbc_model,pickle_out)
    pickle_out.close()


    Owned_Car=st.number_input("Enter the number of car owned" , value=None, placeholder="Type a number...")
    Owned_Realty=st.number_input("Enter the realty owned", value=None, placeholder="Type a number...")	
    Total_Income=st.slider("Enter total income",0,100000,1000000)
    Total_Bad_Debt=st.number_input("Enter the Total Bad Debt ", value=None, placeholder="Type a number...")	
    Total_Good_Debt=st.number_input("Enter the Total Good Debt", value=None, placeholder="Type a number...")
    if st.button('predict',type="secondary"):
       model=joblib.load("Credit_card.pkl")
       x=np.array([Owned_Car,Owned_Realty,Total_Income,Total_Bad_Debt,Total_Good_Debt])
       st.markdown(f'### Prediction Is {model.predict([[Owned_Car,Owned_Realty,Total_Income,Total_Bad_Debt,Total_Good_Debt]])}')
       st.markdown(f'The accuracy of model is{gbc_model.score(X_train,y_train)}')   
       st.info("Model predicted sucessfully getting 1 means credit card can be approved!!")

elif(app_mode=='Analysis'):
    st.title("Get the insights 📊 of past few years!!!")
    image_path_one="Screenshot (520).png"
    st.image(image_path_one,width=300)
    image_path_two="Screenshot (521).png"
    st.image(image_path_two,width=300)
    image_path_three="Screenshot (522).png"
    st.image(image_path_three,width=300)
    image_path_four="Screenshot (523).png"
    st.image(image_path_four,width=300)
    image_path_five="Screenshot (524).png"
    st.image(image_path_five,width=300)
    image_path_six="Screenshot (525).png"
    st.image(image_path_six,width=300)
    image_path_seven ="Screenshot (526).png"
    st.image(image_path_seven ,width=300)
    image_path_eight ="Screenshot (527).png"
    st.image(image_path_eight,width=300)