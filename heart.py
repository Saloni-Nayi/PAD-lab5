# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "s20103.sv"
mystreamlitappmodel = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

sex_d = {0:"Woman",1:"Man"}
chestPainType_d = {0:"ATA",1:"NAP", 2:"ABY", 3: "ASY"}
restingECG_d = {0:"Normal", 1:"ST", 2:"LVH"}
exerciseAngina_d = {0:"No", 1:"Yes"}
stslope_d = {0:"No", 1:"Yes"}
heartdisease_d = {0: "Normal", 1: "Fixed defect", 2 : "Reversable defect"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():
    st.set_page_config(page_title="predict heart disease app")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://www.bing.com/images/search?view=detailV2&ccid=%2BFOJN%2FZ3&id=4F4B699509DB0B2DE02F165C5498EE4AEF376E49&thid=OIP.-FOJN_Z3SRUCfGVI2zk2JgHaEj&mediaurl=https%3A%2F%2Fwww.gossipgrasp.com%2Fwp-content%2Fuploads%2F2020%2F08%2Fheart-concept-illustration-1068x656.jpg&cdnurl=https%3A%2F%2Fth.bing.com%2Fth%2Fid%2FR.f8538937f6774915027c6548db393626%3Frik%3DSW4370rumFRcFg%26pid%3DImgRaw%26r%3D0&exph=656&expw=1068&q=heart+disease+images&simid=608029634100267284&form=IRPRST&ck=BE7BBA3C898EA89A609AD69076E9C6F9&selectedindex=9&ajaxhist=0&ajaxserp=0&pivotparams=insightsToken%3Dccid_gBUvxkob*cp_5860B5C98C874B94B673049DF2E987F7*mid_8581B926E4F249817DDCA29F14EBA7905CA58EA3*simid_608010160719481104*thid_OIP.gBUvxkobxSgVpDsyNUPZCQHaE8&vt=0&sim=11&iss=VSI")

    with overview:
        st.title("predict heart disease app")

    with left:
        sex_radio = st.radio( "Gender", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
        chestPainType_radio = st.radio( "Chest Pain Type", list(chestPainType_d.keys()), format_func=lambda x: chestPainType_d[x])
        restingECG_radio = st.radio( "Resting ECG", list(restingECG_d.keys()), index=2, format_func= lambda x: restingECG_d[x])
        exerciseAngina_radio = st.radio( "Exercise Induced Angina", list(exerciseAngina_d.keys()), format_func=lambda x: exerciseAngina_d[x])
        stslope_radio = st.radio( "Slope of the peak exercise ST segment", list(stslope_d.keys()), format_func=lambda x: stslope_d[x])
        heartdisease_radio = st.radio( "Heart disease", list(heartdisease_d.keys()), index=2, format_func= lambda x: heartdisease_d[x])

    with right:
        age_slider = st.slider("Age", value=1, min_value=1, max_value=80)
        cholesterol_slider = st.slider("Cholesterol", min_value=0, max_value=350)
        maxHR_slider = st.slider("MaxHR", min_value=0, max_value=500, step=1)
        oldpeak_slider = st.slider("Oldpeak", min_value=0, max_value=3, step=1)
        restingBP_slider = st.slider("RestingBP", min_value=0, max_value=1)
        fastingBS_slider = st.slider("FastingBS", min_value=0, max_value=1)

        data = [[sex_radio, chestPainType_radio, restingBP_slider, cholesterol_slider,
        fastingBS_slider, restingECG_radio, maxHR_slider, exerciseAngina_radio, oldpeak_slider, stslope_radio, heartdisease_radio]]
        develop_heart_disease = mystreamlitappmodel.predict(data)
        s_confidence = mystreamlitappmodel.predict_proba(data)

    with prediction:
        st.subheader("Would I develop a heart disease?")
        st.subheader(("Yes" if develop_heart_disease[0] == 1 else "No"))
        st.write("Probability {0:.2f} %".format(s_confidence[0][develop_heart_disease][0] * 100))

if __name__ == "__main__":
    main()