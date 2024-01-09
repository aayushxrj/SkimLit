import streamlit as st
import tensorflow as tf 

from utils import example_input, make_prediction, make_skimlit_predictions

st.set_page_config(page_title="SkimLit",
                    page_icon="📄",
                    layout="wide",
                    initial_sidebar_state="expanded")

model  = tf.keras.models.load_model("/home/aayushranjan/Codes/SkimLit/skimlit_tribrid_model_10percent_20k_pubmed20k_rct/content/skimlit_tribrid_model_10percent_20k_pubmed20k_rct")

col1, col2 = st.columns(2)

with col1:
    st.write('#### Enter Abstract Here !!')

    abstract=st.text_area(label='', height=50)
    if not abstract:
        st.warning("Kindly input the abstract in the provided text area or refer to the example below for a sample abstract summary.")    
    predict= st.button('Extract !')

    agree = st.checkbox('Show Example Abstract')
    if agree:
        st.info(example_input)
    

def model_prediction():
    objective = ''
    background = ''
    method = ''
    conclusion = ''
    result = ''

    pred, lines = make_prediction(model, abstract)
    # lines,pred = make_skimlit_predictions(model, abstract)

    for i, line in enumerate(lines):
        if pred[i] == 'OBJECTIVE':
            objective = objective + line
        
        elif pred[i] == 'BACKGROUND':
            background = background + line
        
        elif pred[i] == 'METHODS':
            method = method + line
        
        elif pred[i] == 'RESULTS':
            result = result + line
        
        elif pred[i] == 'CONCLUSIONS':
            conclusion = conclusion + line

    return objective, background, method, conclusion, result

if predict:
    with st.spinner('Wait for prediction....'):
        objective, background, method, conclusion, result = model_prediction()
    with col2:
        st.markdown(f'### Objective : ')
        st.write(f'{objective}')
        st.markdown(f'### Background : ')
        st.write(f'{background}')
        st.markdown(f'### Methods : ')
        st.write(f'{method}')
        st.markdown(f'### Result : ')
        st.write(f'{result}')
        st.markdown(f'### Conclusion : ')
        st.write(f'{conclusion}')
