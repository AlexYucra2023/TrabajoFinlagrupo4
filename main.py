import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar los modelos y el encoder guardados
try:
    modelo_rf = joblib.load('churn_model_rf.pkl')
    modelo_xgb = joblib.load('churn_model_xgb.pkl')
    oneHE = joblib.load('oneHE_encoder.pkl')
    st.success("Modelos y encoder cargados correctamente.")
except FileNotFoundError:
    st.error("Error: Asegúrate de que los archivos 'churn_model_rf.pkl', 'churn_model_xgb.pkl' y 'oneHE_encoder.pkl' estén en el mismo directorio.")
    st.stop()  # Detener la ejecución si los archivos no se encuentran

# Estilo de la interfaz de usuario
st.title("📊 Predicción de Cancelación de Clientes (Churn) - Grupo 4")
st.markdown(
    """
    ## Introducción
    Esta herramienta predice si un cliente abandonará el servicio (churn) basándose en varios factores. 
    Selecciona un modelo y llena los datos del cliente para hacer la predicción.
    """
)

# Selección de modelo
model_choice = st.radio("🔍 **Selecciona el modelo para la predicción**", ["Random Forest", "XGBoost"], index=0)

# Subir archivo .pkl (modelo) si se selecciona "Custom model"
uploaded_model = None
if model_choice == "Random Forest":
    uploaded_model = st.file_uploader("📂 Subir el modelo Random Forest (.pkl)", type=["pkl"])
elif model_choice == "XGBoost":
    uploaded_model = st.file_uploader("📂 Subir el modelo XGBoost (.pkl)", type=["pkl"])

# Subir archivo OneHotEncoder
uploaded_oneHE_encoder = st.file_uploader("📂 Subir el OneHotEncoder (.pkl)", type=["pkl"])

if uploaded_model is not None and uploaded_oneHE_encoder is not None:
    try:
        # Cargar el modelo y encoder desde los archivos subidos
        model = joblib.load(uploaded_model)
        oneHE = joblib.load(uploaded_oneHE_encoder)

        # Formulario para ingresar datos de un nuevo cliente manualmente
        st.subheader("🔢 **Ingresa los Datos del Cliente para Predicción**")

        # Campos para ingresar datos del cliente
        gender = st.selectbox("👨‍👩‍👧‍👦 Género", ["Male", "Female"])
        seniorcitizen = st.selectbox("👵 Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("❤️ Tiene pareja", ["No", "Yes"])
        dependents = st.selectbox("👶 Tiene dependientes", ["No", "Yes"])
        phoneservice = st.selectbox("📞 Servicio telefónico", ["No", "Yes"])
        paperlessbilling = st.selectbox("💳 Factura sin papel", ["No", "Yes"])

        tenure = st.number_input("📅 Antigüedad (meses)", min_value=0, max_value=100)
        monthlycharges = st.number_input("💸 Cargo mensual", min_value=0)
        totalcharges = st.number_input("💰 Cargo total", min_value=0)

        # Crear el DataFrame con los datos ingresados
        input_data = pd.DataFrame({
            'gender': [gender],
            'seniorcitizen': [1 if seniorcitizen == "Yes" else 0],
            'partner': [1 if partner == "Yes" else 0],
            'dependents': [1 if dependents == "Yes" else 0],
            'phoneservice': [1 if phoneservice == "Yes" else 0],
            'paperlessbilling': [1 if paperlessbilling == "Yes" else 0],
            'tenure': [tenure],
            'monthlycharges': [monthlycharges],
            'totalcharges': [totalcharges]
        })

        # Preprocesamiento de los datos ingresados
        input_data_encoded = pd.get_dummies(input_data, drop_first=True)

        # Obtener las columnas del modelo dependiendo del tipo de modelo
        if model_choice == "Random Forest":
            model_columns = model.feature_names_in_  # Obtenemos las columnas que fueron usadas en el modelo
        elif model_choice == "XGBoost":
            model_columns = model.get_booster().feature_names  # Obtener los nombres de las características de XGBoost

        # Asegurar que las columnas del DataFrame de entrada coincidan con las del modelo
        input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)

        # Realizar la predicción
        if st.button("🔮 Predecir Churn"):
            try:
                pred_input = model.predict(input_data_encoded)
                prob_input = model.predict_proba(input_data_encoded)[:, 1]

                st.write(f"**Predicción Churn:** {'Yes' if pred_input[0] == 1 else 'No'}")
                st.write(f"**Probabilidad de Churn:** {prob_input[0]:.2f}")
            except Exception as e:
                st.error(f"Ocurrió un error con la predicción: {e}")

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
else:
    st.info("Por favor, sube el archivo del modelo y el OneHotEncoder para continuar.")
