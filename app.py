# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import io
import streamlit as st

# --- Configuración de página ---
st.set_page_config(page_title='Predicción de Cargas', page_icon='⛏️', layout='wide')

# --- Carga del modelo ---
filename = 'modelo-ensamble-reg-loads.pkl'
modelo, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

# --- Columnas numéricas para normalizar ---
cols_normalizar = [
    'UsodeDisp_6231','UsodeDisp_6232','UsodeDisp_6233','UsodeDisp_6234','UsodeDisp_6235',
    'UsodeDisp_6236','UsodeDisp_6237','UsodeDisp_6238','UsodeDisp_6239','UsodeDisp_6241',
    'UsodeDisp_6242','UsodeDisp_6243','UsodeDisp_6244','UsodeDisp_6245','UsodeDisp_6246',
    'UsodeDisp_6247','UsodeDisp_6248','UsodeDisp_6249','UsodeDisp_6250','UsodeDisp_6260',
    'UsodeDisp_6261','UsodeDisp_6262','UsodeDisp_6263','UsodeDisp_6264','UsodeDisp_6268',
    'UsodeDisp_6269','UsodeDisp_6449','UsodeDisp_6455','UsodeDisp_6457',
    'Tks Availability OB','Tks Utilised Time','OB Cycle time (Min)'
]

def preparar_y_predecir(data):
    data_preparada = data.copy()
    data_preparada = pd.get_dummies(data_preparada, columns=['turno'], drop_first=False, dtype=int)
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
    data_preparada[cols_normalizar] = min_max_scaler.transform(data_preparada[cols_normalizar])
    predicciones = modelo.predict(data_preparada)
    return predicciones

# --- Encabezado ---
st.title('⛏️ Predicción de Cargas — Complex')
st.markdown('Modelo de ensamble para predicción de cargas operacionales.')
st.divider()

# --- Selector de modo ---
modo = st.radio(
    'Modo de predicción',
    ['📋 Registro manual', '📂 Carga masiva (CSV)'],
    horizontal=True
)

st.divider()

# ================================================================
# MODO 1: REGISTRO MANUAL
# ================================================================
if modo == '📋 Registro manual':

    st.subheader('% de utilización por Equipo')
    st.caption('Valores normalizados entre 0.0 y 1.0')

    equipos = [
        '6231','6232','6233','6234','6235',
        '6236','6237','6238','6239','6241',
        '6242','6243','6244','6245','6246',
        '6247','6248','6249','6250','6260',
        '6261','6262','6263','6264','6268',
        '6269','6449','6455','6457'
    ]
    cols = st.columns(5)
    valores = {}
    for i, eq in enumerate(equipos):
        with cols[i % 5]:
            valores[eq] = st.number_input(
                f'Equipo {eq}', min_value=0.0, max_value=1.0,
                value=0.0, step=0.01, format='%.2f', key=f'eq_{eq}'
            )

    st.divider()
    st.subheader('Parámetros Operacionales de Camiones')
    col1, col2, col3 = st.columns(3)
    with col1:
        Tks_Availability_OB = st.number_input('Disponibilidad de camiones', min_value=0.0, max_value=1.0, value=0.0, step=0.01, format='%.2f')
    with col2:
        Tks_Utilization_OB = st.number_input('Utilización de camiones', min_value=0.0, max_value=1.0, value=0.0, step=0.01, format='%.2f')
    with col3:
        Cycle = st.number_input('Tiempo de ciclo (min)', min_value=20.0, max_value=42.0, value=20.0, step=0.1, format='%.1f')

    st.divider()
    st.subheader('Turno')
    Shift = st.radio('Selecciona el turno', ['D', 'N'], horizontal=True,
                     format_func=lambda x: '☀️ Diurno' if x == 'D' else '🌙 Nocturno')

    st.divider()

    if st.button('🔮 Predecir', use_container_width=True, type='primary'):
        datos = [[
            valores['6231'], valores['6232'], valores['6233'], valores['6234'], valores['6235'],
            valores['6236'], valores['6237'], valores['6238'], valores['6239'], valores['6241'],
            valores['6242'], valores['6243'], valores['6244'], valores['6245'], valores['6246'],
            valores['6247'], valores['6248'], valores['6249'], valores['6250'], valores['6260'],
            valores['6261'], valores['6262'], valores['6263'], valores['6264'], valores['6268'],
            valores['6269'], valores['6449'], valores['6455'], valores['6457'],
            Tks_Availability_OB, Tks_Utilization_OB, Cycle, Shift
        ]]
        data = pd.DataFrame(datos, columns=cols_normalizar + ['turno'])

        try:
            Y_pred = preparar_y_predecir(data)
            st.divider()
            st.subheader('Resultado')
            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                st.metric(label='Cargas estimadas', value=f'{int(round(Y_pred[0]))} cargas')
            with col_r2:
                st.info('ℹ️ El modelo tiene un error aproximado del 1.5%')
        except Exception as e:
            st.error(f'Error en la predicción: {e}')

# ================================================================
# MODO 2: CARGA MASIVA
# ================================================================
else:
    st.subheader('Carga masiva de datos')

    with st.expander('ℹ️ Formato requerido del CSV', expanded=False):
        st.markdown("""
        El archivo CSV debe contener las siguientes columnas exactamente con estos nombres:
        """)
        cols_requeridas = cols_normalizar + ['turno']
        st.code(', '.join(cols_requeridas))
        st.markdown("""
        - Las columnas `UsodeDisp_*`, `Tks Availability OB`, `Tks Utilised Time` deben estar entre **0.0 y 1.0**
        - `OB Cycle time (Min)` debe estar entre **20 y 42**
        - `turno` debe ser **D** o **N**
        - Puedes incluir columnas adicionales (fecha, turno ID, etc.) — serán conservadas en el resultado
        """)

        # Botón para descargar plantilla
        plantilla = pd.DataFrame(columns=cols_requeridas)
        csv_plantilla = plantilla.to_csv(index=False).encode('utf-8')
        st.download_button(
            '⬇️ Descargar plantilla CSV',
            data=csv_plantilla,
            file_name='plantilla_prediccion.csv',
            mime='text/csv'
        )

    archivo = st.file_uploader('Sube tu archivo CSV', type=['csv'])

    if archivo is not None:
        try:
            df_input = pd.read_csv(archivo)
            st.success(f'Archivo cargado: {len(df_input)} registros')

            # Validar columnas mínimas
            cols_faltantes = [c for c in cols_normalizar + ['turno'] if c not in df_input.columns]
            if cols_faltantes:
                st.error(f'Faltan las siguientes columnas: {", ".join(cols_faltantes)}')
            else:
                with st.expander('Vista previa de los datos cargados'):
                    st.dataframe(df_input.head(10), use_container_width=True)

                if st.button('🔮 Predecir todos los registros', use_container_width=True, type='primary'):
                    with st.spinner('Calculando predicciones...'):
                        data_pred = df_input[cols_normalizar + ['turno']].copy()
                        predicciones = preparar_y_predecir(data_pred)

                        df_resultado = df_input.copy()
                        df_resultado['Cargas_Predichas'] = np.round(predicciones).astype(int)

                    st.divider()
                    st.subheader(f'Resultados — {len(df_resultado)} predicciones')

                    # Métricas resumen
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric('Total registros', len(df_resultado))
                    c2.metric('Promedio cargas', f'{df_resultado["Cargas_Predichas"].mean():.0f}')
                    c3.metric('Máximo', f'{df_resultado["Cargas_Predichas"].max()}')
                    c4.metric('Mínimo', f'{df_resultado["Cargas_Predichas"].min()}')

                    st.dataframe(df_resultado, use_container_width=True)
                    st.info('ℹ️ El modelo tiene un error aproximado del 1.5%')

                    st.divider()
                    st.subheader('Descargar resultados')
                    col_d1, col_d2 = st.columns(2)

                    # Descarga CSV
                    with col_d1:
                        csv_out = df_resultado.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            '⬇️ Descargar CSV',
                            data=csv_out,
                            file_name='predicciones_cargas.csv',
                            mime='text/csv',
                            use_container_width=True
                        )

                    # Descarga Excel
                    with col_d2:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            df_resultado.to_excel(writer, index=False, sheet_name='Predicciones')
                        buffer.seek(0)
                        st.download_button(
                            '⬇️ Descargar Excel',
                            data=buffer,
                            file_name='predicciones_cargas.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f'Error procesando el archivo: {e}')
