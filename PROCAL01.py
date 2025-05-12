# Importar bibliotecas
import streamlit as st
import pandas as pd
from hubspot import HubSpot
from hubspot.crm.objects import ApiException, PublicObjectSearchRequest
import datetime
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import speech_recognition as sr
import matplotlib.pyplot as plt
from datetime import datetime as dt

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Llamadas de Ventas",
    page_icon="📞",
    layout="wide"
)

# Título de la aplicación
st.title("📞 Analizador de Llamadas de Ventas")
st.markdown("""
Esta aplicación analiza grabaciones de llamadas de ventas y evalúa su cumplimiento con los protocolos establecidos.
""")

# Sidebar para credenciales y configuración
with st.sidebar:
    st.header("🔐 Configuración")
    
    # Credenciales de HubSpot
    hubspot_token = st.text_input("Token de API de HubSpot", type="password")
    if hubspot_token:
        os.environ["HUBSPOT_ACCESS_TOKEN"] = hubspot_token
    
    # Credenciales de Google
    google_api_key = st.text_input("API Key de Google Gemini", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Selector de fechas
    st.header("📅 Rango de Fechas")
    col1, col2 = st.columns(2)
    with col1:
        fecha_desde = st.date_input("Fecha de inicio", value=dt.now() - datetime.timedelta(days=7))
    with col2:
        fecha_hasta = st.date_input("Fecha de fin", value=dt.now())
    
    st.info("Selecciona un rango de fechas para buscar llamadas en HubSpot")

# Función para buscar llamadas en HubSpot
@st.cache_data(ttl=3600, show_spinner="Buscando llamadas en HubSpot...")
def buscar_llamadas(fecha_desde, fecha_hasta):
    try:
        # Convertir fechas a timestamp UNIX en milisegundos
        fecha_desde_timestamp = int(fecha_desde.timestamp() * 1000)
        fecha_hasta_timestamp = int(fecha_hasta.timestamp() * 1000)
        
        # Inicializar cliente de HubSpot
        client = HubSpot(access_token=os.environ["HUBSPOT_ACCESS_TOKEN"])
        
        # Definir la búsqueda
        search_request = PublicObjectSearchRequest(
            filter_groups=[{
                "filters": [
                    {
                        "propertyName": "hs_createdate",
                        "operator": "GTE",
                        "value": str(fecha_desde_timestamp)
                    },
                    {
                        "propertyName": "hs_createdate",
                        "operator": "LTE",
                        "value": str(fecha_hasta_timestamp)
                    }
                ]
            }],
            properties=["hs_call_recording_url", "hs_createdate", "hs_call_title"]
        )
        
        # Realizar la búsqueda
        api_response = client.crm.objects.search_api.do_search("calls", search_request)
        results = api_response.results
        
        # Procesar resultados
        llamadas = []
        for result in results:
            url = result.properties.get("hs_call_recording_url")
            if url:
                llamadas.append({
                    "Call ID": result.id,
                    "Título": result.properties.get("hs_call_title", "Sin título"),
                    "Fecha": dt.fromtimestamp(int(result.properties["hs_createdate"])/1000).strftime('%Y-%m-%d %H:%M'),
                    "Recording URL": url
                })
        
        return pd.DataFrame(llamadas)
    
    except ApiException as e:
        st.error(f"Error al buscar llamadas en HubSpot: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return pd.DataFrame()

# Función para descargar audio
def descargar_audio(call_id, recording_url):
    try:
        headers = {"Authorization": f"Bearer {os.environ['HUBSPOT_ACCESS_TOKEN']}"}
        response = requests.get(recording_url, headers=headers)
        response.raise_for_status()
        
        audio_file_path = f"{call_id}.wav"
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(response.content)
        
        return audio_file_path
    except Exception as e:
        st.error(f"Error al descargar la grabación {call_id}: {str(e)}")
        return None

# Función para transcribir audio
def transcribir_audio(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="es-ES")
        return text
    except sr.UnknownValueError:
        st.warning("Google Speech Recognition no pudo entender el audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Error al solicitar resultados de Google Speech Recognition: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado durante la transcripción: {e}")
        return None

# Función para analizar transcripción
def analizar_transcripcion(transcription):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    messages = [
        {
            "role": "system",
            "content": """
Eres un experto en feedback y ventas por teléfono. Analiza esta conversación evaluando su cumplimiento con los PASOS OBLIGATORIOS:

###
PASOS OBLIGATORIOS (✅)

1. **Apertura**:
   - Saludo casual ("¡Hola!")
   - Usar solo el nombre del lead
   - Presentarte solo con tu nombre
   - Mencionar que llamas del Taller de Bienes Raíces con Carlos Devis

2. **Romper el hielo**:
   - Elegir UN solo tema: ciudad, clima, gastronomía o lugares turísticos
   - Hacer preguntas sobre el tema elegido

3. **Identificación del dolor/necesidad**:
   - Preguntar motivación sobre bienes raíces
   - Identificar obstáculos
   - Profundizar con preguntas si no es claro
   - Confirmar el dolor identificado

4. **Presentación de credenciales**:
   - Mencionar los 700+ testimonios de éxito
   - Compartir un ejemplo relevante
   - Preguntar si quisieran lograr resultados similares

5. **Presentación de la metodología**:
   - Explicar los 5 pasos:
     1. Cambio de pensamiento
     2. Organización financiera
     3. Ahorrar
     4. Invertir
     5. Repetir el proceso

6. **Verificar dudas**:
   - Preguntar si hay dudas o preguntas

7. **Presentación de programas**:
   - Mencionar las dos opciones principales:
     - Programa Avanzado ($1,497 USD)
     - Programa Mentoría ($4,999 USD)

8. **Cierre (Obligatorio)**:
   - Mencionar SIEMPRE el precio de página
   - Ofrecer SIEMPRE precio promocional
   - Dar máximo 48 horas de plazo como último recurso

###

Para cada paso, indica si se cumplió (✅) o no (❌) con explicación breve. 
Al final, da una calificación de 0 a 5 (5 = perfecta) y sugerencias de mejora.
"""
        },
        {
            "role": "user",
            "content": transcription
        }
    ]

    try:
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        st.error(f"Error al analizar la transcripción: {e}")
        return None

# Función para extraer calificación
def extraer_calificacion(analisis):
    lines = analisis.split('\n')
    for line in lines:
        if "calificación" in line.lower() and "/5" in line:
            try:
                return float(line.split()[-1].split("/")[0])
            except:
                continue
    return 0

# Interfaz principal
if not hubspot_token or not google_api_key:
    st.warning("Por favor ingresa tus credenciales en la barra lateral para continuar.")
else:
    # Buscar llamadas
    with st.spinner("Buscando llamadas en HubSpot..."):
        df_llamadas = buscar_llamadas(fecha_desde, fecha_hasta)
    
    if df_llamadas.empty:
        st.warning("No se encontraron llamadas con grabaciones en el rango de fechas seleccionado.")
    else:
        st.success(f"Se encontraron {len(df_llamadas)} llamadas con grabaciones.")
        
        # Mostrar tabla de llamadas
        st.subheader("Llamadas Disponibles")
        st.dataframe(
            df_llamadas[["Call ID", "Título", "Fecha"]],
            use_container_width=True,
            hide_index=True
        )
        
        # Selección de llamadas para analizar
        llamadas_seleccionadas = st.multiselect(
            "Selecciona las llamadas a analizar (máximo 3 para evitar costos altos):",
            df_llamadas["Call ID"].tolist(),
            max_selections=3
        )
        
        if st.button("Analizar Llamadas Seleccionadas", disabled=not llamadas_seleccionadas):
            resultados = []
            progreso = st.progress(0)
            total_llamadas = len(llamadas_seleccionadas)
            
            for i, call_id in enumerate(llamadas_seleccionadas):
                try:
                    # Obtener URL de la llamada
                    recording_url = df_llamadas.loc[df_llamadas['Call ID'] == call_id, 'Recording URL'].values[0]
                    
                    # Actualizar UI
                    progreso.progress((i + 1) / total_llamadas, text=f"Procesando llamada {call_id}...")
                    
                    # Descargar audio
                    with st.spinner(f"Descargando grabación {call_id}..."):
                        audio_path = descargar_audio(call_id, recording_url)
                        if not audio_path:
                            continue
                    
                    # Transcribir audio
                    with st.spinner(f"Transcribiendo llamada {call_id}..."):
                        transcripcion = transcribir_audio(audio_path)
                        if not transcripcion:
                            continue
                    
                    # Analizar transcripción
                    with st.spinner(f"Analizando llamada {call_id}..."):
                        analisis = analizar_transcripcion(transcripcion)
                        if not analisis:
                            continue
                    
                    # Guardar resultados
                    resultados.append({
                        "Call ID": call_id,
                        "Transcripción": transcripcion,
                        "Análisis": analisis,
                        "Calificación": extraer_calificacion(analisis)
                    })
                    
                    # Limpiar archivo temporal
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                
                except Exception as e:
                    st.error(f"Error al procesar la llamada {call_id}: {str(e)}")
                    continue
            
            progreso.empty()
            
            if resultados:
                # Mostrar resultados
                st.subheader("Resultados del Análisis")
                
                # Calificación promedio
                df_resultados = pd.DataFrame(resultados)
                promedio = df_resultados["Calificación"].mean()
                
                # Mostrar semáforo
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric("Calificación Promedio", f"{promedio:.1f}/5.0")
                    
                    # Gráfico semaforizado
                    fig, ax = plt.subplots(figsize=(8, 1))
                    color = "green" if promedio >= 4 else "yellow" if promedio >= 2.5 else "red"
                    ax.barh(0, promedio, color=color)
                    ax.set_xlim(0, 5)
                    ax.set_xticks(range(6))
                    ax.set_yticks([])
                    ax.set_title("Desempeño General (Semaforizado)")
                    st.pyplot(fig)
                
                # Mostrar análisis detallado por llamada
                for idx, resultado in enumerate(resultados, 1):
                    with st.expander(f"Análisis de Llamada {resultado['Call ID']} (Calificación: {resultado['Calificación']}/5.0)"):
                        st.subheader("Transcripción")
                        st.text_area("", resultado["Transcripción"], height=200, key=f"trans_{idx}")
                        
                        st.subheader("Análisis Detallado")
                        st.markdown(resultado["Análisis"])
                
                # Opción para descargar resultados
                csv = df_resultados.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar Resultados (CSV)",
                    data=csv,
                    file_name="analisis_llamadas.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No se pudo analizar ninguna llamada. Por favor revisa los errores.")