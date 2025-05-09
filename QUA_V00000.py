# Importar bibliotecas
import streamlit as st
import speech_recognition as sr
import pandas as pd
from hubspot import HubSpot
from hubspot.crm.objects import ApiException, PublicObjectSearchRequest
import datetime
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import time
from colorama import Fore, Style, init
import tempfile

# Inicializar configuraciones
init(autoreset=True)

# =============================================
# CONFIGURACIÓN SEGURA DE API KEYS
# =============================================
def setup_api_keys():
    """Configura las API keys de forma segura"""
    st.sidebar.title("Configuración de API Keys")
    
    if "HUBSPOT_ACCESS_TOKEN" not in os.environ:
        hubspot_token = st.sidebar.text_input("HubSpot Private App Token:", type="password")
        if hubspot_token:
            os.environ["HUBSPOT_ACCESS_TOKEN"] = hubspot_token
            st.sidebar.success("Token de HubSpot configurado")
    
    if "GOOGLE_API_KEY" not in os.environ:
        google_api_key = st.sidebar.text_input("Google API Key:", type="password")
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            st.sidebar.success("Clave de Google configurada")

    if not os.getenv("HUBSPOT_ACCESS_TOKEN") or not os.getenv("GOOGLE_API_KEY"):
        st.warning("Por favor ingresa ambas claves API para continuar")
        st.stop()

# Configurar API keys al inicio
setup_api_keys()

# =============================================
# INICIALIZACIÓN DE CLIENTES
# =============================================
def initialize_clients():
    """Inicializa los clientes de HubSpot y Google con manejo de errores"""
    try:
        client = HubSpot(access_token=os.environ["HUBSPOT_ACCESS_TOKEN"])
        st.session_state.client = client
    except Exception as e:
        st.error(f"Error al inicializar HubSpot: {str(e)[:200]}")
        if "401" in str(e):
            st.error("Token inválido o expirado. Por favor verifica tu Private App Token.")
        st.stop()

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=120,
            max_retries=3,
        )
        st.session_state.llm = llm
    except Exception as e:
        st.error(f"Error al inicializar Google AI: {str(e)[:200]}")
        st.stop()

# Inicializar clientes
initialize_clients()

# =============================================
# FUNCIONES PRINCIPALES (OPTIMIZADAS)
# =============================================
def fetch_all_calls(fecha_desde, fecha_hasta):
    """Busca llamadas en HubSpot con manejo mejorado de errores"""
    all_results = []
    after = None
    
    with st.spinner("Buscando llamadas en HubSpot..."):
        while True:
            try:
                search_request = PublicObjectSearchRequest(
                    filter_groups=[{
                        "filters": [
                            {"propertyName": "hs_createdate", "operator": "GTE", "value": str(fecha_desde)},
                            {"propertyName": "hs_createdate", "operator": "LTE", "value": str(fecha_hasta)}
                        ]
                    }],
                    properties=["hs_call_recording_url", "hs_createdate", "hs_call_title"],
                    limit=100,
                    after=after
                )
                
                response = st.session_state.client.crm.objects.search_api.do_search("calls", search_request)
                all_results.extend(response.results)
                
                if not response.paging or not response.paging.next:
                    break
                    
                after = response.paging.next.after
                
            except ApiException as e:
                if e.status == 401:
                    st.error("🔐 Error de autenticación. Verifica tu token de HubSpot.")
                    st.stop()
                st.error(f"Error en HubSpot API: {str(e.body)[:200]}")
                break
            except Exception as e:
                st.error(f"Error inesperado: {str(e)[:200]}")
                break
                
    return all_results

def download_call_audio(call_id, url):
    """Descarga grabación de llamada con validaciones"""
    try:
        headers = {
            "Authorization": f"Bearer {os.environ['HUBSPOT_ACCESS_TOKEN']}",
            "User-Agent": "Mozilla/5.0"
        }
        
        with st.spinner(f"Descargando {call_id}..."):
            response = requests.get(url, headers=headers, timeout=(10, 30))
            response.raise_for_status()
            
            if 'audio' not in response.headers.get('Content-Type', ''):
                st.error("El contenido no es un archivo de audio válido")
                return None
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name
            
    except requests.HTTPError as e:
        st.error(f"Error HTTP {e.response.status_code}: {str(e)[:200]}")
    except Exception as e:
        st.error(f"Error al descargar: {str(e)[:200]}")
    return None

def transcribe_audio(audio_path, call_id):
    """Transcribe audio a texto con configuración optimizada"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="es-ES")
            
            with st.expander(f"Transcripción {call_id}"):
                st.text_area("", value=text, height=150)
            return text
            
    except sr.UnknownValueError:
        st.error("No se pudo transcribir el audio (calidad baja o silencio)")
    except Exception as e:
        st.error(f"Error en transcripción: {str(e)[:200]}")
    return None

def analyze_call(transcript):
    """Analiza la transcripción con IA"""
    prompt = """Eres un experto en análisis de llamadas comerciales. Evalúa:
1. ✅ Apertura profesional
2. ✅ Identificación de necesidades  
3. ✅ Presentación de solución
4. ✅ Manejo de objeciones
5. ✅ Cierre efectivo

Para cada punto indica ✅ o ❌ con breve explicación.
Finaliza con puntuación 1-5 y feedback constructivo."""
    
    try:
        response = st.session_state.llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript[:10000]}
        ])
        return response.content
    except Exception as e:
        st.error(f"Error en análisis: {str(e)[:200]}")
    return None

# =============================================
# INTERFAZ DE USUARIO
# =============================================
def main_interface():
    """Interfaz principal de la aplicación"""
    st.set_page_config(
        page_title="Analizador de Llamadas",
        page_icon="📞",
        layout="wide"
    )
    
    st.title("📊 Analizador de Llamadas Comerciales")
    
    # Paso 1: Selección de fechas
    with st.expander("📅 Seleccionar rango de fechas", expanded=True):
        hoy = datetime.date.today()
        col1, col2 = st.columns(2)
        with col1:
            inicio = st.date_input("Desde", hoy-datetime.timedelta(days=7))
        with col2:
            fin = st.date_input("Hasta", hoy)
        
        if inicio > fin:
            st.error("La fecha de inicio debe ser anterior")
            return

    # Paso 2: Buscar llamadas
    with st.expander("🔍 Llamadas encontradas", expanded=False):
        calls = fetch_all_calls(
            int(datetime.datetime.combine(inicio, datetime.time.min).timestamp() * 1000),
            int(datetime.datetime.combine(fin, datetime.time.max).timestamp() * 1000)
        )
        
        if not calls:
            st.warning("No se encontraron llamadas")
            return
            
        valid_calls = [
            {"ID": c.id, "URL": c.properties["hs_call_recording_url"], "Fecha": c.properties["hs_createdate"]}
            for c in calls if c.properties.get("hs_call_recording_url")
        ]
        
        if not valid_calls:
            st.warning("No hay llamadas con grabaciones")
            return
            
        df_calls = pd.DataFrame(valid_calls)
        st.dataframe(df_calls)

    # Paso 3: Selección para análisis
    with st.expander("📌 Seleccionar llamadas a analizar", expanded=False):
        selected = st.multiselect(
            "Selecciona llamadas",
            options=df_calls["ID"].tolist(),
            default=df_calls["ID"].tolist()[:3]
        )
        
        if not selected:
            st.warning("Selecciona al menos una llamada")
            return

    # Paso 4: Procesamiento
    resultados = []
    progress = st.progress(0)
    
    for i, call_id in enumerate(selected, 1):
        call = df_calls[df_calls["ID"] == call_id].iloc[0]
        
        with st.expander(f"Procesando {call_id}"):
            # Descargar
            audio = download_call_audio(call_id, call["URL"])
            if not audio:
                continue
                
            # Transcribir
            text = transcribe_audio(audio, call_id)
            if not text:
                continue
                
            # Analizar
            analysis = analyze_call(text)
            if analysis:
                score = min(5, max(1, analysis.count("✅")))  # Puntaje 1-5
                resultados.append({
                    "ID": call_id,
                    "Fecha": call["Fecha"],
                    "Transcripción": text[:300] + "..." if len(text) > 300 else text,
                    "Análisis": analysis,
                    "Puntaje": score
                })
            
            # Limpiar
            try:
                if os.path.exists(audio):
                    os.unlink(audio)
            except:
                pass
            
        progress.progress(i/len(selected))

    # Resultados
    if resultados:
        st.success(f"Análisis completado para {len(resultados)} llamadas")
        df_resultados = pd.DataFrame(resultados)
        
        with st.expander("📋 Resultados detallados", expanded=True):
            st.dataframe(df_resultados)
            
        # Reporte consolidado
        avg_score = df_resultados["Puntaje"].mean()
        st.metric("Puntaje promedio", f"{avg_score:.1f}/5")
        
        # Gráfico de distribución
        st.bar_chart(df_resultados["Puntaje"].value_counts().sort_index())
    else:
        st.warning("No se pudo completar ningún análisis")

# Ejecutar aplicación
if __name__ == "__main__":
    main_interface()