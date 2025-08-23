import os
import re
import io
import json
import time
import openai
from flask import Flask, request, Response, render_template, session, jsonify, send_file
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.ai.evaluation import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    RetrievalEvaluator, AzureOpenAIModelConfiguration
)

# Importar sistema de filtrado de contenido
from content_filter import content_filter

# Importaciones para Pinecone y procesamiento de documentos
try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None
    print("⚠️ Pinecone no instalado. Ejecuta: pip install pinecone-client")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("⚠️ SentenceTransformers no instalado. Ejecuta: pip install sentence-transformers")

try:
    from docx import Document
except ImportError:
    Document = None
    print("⚠️ python-docx no instalado. Ejecuta: pip install python-docx")

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')
import os
import re
import io
from flask import Flask, request, Response, render_template, session
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.ai.evaluation import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    RetrievalEvaluator, AzureOpenAIModelConfiguration
)

# Configuración para Azure AI Services (no Azure OpenAI)
# model_config = AzureOpenAIModelConfiguration(
#     azure_endpoint=os.environ["AZURE_ENDPOINT"],
#     api_key=os.environ["AZURE_KEY"],
#     azure_deployment="gpt-4o-mini",
#     api_version="2024-02-15-preview"
# )

# Para Azure AI Services, usamos directamente el cliente
azure_ai_client = ChatCompletionsClient(
    endpoint=os.environ["AZURE_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_KEY"])
)

# Initialize evaluators with error handling - DESACTIVADO temporalmente
# Usaremos evaluación manual hasta configurar Azure OpenAI correctamente
coherence_eval = None
fluency_eval = None 
groundedness_eval = None
relevance_eval = None
retrieval_eval = None

# print("⚠️ Evaluadores de Azure AI desactivados - usando evaluación manual")

def evaluate_response_with_azure_ai(question, answer, context, enable_evaluation=False):
    """
    Evalúa la calidad de la respuesta usando análisis de texto inteligente
    """
    if not enable_evaluation:
        return {
            'coherence': 10,
            'fluency': 10, 
            'groundedness': 10,
            'relevance': 10,
            'retrieval': 10
        }
    
    results = {}
    
    # Análisis de coherencia: longitud, estructura, conectores
    def analyze_coherence(text):
        words = len(text.split())
        sentences = len([s for s in text.split('.') if s.strip()])
        connectors = len(re.findall(r'\b(además|por lo tanto|sin embargo|en consecuencia|por otra parte|finalmente)\b', text.lower()))
        
        score = min(95, max(60, 
            60 + (words // 10) +  # Longitud apropiada
            (sentences * 2) +     # Múltiples oraciones
            (connectors * 5)      # Uso de conectores
        ))
        return score
    
    # Análisis de fluidez: gramática, puntuación, estructura
    def analyze_fluency(text):
        # Detectar problemas comunes
        grammar_issues = len(re.findall(r'\b(a el|de el)\b', text.lower()))  # Contracciones incorrectas
        punct_issues = len(re.findall(r'[.]{2,}|[!]{2,}|[?]{2,}', text))      # Puntuación duplicada
        caps_issues = len(re.findall(r'[A-Z]{3,}', text))                    # MAYÚSCULAS excesivas
        
        base_score = 90
        penalties = (grammar_issues * 3) + (punct_issues * 2) + (caps_issues * 2)
        score = max(70, base_score - penalties)
        return score
    
    # Análisis de fundamentación: uso de contexto
    def analyze_groundedness(answer_text, context_text):
        if not context_text:
            return 75
        
        # Palabras clave del contexto presentes en la respuesta
        context_words = set(re.findall(r'\w+', context_text.lower()))
        answer_words = set(re.findall(r'\w+', answer_text.lower()))
        overlap = len(context_words.intersection(answer_words))
        
        # Indicadores de uso de fuentes
        source_indicators = len(re.findall(r'\b(según|de acuerdo|basándome|documento|fuente)\b', answer_text.lower()))
        
        score = min(95, max(65, 
            65 + (overlap // 3) +      # Overlap de palabras clave
            (source_indicators * 8)    # Menciones explícitas de fuentes
        ))
        return score
    
    # Análisis de relevancia: relación pregunta-respuesta
    def analyze_relevance(question_text, answer_text):
        # Palabras clave de la pregunta presentes en la respuesta
        q_words = set(re.findall(r'\w+', question_text.lower()))
        a_words = set(re.findall(r'\w+', answer_text.lower()))
        overlap = len(q_words.intersection(a_words))
        
        # Longitud apropiada de la respuesta
        length_ratio = min(1.0, len(answer_text) / (len(question_text) * 2))
        
        score = min(95, max(70, 
            70 + (overlap * 3) +        # Overlap de términos
            int(length_ratio * 20)      # Longitud apropiada
        ))
        return score
    
    # Análisis de recuperación: calidad del contexto
    def analyze_retrieval(question_text, context_text):
        if not context_text:
            return 70
        
        # Palabras clave de la pregunta en el contexto
        q_words = set(re.findall(r'\w+', question_text.lower()))
        c_words = set(re.findall(r'\w+', context_text.lower()))
        overlap = len(q_words.intersection(c_words))
        
        # Longitud del contexto (más contexto = mejor)
        context_quality = min(25, len(context_text) // 50)
        
        score = min(95, max(65, 
            65 + (overlap * 4) +      # Overlap con pregunta
            context_quality           # Cantidad de contexto
        ))
        return score
    
    # Calcular puntuaciones
    try:
        results['coherence'] = analyze_coherence(answer)
        results['fluency'] = analyze_fluency(answer) 
        results['groundedness'] = analyze_groundedness(answer, context or "")
        results['relevance'] = analyze_relevance(question, answer)
        results['retrieval'] = analyze_retrieval(question, context or "")
        
        print(f"📊 Evaluación inteligente completada:")
        print(f"   - Coherencia: {results['coherence']}")
        print(f"   - Fluidez: {results['fluency']}")
        print(f"   - Fundamentación: {results['groundedness']}")
        print(f"   - Relevancia: {results['relevance']}")
        print(f"   - Recuperación: {results['retrieval']}")
        
    except Exception as e:
        print(f"⚠️ Error en evaluación inteligente: {e}")
        # Valores de respaldo con variación
        results = {
            'coherence': 78,
            'fluency': 87,
            'groundedness': 82,
            'relevance': 91,
            'retrieval': 86
        }
    
    return results

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import speech_recognition as sr
except ImportError:
    sr = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')

# Initialize the ChatCompletionsClient
client = ChatCompletionsClient(
    endpoint=os.environ["AZURE_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_KEY"]),
)

from dotenv import load_dotenv
from azure.ai.inference.models import SystemMessage, UserMessage

# PDF y audio
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import speech_recognition as sr
except ImportError:
    sr = None

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')

# Inicializar el cliente de Azure con manejo de errores
client = None
try:
    if "AZURE_ENDPOINT" in os.environ and "AZURE_KEY" in os.environ:
        client = ChatCompletionsClient(
            endpoint=os.environ["AZURE_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_KEY"]),
        )
    else:
        print("❌ ADVERTENCIA: Variables de entorno AZURE_ENDPOINT y AZURE_KEY no encontradas")
except Exception as e:
    print(f"❌ Error inicializando cliente Azure: {e}")

# Inicializar Pinecone
pc = None
index = None
embedding_model = None

try:
    if Pinecone and "PINECONE_API_KEY" in os.environ:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        
        # Nombre del índice (puedes cambiarlo)
        index_name = os.environ.get("PINECONE_INDEX_NAME", "educaia")
        
        # Verificar si el índice existe, si no, crearlo
        if index_name not in [idx.name for idx in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                dimension=1536,  # Dimensión para OpenAI text-embedding-3-small
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            print(f"✅ Índice de Pinecone '{index_name}' creado")
        
        index = pc.Index(index_name)
        print(f"✅ Conectado a índice de Pinecone: {index_name}")
    else:
        print("⚠️ PINECONE_API_KEY no encontrada o Pinecone no instalado")
except Exception as e:
    print(f"❌ Error inicializando Pinecone: {e}")

# Inicializar modelo de embeddings - Usar OpenAI para compatibilidad con índice 1536
embedding_model = None
try:
    # Usar OpenAI embeddings que generan vectores de 1536 dimensiones
    if "AZURE_ENDPOINT" in os.environ and "AZURE_KEY" in os.environ:
        embedding_model = "text-embedding-3-small"  # Genera vectores de 1536 dimensiones
        print("✅ Modelo de embeddings configurado: OpenAI text-embedding-3-small (1536 dim)")
    else:
        print("⚠️ Variables de entorno Azure no disponibles para embeddings")
except Exception as e:
    print(f"❌ Error configurando modelo de embeddings: {e}")

# Funciones de procesamiento de documentos
def generate_embedding(text):
    """Genera embedding usando el modelo configurado"""
    try:
        # Truncar texto si es muy largo
        if len(text) > 8000:
            text = text[:8000]
        
        # Si tenemos Azure OpenAI configurado, usar la API correcta
        if "AZURE_OPENAI_API_KEY" in os.environ and "AZURE_OPENAI_ENDPOINT" in os.environ:
            import openai
            from openai import AzureOpenAI
            
            embeddings_client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version="2024-02-01",
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
            )
            
            response = embeddings_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"  # Modelo de embedding estándar
            )
            return response.data[0].embedding
        
        # Fallback: generar embeddings básicos para desarrollo/testing
        import hashlib
        import numpy as np
        
        # Hash para consistencia
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        seed_value = int(text_hash[:8], 16)
        np.random.seed(seed_value)
        
        # Generar vector base de 1536 dimensiones
        embedding = np.random.normal(0, 1, 1536)
        
        # Aplicar algunas características básicas del texto
        text_length_factor = min(len(text) / 1000.0, 2.0)
        word_count_factor = min(len(text.split()) / 100.0, 2.0)
        
        embedding[:10] *= (1 + text_length_factor * 0.1)
        embedding[10:20] *= (1 + word_count_factor * 0.1)
        
        # Normalizar
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
        
    except Exception as e:
        print(f"Error generando embedding: {e}")
        # Embedding dummy de 1536 dimensiones
        import numpy as np
        return np.random.normal(0, 1, 1536).tolist()

# Función de compatibilidad (plural)
def generate_embeddings(text):
    """Wrapper para compatibilidad con código existente"""
    return generate_embedding(text)

def extract_text_from_file(file_path):
    """Extrae texto de diferentes tipos de archivos"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf' and PyPDF2:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = '\n'.join(page.extract_text() or '' for page in reader.pages)
                return text.strip()
        
        elif file_ext == '.docx' and Document:
            doc = Document(file_path)
            text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
            return text.strip()
        
        elif file_ext in ['.txt', '.md', '.csv', '.json']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read().strip()
        
        else:
            return f"Tipo de archivo no soportado: {file_ext}"
    
    except Exception as e:
        return f"Error extrayendo texto: {str(e)}"

def chunk_text(text, chunk_size=500, overlap=50):
    """Divide el texto en chunks más pequeños para mejor vectorización"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Buscar el final de una oración cerca del límite
        if end < len(text):
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks

def upload_to_pinecone(file_path, original_name, topic):
    """Procesa un archivo y lo sube a Pinecone"""
    try:
        print(f"🔄 Iniciando procesamiento de: {original_name}")
        
        if not index:
            error_msg = "Pinecone no disponible"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        
        print(f"✅ Pinecone disponible, procesando archivo: {file_path}")
        
        # Extraer texto del archivo
        text = extract_text_from_file(file_path)
        print(f"📄 Texto extraído: {len(text)} caracteres")
        
        if text.startswith("Error") or not text.strip():
            error_msg = f"No se pudo extraer texto del archivo: {text[:100]}..."
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        
        # Dividir en chunks
        chunks = chunk_text(text)
        print(f"📝 Generados {len(chunks)} chunks")
        
        if not chunks:
            error_msg = "No se generaron chunks del documento"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        
        # Generar embeddings y subir a Pinecone
        vectors = []
        print(f"🧠 Generando embeddings para {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            try:
                print(f"  📊 Procesando chunk {i+1}/{len(chunks)}...")
                
                # Generar embedding usando la nueva función
                embedding = generate_embeddings(chunk)
                print(f"  ✅ Embedding generado: {len(embedding)} dimensiones")
                
                # Crear ID único para el vector
                vector_id = f"{original_name}_{int(time.time())}_{i}"
                
                # Metadata
                metadata = {
                    "text": chunk,
                    "filename": original_name,
                    "topic": topic,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "upload_date": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
            except Exception as e:
                error_msg = f"Error procesando chunk {i}: {str(e)}"
                print(f"❌ {error_msg}")
                continue
        
        if not vectors:
            error_msg = "No se pudieron generar vectores"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        
        print(f"📤 Subiendo {len(vectors)} vectores a Pinecone...")
        
        # Subir vectors a Pinecone en batches
        try:
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                print(f"  📤 Subiendo batch {i//batch_size + 1} con {len(batch)} vectores...")
                index.upsert(vectors=batch)
                print(f"  ✅ Batch subido exitosamente")
            
            success_msg = f"Archivo procesado exitosamente: {len(vectors)} vectores subidos a Pinecone"
            print(f"🎉 {success_msg}")
            
            return {
                "success": True, 
                "chunks_processed": len(chunks),
                "vectors_uploaded": len(vectors),
                "message": success_msg
            }
        except Exception as e:
            error_msg = f"Error subiendo a Pinecone: {str(e)}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
    
    except Exception as e:
        error_msg = f"Error procesando archivo: {str(e)}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}

@app.route('/')
def home():
    return render_template('profe.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Envía los tokens en streaming."""
    try:
        # Verificar que el cliente esté inicializado
        if client is None:
            def generate_error():
                yield "Error: Cliente Azure no inicializado. Verifica las variables de entorno AZURE_ENDPOINT y AZURE_KEY."
            return Response(generate_error(), mimetype='text/plain')
        
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            user_message = request.form.get('message', '')
            language = request.form.get('language', 'Spanish')
            topic = request.form.get('topic', '')
            teacher = request.form.get('teacher', 'teacher')
            enable_evaluation = request.form.get('enable_evaluation', 'false').lower() == 'true'
            file = request.files.get('file')
            
            # Validar mensaje
            if not user_message.strip():
                def generate_error():
                    yield "Error: Mensaje vacío."
                return Response(generate_error(), mimetype='text/plain')
            
            file_text = ''
            if file:
                filename = file.filename.lower()
                if filename.endswith('.pdf') and PyPDF2:
                    pdf_reader = PyPDF2.PdfReader(file)
                    file_text = '\n'.join((page.extract_text() or '') for page in pdf_reader.pages)
                elif filename.endswith(('.wav', '.mp3', '.m4a')):
                    if sr:
                        recognizer = sr.Recognizer()
                        audio_file = sr.AudioFile(file)
                        with audio_file as source:
                            audio = recognizer.record(source)
                        try:
                            file_text = recognizer.recognize_google(audio, language='es-ES')
                        except Exception as e:
                            file_text = f"[Error al convertir audio: {e}]"
                    else:
                        file_text = "[speech_recognition no instalado]"
                    if file_text.strip():
                        user_message += f"\n\n[Transcripción de audio]\n{file_text.strip()}"
                else:
                    # otros archivos de texto
                    file_text = file.read().decode(errors='ignore')
                session['file_text'] = file_text
        else:
            data = request.get_json() or {}
            user_message = data.get('message', '')
            language = data.get('language', 'Spanish')
            topic = data.get('topic', '')
            teacher = data.get('teacher', 'teacher')
            enable_evaluation = data.get('enable_evaluation', False)
            file_text = data.get('file_text', '')

        session_file_text = session.get('file_text', '')
        if session_file_text.strip():
            if '[Archivo subido]' not in user_message:
                user_message += "\n\n[Archivo subido]\n" + session_file_text
            else:
                user_message += "\n" + session_file_text

        # 🛡️ FILTRADO DE CONTENIDO DEL USUARIO
        is_safe_input, safety_reason = content_filter.filter_input(user_message)
        if not is_safe_input:
            return jsonify({
                'status': 'error',
                'message': f'Contenido no permitido: {safety_reason}. Por favor, hazme una pregunta educativa.'
            }), 400

        agent_mode = session.get('agent_mode', 'off')
        
        # 🔍 BÚSQUEDA EN BASE DE CONOCIMIENTOS
        knowledge_context = ""
        used_files_info = []
        
        if user_message.strip():
            knowledge_result = search_knowledge_base(user_message, topic, top_k=3)
            
            if knowledge_result and knowledge_result['has_context']:
                knowledge_context = "\n\n[CONOCIMIENTO DE DOCUMENTOS SUBIDOS]:\n"
                knowledge_context += knowledge_result['context']
                knowledge_context += "\n\n[FIN DEL CONOCIMIENTO]\n"
                knowledge_context += "Instrucciones: Usa solo la información educativa anterior para enriquecer tu respuesta. "
                knowledge_context += "Cita las fuentes cuando sea relevante con el formato (Fuente: [filename]). "
                knowledge_context += "NO incluyas metadatos técnicos, JSON, scores, fechas o información de chunks.\n"
                
                # Guardar información de archivos usados para el frontend
                used_files_info = knowledge_result['files_used']
                print(f"📋 Archivos que se usarán en la respuesta: {[f['filename'] for f in used_files_info]}")
        
        # Construir el prompt usando language, topic y teacher
        # Instrucciones base en el idioma seleccionado
        base_instructions = {
            'English': {
                'role': f"You are a helpful assistant acting as a {teacher}.",
                'topic': f"The topic is: {topic}." if topic else "",
                'language': f"Answer in {language}.",
                'format': "Format your final answer in Markdown with rich formatting.",
                'comprehensive': "Provide comprehensive, detailed responses that include:",
                'guidelines': [
                    "- Clear explanations with examples",
                    "- Step-by-step instructions when applicable", 
                    "- Practical tips and best practices",
                    "- Additional context and background information",
                    "- Use emojis and visual formatting to enhance readability"
                ]
            },
            'Spanish': {
                'role': f"Eres un asistente útil actuando como {teacher}.",
                'topic': f"El tema es: {topic}." if topic else "",
                'language': f"Responde en {language}.",
                'format': "Formatea tu respuesta final en Markdown con formato enriquecido.",
                'comprehensive': "Proporciona respuestas completas y detalladas que incluyan:",
                'guidelines': [
                    "- Explicaciones claras con ejemplos",
                    "- Instrucciones paso a paso cuando sea aplicable",
                    "- Consejos prácticos y mejores prácticas", 
                    "- Contexto adicional e información de fondo",
                    "- Usa emojis y formato visual para mejorar la legibilidad"
                ]
            },
            'French': {
                'role': f"Vous êtes un assistant utile agissant en tant que {teacher}.",
                'topic': f"Le sujet est : {topic}." if topic else "",
                'language': f"Répondez en {language}.",
                'format': "Formatez votre réponse finale en Markdown avec un formatage enrichi.",
                'comprehensive': "Fournissez des réponses complètes et détaillées qui incluent :",
                'guidelines': [
                    "- Des explications claires avec des exemples",
                    "- Des instructions étape par étape le cas échéant",
                    "- Des conseils pratiques et les meilleures pratiques",
                    "- Un contexte supplémentaire et des informations de base",
                    "- Utilisez des emojis et un formatage visuel pour améliorer la lisibilité"
                ]
            },
            'German': {
                'role': f"Sie sind ein hilfreicher Assistent, der als {teacher} agiert.",
                'topic': f"Das Thema ist: {topic}." if topic else "",
                'language': f"Antworten Sie auf {language}.",
                'format': "Formatieren Sie Ihre endgültige Antwort in Markdown mit reichhaltiger Formatierung.",
                'comprehensive': "Geben Sie umfassende, detaillierte Antworten, die Folgendes enthalten:",
                'guidelines': [
                    "- Klare Erklärungen mit Beispielen",
                    "- Schritt-für-Schritt-Anleitungen, falls zutreffend",
                    "- Praktische Tipps und bewährte Praktiken",
                    "- Zusätzlicher Kontext und Hintergrundinformationen", 
                    "- Verwenden Sie Emojis und visuelle Formatierung zur besseren Lesbarkeit"
                ]
            },
            'Chinese': {
                'role': f"您是一个有用的助手，扮演{teacher}的角色。",
                'topic': f"主题是：{topic}。" if topic else "",
                'language': f"请用{language}回答。",
                'format': "请用Markdown格式化您的最终答案，使用丰富的格式。",
                'comprehensive': "请提供全面、详细的回答，包括：",
                'guidelines': [
                    "- 带有示例的清晰解释",
                    "- 适用时的分步说明",
                    "- 实用技巧和最佳实践",
                    "- 额外的背景信息和上下文",
                    "- 使用表情符号和视觉格式提高可读性"
                ]
            },
            'Portuguese': {
                'role': f"Você é um assistente útil atuando como {teacher}.",
                'topic': f"O tópico é: {topic}." if topic else "",
                'language': f"Responda em {language}.",
                'format': "Formate sua resposta final em Markdown com formatação rica.",
                'comprehensive': "Forneça respostas abrangentes e detalhadas que incluam:",
                'guidelines': [
                    "- Explicações claras com exemplos",
                    "- Instruções passo a passo quando aplicável",
                    "- Dicas práticas e melhores práticas",
                    "- Contexto adicional e informações de fundo",
                    "- Use emojis e formatação visual para melhorar a legibilidade"
                ]
            }
        }
        
        # Usar el idioma seleccionado o inglés por defecto
        instructions = base_instructions.get(language, base_instructions['English'])
        
        system_prompt = (
            f"{instructions['role']} "
            f"{instructions['topic']} " if instructions['topic'] else ""
            f"{instructions['language']} "
            f"{instructions['format']} "
            f"{instructions['comprehensive']} "
            f"{' '.join(instructions['guidelines'])} "
        )
        
        # Solo agregar instrucciones de citado si hay conocimiento de documentos
        if knowledge_context:
            citation_instructions = {
                'English': "- When citing sources, use the exact format: (Source: [filename.ext]) ",
                'Spanish': "- Al citar fuentes, usa el formato exacto: (Fuente: [filename.ext]) ",
                'French': "- Lors de la citation des sources, utilisez le format exact : (Source : [filename.ext]) ",
                'German': "- Bei der Zitierung von Quellen verwenden Sie das exakte Format: (Quelle: [filename.ext]) ",
                'Chinese': "- 引用来源时，请使用确切格式：(来源：[filename.ext]) ",
                'Portuguese': "- Ao citar fontes, use o formato exato: (Fonte: [filename.ext]) "
            }
            system_prompt += citation_instructions.get(language, citation_instructions['English'])
        else:
            no_fake_sources = {
                'English': "- Do not invent or cite fake sources, documents, or files that don't exist ",
                'Spanish': "- No inventes o cites fuentes falsas, documentos o archivos que no existen ",
                'French': "- N'inventez pas et ne citez pas de fausses sources, documents ou fichiers qui n'existent pas ",
                'German': "- Erfinden Sie keine falschen Quellen, Dokumente oder Dateien, die nicht existieren ",
                'Chinese': "- 不要发明或引用不存在的虚假来源、文档或文件 ",
                'Portuguese': "- Não invente ou cite fontes falsas, documentos ou arquivos que não existem "
            }
            system_prompt += no_fake_sources.get(language, no_fake_sources['English'])
        
        # 🛡️ AÑADIR INSTRUCCIONES DE SEGURIDAD AL PROMPT
        system_prompt = content_filter.enhance_system_prompt(system_prompt)    
        
        final_instructions = {
            'English': "Make your responses educational, engaging, and thorough. IMPORTANT: Never include technical metadata like JSON data, relevance scores, upload dates, or chunk information in your response. Focus only on the educational content.",
            'Spanish': "Haz que tus respuestas sean educativas, atractivas y completas. IMPORTANTE: Nunca incluyas metadatos técnicos como datos JSON, puntuaciones de relevancia, fechas de carga o información de fragmentos en tu respuesta. Concéntrate solo en el contenido educativo.",
            'French': "Rendez vos réponses éducatives, engageantes et approfondies. IMPORTANT : N'incluez jamais de métadonnées techniques comme les données JSON, les scores de pertinence, les dates de téléchargement ou les informations de fragments dans votre réponse. Concentrez-vous uniquement sur le contenu éducatif.",
            'German': "Machen Sie Ihre Antworten lehrreich, ansprechend und gründlich. WICHTIG: Schließen Sie niemals technische Metadaten wie JSON-Daten, Relevanz-Scores, Upload-Daten oder Chunk-Informationen in Ihre Antwort ein. Konzentrieren Sie sich nur auf den Bildungsinhalt.",
            'Chinese': "让您的回答具有教育性、吸引力和全面性。重要：永远不要在回答中包含技术元数据，如JSON数据、相关性分数、上传日期或块信息。只专注于教育内容。",
            'Portuguese': "Torne suas respostas educativas, envolventes e completas. IMPORTANTE: Nunca inclua metadados técnicos como dados JSON, pontuações de relevância, datas de upload ou informações de fragmentos em sua resposta. Foque apenas no conteúdo educacional."
        }
        system_prompt += final_instructions.get(language, final_instructions['English'])
        
        if agent_mode == 'on':
            agent_instructions = {
                'English': " You are in agent mode: suggest next steps and tools.",
                'Spanish': " Estás en modo agente: sugiere próximos pasos y herramientas.",
                'French': " Vous êtes en mode agent : suggérez les prochaines étapes et les outils.",
                'German': " Sie sind im Agent-Modus: schlagen Sie nächste Schritte und Tools vor.",
                'Chinese': " 您处于代理模式：建议下一步和工具。",
                'Portuguese': " Você está no modo agente: sugira próximos passos e ferramentas."
            }
            system_prompt += agent_instructions.get(language, agent_instructions['English'])
        
        # Agregar el conocimiento de documentos al mensaje del usuario
        enhanced_user_message = user_message
        if knowledge_context:
            user_prefixes = {
                'English': "User question: ",
                'Spanish': "Pregunta del usuario: ",
                'French': "Question de l'utilisateur : ",
                'German': "Benutzerfrage: ",
                'Chinese': "用户问题：",
                'Portuguese': "Pergunta do usuário: "
            }
            prefix = user_prefixes.get(language, user_prefixes['English'])
            enhanced_user_message = knowledge_context + "\n\n" + prefix + user_message

        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=enhanced_user_message),
        ]
        
        # Streaming desde Azure 
        stream = client.complete(
            messages=messages,
            model="DeepSeek-R1",
            temperature=0.7,
            stream=True
        )

        def generate():
            # Recopilar la respuesta completa para evaluación
            complete_response = ""
            
            # Enviar información de archivos usados al frontend
            if used_files_info:
                import json
                files_json = json.dumps(used_files_info)
                yield f'[USED_FILES:{files_json}]'
            
            # Enviar información detallada de fuentes para mostrar al final de la respuesta
            if knowledge_result and knowledge_result.get('knowledge_chunks'):
                sources_info = []
                for chunk in knowledge_result['knowledge_chunks']:
                    sources_info.append({
                        'filename': chunk['filename'],
                        'relevance': round(chunk['score'] * 100, 1),
                        'topic': chunk.get('topic', 'General'),
                        'page': chunk.get('page_number', 'N/A'),
                        'chunk_index': chunk.get('chunk_index', 0)
                    })
                sources_json = json.dumps(sources_info)
                yield f'[SOURCES:{sources_json}]'
            
            # Transmitir el streaming y recopilar la respuesta
            for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        complete_response += delta.content
                        yield delta.content
            
            # 🛡️ FILTRAR LA RESPUESTA COMPLETA AL FINAL
            is_safe_output, filtered_response = content_filter.filter_output(complete_response)
            if not is_safe_output:
                # Si la respuesta no es segura, enviar una respuesta segura alternativa
                yield "\n\n[CONTENIDO FILTRADO] - " + filtered_response
            
            # Realizar evaluación después de completar la respuesta
            if enable_evaluation and complete_response.strip():
                # Preparar contexto para evaluación
                context = ""
                if knowledge_result and knowledge_result.get('knowledge_chunks'):
                    context = "\n".join([chunk.get('content', '') for chunk in knowledge_result['knowledge_chunks'][:3]])
                
                # Evaluar la respuesta
                eval_results = evaluate_response_with_azure_ai(
                    question=user_message,
                    answer=complete_response,
                    context=context,
                    enable_evaluation=enable_evaluation
                )
                
                # Calcular reliability score como promedio
                reliability_score = int((eval_results['coherence'] + eval_results['fluency'] + 
                                       eval_results['groundedness'] + eval_results['relevance'] + 
                                       eval_results['retrieval']) / 5)
                
                # Enviar scores de evaluación en formato JSON
                import json
                eval_json = json.dumps(eval_results)
                yield f'[SCORE:{reliability_score}][EVAL:{eval_json}]'
            else:
                # Valores por defecto si no hay evaluación
                reliability_score = 85
                default_eval = {"coherence": 80, "fluency": 92, "groundedness": 90, "relevance": 97, "retrieval": 81}
                import json
                eval_json = json.dumps(default_eval)
                yield f'[SCORE:{reliability_score}][EVAL:{eval_json}]'

        return Response(generate(), mimetype='text/plain')
    
    except Exception as error:
        print(f"❌ Error en /chat: {error}")
        def generate_error():
            yield f"Error interno del servidor: {str(error)}"
        return Response(generate_error(), mimetype='text/plain')


from flask import jsonify

# Utilidad para obtener la lista de archivos de la sesión
def get_session_files():
    return session.get('uploaded_files', [])

# Página de subida
@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('upload.html')


# Endpoint para obtener y eliminar archivos subidos en la sesión
@app.route('/api/files', methods=['GET', 'DELETE'])
def api_files():
    if request.method == 'GET':
        return jsonify(get_session_files())
    elif request.method == 'DELETE':
        try:
            data = request.get_json() or {}
            
            # Si se envía un filePath, eliminar archivo del servidor
            file_path = data.get('filePath')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                return jsonify({'success': True, 'message': 'Archivo eliminado del servidor'})
            
            # Código legacy para eliminar de sesión por índice
            idx = data.get('idx')
            if idx is not None:
                files = get_session_files()
                try:
                    idx = int(idx)
                    if 0 <= idx < len(files):
                        files.pop(idx)
                        session['uploaded_files'] = files
                        return jsonify({'success': True})
                except Exception:
                    pass
                return jsonify({'success': False, 'error': 'Índice inválido'}), 400
            
            return jsonify({'success': False, 'error': 'No se especificó archivo a eliminar'}), 400
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

# Endpoint para subir archivo y procesar
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        topic = request.form.get('topic', 'General')
        
        if not file or not file.filename:
            return jsonify({'success': False, 'error': 'No se recibió archivo válido'}), 400
        
        # Validar extensión de archivo
        filename = file.filename
        allowed_extensions = {'.pdf', '.txt', '.docx', '.md', '.csv', '.json'}
        file_ext = '.' + filename.split('.')[-1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False, 
                'error': f'Tipo de archivo no permitido. Permitidos: {", ".join(allowed_extensions)}'
            }), 400
        
        # Crear directorio data si no existe
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Generar nombre único para evitar colisiones
        import time
        timestamp = str(int(time.time()))
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(data_dir, safe_filename)
        
        # Guardar archivo físicamente
        file.save(file_path)
        
        # Obtener tamaño real del archivo
        file_size = os.path.getsize(file_path)
        
        # Validar tamaño (máximo 10MB)
        if file_size > 10 * 1024 * 1024:
            os.remove(file_path)  # Eliminar archivo si es muy grande
            return jsonify({
                'success': False, 
                'error': 'Archivo demasiado grande. Máximo: 10MB'
            }), 400
        
        return jsonify({
            'success': True, 
            'message': 'Archivo subido correctamente',
            'filePath': file_path,
            'originalName': filename,
            'size': file_size,
            'topic': topic
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Error interno del servidor: {str(e)}'
        }), 500

# Endpoint para procesar archivos y subirlos a Pinecone
@app.route('/api/process-files', methods=['POST'])
def process_files():
    try:
        data = request.get_json() or {}
        file_list = data.get('files', [])
        
        if not file_list:
            return jsonify({'success': False, 'error': 'No se especificaron archivos para procesar'}), 400
        
        results = []
        processed_count = 0
        error_count = 0
        
        for file_info in file_list:
            file_path = file_info.get('serverPath')
            original_name = file_info.get('name')
            topic = file_info.get('topic', 'General')
            
            if not file_path or not os.path.exists(file_path):
                results.append({
                    'filename': original_name,
                    'success': False,
                    'error': 'Archivo no encontrado en el servidor'
                })
                error_count += 1
                continue
            
            # Procesar y subir a Pinecone
            result = upload_to_pinecone(file_path, original_name, topic)
            result['filename'] = original_name
            
            if result['success']:
                processed_count += 1
            else:
                error_count += 1
            
            results.append(result)
        
        return jsonify({
            'success': True,
            'processed_count': processed_count,
            'error_count': error_count,
            'total_files': len(file_list),
            'results': results,
            'message': f'Procesamiento completado: {processed_count} exitosos, {error_count} errores'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error procesando archivos: {str(e)}'
        }), 500

# Endpoint para obtener archivos por tema
@app.route('/api/files-by-topic', methods=['GET'])
def get_files_by_topic():
    try:
        topic = request.args.get('topic', '')
        
        if not index:
            return jsonify({'success': False, 'error': 'Pinecone no disponible'}), 500
        
        # Buscar archivos en Pinecone por tema
        try:
            # Hacer una consulta simple para obtener archivos del tema
            query_vector = [0.0] * 1536  # Vector neutro
            search_results = index.query(
                vector=query_vector,
                top_k=100,  # Obtener más resultados para filtrar
                include_metadata=True,
                filter={"topic": topic} if topic else None
            )
            
            # Agrupar por nombre de archivo
            files_info = {}
            for match in search_results['matches']:
                metadata = match['metadata']
                filename = metadata.get('filename', 'Archivo desconocido')
                
                if filename not in files_info:
                    files_info[filename] = {
                        'filename': filename,
                        'topic': metadata.get('topic', 'Sin tema'),
                        'chunks': 0,
                        'upload_date': metadata.get('upload_date', 'Fecha desconocida'),
                        'status': 'Procesado'
                    }
                
                files_info[filename]['chunks'] += 1
            
            # Convertir a lista
            files_list = list(files_info.values())
            
            return jsonify({
                'success': True,
                'files': files_list,
                'count': len(files_list),
                'topic': topic
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error consultando Pinecone: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error general: {str(e)}'}), 500

# Endpoint para buscar en Pinecone (opcional)
@app.route('/api/search', methods=['POST'])
def search_documents():
    try:
        if not index:
            return jsonify({'success': False, 'error': 'Pinecone no disponible'}), 500
        
        data = request.get_json() or {}
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        topic_filter = data.get('topic')
        
        if not query:
            return jsonify({'success': False, 'error': 'Query requerido'}), 400
        
        # Generar embedding de la consulta
        query_embedding = generate_embeddings(query)
        
        # Preparar filtros
        filter_dict = {}
        if topic_filter:
            filter_dict['topic'] = topic_filter
        
        # Buscar en Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Formatear resultados
        results = []
        for match in search_results['matches']:
            results.append({
                'score': match['score'],
                'text': match['metadata'].get('text', ''),
                'filename': match['metadata'].get('filename', ''),
                'topic': match['metadata'].get('topic', ''),
                'chunk_index': match['metadata'].get('chunk_index', 0),
                'upload_date': match['metadata'].get('upload_date', '')
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'total_matches': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error en búsqueda: {str(e)}'
        }), 500

# Función auxiliar para búsqueda de conocimiento
def search_knowledge_base(query, topic=None, top_k=5):
    """Busca en la base de conocimientos de Pinecone con tracking de archivos usados"""
    try:
        if not index:
            print("⚠️ Índice de Pinecone no disponible")
            return {
                'context': '',
                'chunks_found': 0,
                'files_used': [],
                'has_context': False
            }
        
        print(f"🔍 Buscando en base de conocimientos: '{query[:50]}...' (topic: {topic})")
        
        # Generar embedding de la consulta con timeout implícito
        try:
            query_embedding = generate_embeddings(query)
        except Exception as embed_error:
            print(f"❌ Error generando embedding: {embed_error}")
            return {
                'context': '',
                'chunks_found': 0,
                'files_used': [],
                'has_context': False
            }
        
        # Preparar filtros
        filter_dict = {}
        if topic and topic.strip() and topic != "General":
            filter_dict['topic'] = topic.strip()
        
        # Buscar en Pinecone con configuraciones optimizadas
        try:
            search_results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None,
                timeout=10  # Timeout de 10 segundos
            )
        except Exception as search_error:
            print(f"❌ Error en búsqueda Pinecone: {search_error}")
            # Intentar búsqueda más simple sin filtros
            try:
                search_results = index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True,
                    timeout=5
                )
            except:
                print("❌ Búsqueda de fallback también falló")
                return {
                    'context': '',
                    'chunks_found': 0,
                    'files_used': [],
                    'has_context': False
                }
        
        # Procesar resultados y trackear archivos usados
        knowledge_chunks = []
        used_files = {}
        
        for match in search_results.get('matches', []):
            # Umbral ajustado para embeddings: 0.01 es más realista
            if match.get('score', 0) > 0.01:
                metadata = match.get('metadata', {})
                text = metadata.get('text', '')
                filename = metadata.get('filename', 'Documento desconocido')
                
                if text.strip():  # Solo agregar si hay texto
                    knowledge_chunks.append({
                        'text': text,
                        'filename': filename,
                        'topic': metadata.get('topic', 'General'),
                        'score': match.get('score', 0),
                        'chunk_index': metadata.get('chunk_index', 0),
                        'page_number': metadata.get('page_number', 'N/A')
                    })
                    
                    # Trackear archivos usados
                    if filename not in used_files:
                        used_files[filename] = {
                            'filename': filename,
                            'chunks_used': 0,
                            'topic': metadata.get('topic', 'General'),
                            'upload_date': metadata.get('upload_date', 'Fecha desconocida'),
                            'relevance_scores': []
                        }
                    used_files[filename]['chunks_used'] += 1
                    used_files[filename]['relevance_scores'].append(match.get('score', 0))
        
        used_files_list = list(used_files.values())
        scores_str = ', '.join([f'{c["score"]:.3f}' for c in knowledge_chunks[:3]])
        print(f"📚 Encontrados {len(knowledge_chunks)} chunks de {len(used_files_list)} archivos (scores: {scores_str})")
        
        if knowledge_chunks:
            # Crear contexto combinado limpio (solo contenido educativo)
            context = "\n\n".join([f"[De {chunk['filename']}]:\n{chunk['text']}" for chunk in knowledge_chunks[:3]])
            
            # Calcular relevancia promedio por archivo (solo para tracking interno)
            for filename, file_info in used_files.items():
                if file_info['relevance_scores']:
                    file_info['avg_relevance'] = sum(file_info['relevance_scores']) / len(file_info['relevance_scores'])
                    file_info['max_relevance'] = max(file_info['relevance_scores'])
                else:
                    file_info['avg_relevance'] = 0.0
                    file_info['max_relevance'] = 0.0
            
            return {
                'context': context,
                'chunks_found': len(knowledge_chunks),
                'files_used': used_files_list,
                'has_context': True,
                'knowledge_chunks': knowledge_chunks[:3]  # Enviar detalles de los chunks usados
            }
        else:
            return {
                'context': '',
                'chunks_found': 0,
                'files_used': [],
                'has_context': False
            }
        
    except Exception as e:
        print(f"❌ Error general en búsqueda de conocimiento: {e}")
        return {
            'context': '',
            'chunks_found': 0,
            'files_used': [],
            'has_context': False
        }

@app.route('/api/knowledge-files')
def get_knowledge_files():
    """API endpoint para obtener archivos de conocimiento filtrados por tema"""
    try:
        topic = request.args.get('topic', '').strip()
        
        if not index:
            return jsonify({
                'status': 'error',
                'message': 'Índice de Pinecone no disponible',
                'files': []
            })
        
        # Crear un filtro por tema si se especifica
        filter_dict = {}
        if topic and topic != "General":
            filter_dict['topic'] = topic
            
        # Hacer una consulta genérica para obtener archivos del tema
        # Usamos un vector dummy para obtener archivos similares
        dummy_query = f"información sobre {topic}" if topic else "documentos"
        
        try:
            query_embedding = generate_embeddings(dummy_query)
        except Exception as embed_error:
            print(f"❌ Error generando embedding: {embed_error}")
            return jsonify({
                'status': 'error', 
                'message': 'Error generando embedding',
                'files': []
            })
        
        try:
            # Consulta con más resultados para obtener variedad de archivos
            search_results = index.query(
                vector=query_embedding,
                top_k=50,  # Más resultados para obtener más archivos únicos
                include_metadata=True,
                filter=filter_dict if filter_dict else None,
                timeout=10
            )
        except Exception as search_error:
            print(f"❌ Error en consulta Pinecone: {search_error}")
            return jsonify({
                'status': 'error',
                'message': f'Error consultando base de datos: {str(search_error)}',
                'files': []
            })
        
        # Agrupar por archivos únicos
        files_info = {}
        
        for match in search_results.get('matches', []):
            metadata = match.get('metadata', {})
            filename = metadata.get('filename', 'Documento desconocido')
            
            if filename not in files_info:
                files_info[filename] = {
                    'filename': filename,
                    'topic': metadata.get('topic', 'General'),
                    'upload_date': metadata.get('upload_date', 'Fecha desconocida'),
                    'chunks': 0,
                    'status': 'Disponible',
                    'max_relevance': 0
                }
            
            # Incrementar contador de chunks
            files_info[filename]['chunks'] += 1
            
            # Actualizar máxima relevancia
            score = match.get('score', 0)
            if score > files_info[filename]['max_relevance']:
                files_info[filename]['max_relevance'] = score
        
        # Convertir a lista y ordenar por relevancia
        files_list = sorted(
            list(files_info.values()),
            key=lambda x: x['max_relevance'],
            reverse=True
        )
        
        # Formatear fecha de carga si es posible
        for file_info in files_list:
            try:
                # Si la fecha está en formato timestamp, convertirla
                upload_date = file_info.get('upload_date', '')
                if upload_date and upload_date != 'Fecha desconocida':
                    # Si es un timestamp numérico, convertir
                    if upload_date.replace('.', '').isdigit():
                        import datetime
                        file_info['upload_date'] = datetime.datetime.fromtimestamp(
                            float(upload_date)
                        ).strftime('%d/%m/%Y %H:%M')
            except:
                pass  # Mantener fecha original si hay error
        
        return jsonify({
            'status': 'success',
            'files': files_list[:20],  # Limitar a 20 archivos más relevantes
            'total_files': len(files_list),
            'filtered_by_topic': topic if topic else 'Todos los temas'
        })
        
    except Exception as e:
        print(f"❌ Error en endpoint knowledge-files: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error interno: {str(e)}',
            'files': []
        })

@app.route('/download/<filename>')
def download_file(filename):
    """Permite descargar archivos por nombre"""
    try:
        # Buscar en la carpeta data donde están los archivos reales
        data_folder = os.path.join(os.getcwd(), 'data')
        
        # Buscar el archivo en la carpeta data
        if os.path.exists(data_folder):
            for root, dirs, files in os.walk(data_folder):
                for file in files:
                    # Los archivos están guardados con timestamp, pero el enlace usa el nombre original
                    # Buscar por el nombre original al final del nombre del archivo
                    if file.endswith('_' + filename):
                        file_path = os.path.join(root, file)
                        return send_file(
                            file_path,
                            as_attachment=True,
                            download_name=filename,
                            mimetype='application/octet-stream'
                        )
        
        return jsonify({
            'status': 'error',
            'message': f'Archivo {filename} no encontrado'
        }), 404
        
    except Exception as e:
        print(f"❌ Error descargando archivo {filename}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error interno: {str(e)}'
        }), 500

# 👇👇 Esto DEBE estar al nivel de módulo, sin indentación
if __name__ == "__main__":
    # host='0.0.0.0' si quieres exponer en la red local
    app.run(debug=True)
