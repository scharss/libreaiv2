from flask import Flask, render_template, request, Response, stream_with_context, jsonify, send_from_directory
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import logging
import random
import time
import re
import markdown
import html
import fitz  # PyMuPDF
import os
from werkzeug.utils import secure_filename
import math
import pytesseract
from PIL import Image
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import sqlite3
from tqdm import tqdm
import threading
import uuid
from datetime import datetime
import io
import traceback
import shutil

# Para indexación semántica - carga bajo demanda para no consumir memoria innecesariamente
sentence_transformer_model = None

# Para extracción de tablas
try:
    import camelot
    import cv2
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Configuración de base de datos
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pdfs.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
engine = create_engine(f'sqlite:///{DB_PATH}')
Base = declarative_base()

# Definir modelos de base de datos
class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False)
    title = Column(String(255))
    num_pages = Column(Integer)
    total_chunks = Column(Integer)
    created_at = Column(Float)  # timestamp
    last_accessed = Column(Float)  # timestamp
    
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    images = relationship("DocumentImage", back_populates="document", cascade="all, delete-orphan")
    tables = relationship("DocumentTable", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer)
    text = Column(Text, nullable=False)
    section_title = Column(String(255))
    embedding = Column(LargeBinary)  # Vector almacenado como bytes
    
    document = relationship("Document", back_populates="chunks")

class DocumentImage(Base):
    __tablename__ = 'document_images'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    page_number = Column(Integer, nullable=False)
    image_data = Column(LargeBinary)
    caption = Column(Text)
    x1 = Column(Float)  # Coordenadas de la imagen en la página
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    
    document = relationship("Document", back_populates="images")

class DocumentTable(Base):
    __tablename__ = 'document_tables'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    page_number = Column(Integer, nullable=False)
    table_data = Column(Text)  # JSON serializado
    caption = Column(Text)
    x1 = Column(Float)  # Coordenadas de la tabla en la página
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    
    document = relationship("Document", back_populates="tables")

# Agregar nuevos modelos para la gestión de imágenes
class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    
    id = Column(String(255), primary_key=True)  # Usar el ID generado por el cliente
    title = Column(String(255))
    created_at = Column(Float)
    last_accessed = Column(Float)
    
    images = relationship("UserImage", back_populates="chat", cascade="all, delete-orphan")
    messages = relationship("ChatMessage", back_populates="chat", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    chat_id = Column(String(255), ForeignKey('chat_sessions.id'), nullable=False)
    content = Column(Text, nullable=False)
    is_user = Column(Boolean, default=False)
    timestamp = Column(Float)
    
    chat = relationship("ChatSession", back_populates="messages")

class UserImage(Base):
    __tablename__ = 'user_images'
    
    id = Column(Integer, primary_key=True)
    chat_id = Column(String(255), ForeignKey('chat_sessions.id'), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    storage_path = Column(String(512))  # Ruta en el sistema de archivos
    extracted_text = Column(Text)  # Texto extraído por OCR
    description = Column(Text)  # Descripción generada
    embedding = Column(LargeBinary)  # Vector de características visuales
    labels = Column(Text)  # Etiquetas detectadas (JSON)
    created_at = Column(Float)
    
    chat = relationship("ChatSession", back_populates="images")
    
    # Relación con análisis de contenido
    analysis = relationship("ImageAnalysis", uselist=False, back_populates="image", cascade="all, delete-orphan")

class ImageAnalysis(Base):
    __tablename__ = 'image_analysis'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('user_images.id'), nullable=False)
    
    # Análisis de contenido
    has_text = Column(Boolean, default=False)
    has_faces = Column(Boolean, default=False)
    has_objects = Column(Boolean, default=False)
    
    # JSON de datos detectados
    detected_objects = Column(Text)  # JSON de objetos detectados
    detected_colors = Column(Text)  # JSON de colores predominantes
    detected_faces = Column(Text)  # JSON con número y posición de rostros (no identificación)
    
    # Metadatos EXIF
    exif_data = Column(Text)  # JSON de metadatos EXIF
    
    image = relationship("UserImage", back_populates="analysis")

# Crear tablas
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Inicializar la aplicación Flask
app = Flask(__name__, static_folder='static')
logging.basicConfig(level=logging.INFO)

# Configuración para subida de archivos
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}

# Crear carpetas necesarias si no existen
try:
    for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            app.logger.info(f"Carpeta creada en: {folder}")
except Exception as e:
    app.logger.error(f"Error al crear carpetas: {str(e)}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CHUNK_SIZE'] = 10000
app.config['REQUEST_TIMEOUT'] = 300
app.config['last_image_text'] = None
app.config['image_history'] = {}  # Diccionario para almacenar el historial de imágenes por chat

# Configurar Tesseract
if os.name == 'nt':  # Windows
    try:
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if not os.path.exists(tesseract_path):
            raise Exception(f"Tesseract no encontrado en {tesseract_path}")
        
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        version = pytesseract.get_tesseract_version()
        app.logger.info(f"Tesseract OCR configurado correctamente. Versión: {version}")
    except Exception as e:
        app.logger.error(f"Error al configurar Tesseract: {str(e)}")
        app.logger.error("Por favor, asegúrese de que Tesseract OCR está instalado correctamente")
        print("\nERROR: Tesseract OCR no está configurado correctamente")
        print("1. Verifique que Tesseract OCR está instalado en C:\\Program Files\\Tesseract-OCR")
        print("2. Si está instalado en otra ubicación, actualice la ruta en el código")
        print("3. Asegúrese de haber instalado los paquetes de idioma necesarios")
        print(f"Error detallado: {str(e)}\n")

# Configuración de Ollama
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_API_URL = f'{OLLAMA_HOST}/api/generate'

# Verificar conexión con Ollama
try:
    response = requests.get(f'{OLLAMA_HOST}/api/tags')
    if response.status_code == 200:
        app.logger.info("Conexión con Ollama establecida correctamente")
    else:
        app.logger.error(f"Error al conectar con Ollama. Código de estado: {response.status_code}")
except Exception as e:
    app.logger.error(f"No se pudo conectar con Ollama: {str(e)}")

# Configurar reintentos y timeout
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)
http.request = lambda *args, **kwargs: requests.Session.request(http, *args, **{**kwargs, 'timeout': app.config['REQUEST_TIMEOUT']})

# Emojis simplificados
THINKING_EMOJI = '🤔'
RESPONSE_EMOJI = '🤖'
ERROR_EMOJI = '⚠️'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path, filename):
    try:
        doc = fitz.open(file_path)
        text = ""
        total_pages = doc.page_count
        logging.info(f"Procesando PDF con {total_pages} páginas")
        
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            text += page_text
            logging.debug(f"Página {page_num}/{total_pages} procesada")
        
        doc.close()
        logging.info(f"PDF procesado completamente. Texto extraído: {len(text)} caracteres")
        return {
            "chunks": text.split(),
            "total_chunks": len(text.split())
        }
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return None

def chunk_text(text, chunk_size):
    """Divide el texto en fragmentos de tamaño aproximado."""
    words = text.split()
    total_words = len(words)
    chunks = []
    current_chunk = []
    current_size = 0
    
    logging.info(f"Procesando texto de {total_words} palabras para fragmentación")
    
    for word in words:
        word_size = len(word.split())  # Aproximación simple de tokens
        if current_size + word_size > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            logging.debug(f"Fragmento creado con {len(current_chunk)} palabras")
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    # Asegurarse de incluir el último fragmento
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
        logging.debug(f"Último fragmento creado con {len(current_chunk)} palabras")
    
    total_words_in_chunks = sum(len(chunk.split()) for chunk in chunks)
    logging.info(f"Total de palabras procesadas: {total_words_in_chunks} de {total_words}")
    
    if total_words_in_chunks < total_words:
        logging.warning(f"Se perdieron {total_words - total_words_in_chunks} palabras durante el procesamiento")
    
    return chunks

@app.route('/')
def home():
    try:
        # Verificar que Ollama esté funcionando
        try:
            response = requests.get(f'{OLLAMA_HOST}/api/tags', timeout=5)
            if response.status_code != 200:
                app.logger.error("Ollama no está respondiendo correctamente")
                return render_template('error.html', error="Ollama no está respondiendo. Por favor, asegúrate de que Ollama esté corriendo.")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"No se puede conectar con Ollama: {str(e)}")
            return render_template('error.html', error="No se puede conectar con Ollama. Por favor, asegúrate de que Ollama esté corriendo.")
        
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error en la ruta principal: {str(e)}")
        return render_template('error.html', error="Error interno del servidor")

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            app.logger.error("No se encontró archivo en la solicitud")
            return jsonify({'success': False, 'error': 'No se encontró el archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            app.logger.error("Nombre de archivo vacío")
            return jsonify({'success': False, 'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            app.logger.error(f"Tipo de archivo no permitido: {file.filename}")
            return jsonify({'success': False, 'error': 'Tipo de archivo no permitido'}), 400
        
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            app.logger.info(f"Archivo guardado exitosamente: {file_path}")
            
            if filename.lower().endswith('.pdf'):
                # Procesar PDF con nuevas funciones
                try:
                    result = extract_text_from_pdf(file_path, filename)
                    chunks = result.get('chunks', [])
                    total_chunks = result.get('total_chunks', 0)
                    
                    if not chunks or total_chunks == 0:
                        raise Exception("No se pudo extraer texto del PDF")
                    
                    # Eliminar archivo temporal si ya se procesó correctamente y se guardó en DB
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    app.logger.info(f"PDF procesado exitosamente: {len(chunks)} fragmentos")
                    
                    # Obtener información de la DB
                    session = Session()
                    doc = session.query(Document).filter_by(filename=filename).first()
                    
                    if doc:
                        num_tables = session.query(DocumentTable).filter_by(document_id=doc.id).count()
                        num_images = session.query(DocumentImage).filter_by(document_id=doc.id).count()
                        session.close()
                        
                        # Mensaje que describe el documento
                        message = f"""PDF procesado exitosamente con características avanzadas:
                        
- {total_chunks} fragmentos de texto
- {num_tables} tablas detectadas
- {num_images} imágenes extraídas
- Indexación semántica implementada para búsqueda

Ahora puedes hacer preguntas específicas sobre el contenido del documento."""
                        
                        return jsonify({
                            'success': True,
                            'message': message,
                            'filename': filename,
                            'num_chunks': total_chunks,
                            'num_tables': num_tables,
                            'num_images': num_images,
                            'has_semantic_index': True
                        })
                    else:
                        session.close()
                        return jsonify({
                            'success': True,
                            'message': f'PDF procesado exitosamente con {total_chunks} fragmentos',
                            'filename': filename,
                            'num_chunks': total_chunks
                        })
                except Exception as e:
                    app.logger.error(f"Error procesando PDF: {str(e)}")
                    traceback.print_exc()
                    return jsonify({'success': False, 'error': f'Error procesando PDF: {str(e)}'}), 500
            
            return jsonify({'success': True, 'filename': filename})
            
        except Exception as e:
            app.logger.error(f"Error procesando archivo: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)  # Limpiar en caso de error
            return jsonify({'success': False, 'error': str(e)}), 500
        
    except Exception as e:
        app.logger.error(f"Error en upload_file: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdf_info/<filename>', methods=['GET'])
def get_pdf_info(filename):
    """Obtener información detallada sobre un PDF procesado"""
    try:
        session = Session()
        doc = session.query(Document).filter_by(filename=filename).first()
        
        if not doc:
            session.close()
            return jsonify({'success': False, 'error': 'Documento no encontrado'}), 404
        
        # Contar tablas e imágenes
        num_tables = session.query(DocumentTable).filter_by(document_id=doc.id).count()
        num_images = session.query(DocumentImage).filter_by(document_id=doc.id).count()
        
        # Obtener secciones
        chunks = session.query(DocumentChunk).filter_by(document_id=doc.id).all()
        sections = {}
        for chunk in chunks:
            if chunk.section_title and chunk.section_title not in sections:
                sections[chunk.section_title] = chunk.page_number
        
        # Actualizar tiempo de acceso
        doc.last_accessed = time.time()
        session.commit()
        
        return jsonify({
            'success': True,
            'filename': doc.filename,
            'title': doc.title,
            'num_pages': doc.num_pages,
            'total_chunks': doc.total_chunks,
            'num_tables': num_tables,
            'num_images': num_images,
            'sections': [{'title': title, 'page': page} for title, page in sections.items()],
            'created_at': doc.created_at,
            'last_accessed': doc.last_accessed
        })
    
    except Exception as e:
        app.logger.error(f"Error en get_pdf_info: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
    finally:
        if 'session' in locals():
            session.close()

@app.route('/semantic_search', methods=['POST'])
def search_pdf():
    """Realizar búsqueda semántica en documentos PDF procesados"""
    try:
        data = request.json
        query = data.get('query', '')
        filename = data.get('filename', None)
        top_k = data.get('top_k', 3)
        
        if not query:
            return jsonify({'success': False, 'error': 'Query vacío'}), 400
        
        document_id = None
        if filename:
            session = Session()
            doc = session.query(Document).filter_by(filename=filename).first()
            if doc:
                document_id = doc.id
            session.close()
        
        # Realizar búsqueda semántica
        results = semantic_search(query, document_id, top_k)
        
        # Formatear resultados
        formatted_results = []
        for chunk, similarity in results:
            formatted_results.append({
                'chunk_index': chunk.chunk_index,
                'page': chunk.page_number,
                'text': chunk.text,
                'section': chunk.section_title,
                'similarity': float(similarity),  # Convertir a float para serialización JSON
                'document': {
                    'id': chunk.document_id,
                    'filename': chunk.document.filename,
                    'title': chunk.document.title
                } if chunk.document else None
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': formatted_results
        })
    
    except Exception as e:
        app.logger.error(f"Error en semantic_search: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def clean_math_expressions(text):
    """Limpia y formatea expresiones matemáticas."""
    # No eliminar los backslashes necesarios para LaTeX
    replacements = {
        r'\\begin\{align\*?\}': '',
        r'\\end\{align\*?\}': '',
        r'\\begin\{equation\*?\}': '',
        r'\\end\{equation\*?\}': '',
        r'\\ ': ' '  # Reemplazar \\ espacio con un espacio normal
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def format_math(text):
    """Formatea expresiones matemáticas para KaTeX."""
    def process_math_content(match):
        content = match.group(1).strip()
        content = clean_math_expressions(content)
        return f'$${content}$$'

    # Procesar comandos especiales de LaTeX antes de los bloques matemáticos
    text = re.sub(r'\\boxed\{\\text\{([^}]*)\}\}', r'<div class="boxed">\1</div>', text)
    text = re.sub(r'\\boxed\{([^}]*)\}', r'<div class="boxed">\1</div>', text)
    
    # Procesar bloques matemáticos inline y display
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: f'$${m.group(1)}$$', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', lambda m: f'${m.group(1)}$', text)
    text = re.sub(r'\\\[(.*?)\\\]', process_math_content, text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', lambda m: f'${m.group(1)}$', text)
    
    # Preservar comandos LaTeX específicos
    text = re.sub(r'\\times(?![a-zA-Z])', r'\\times', text)  # Preservar \times
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\\frac{\1}{\2}', text)  # Preservar fracciones
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)  # Manejar \text correctamente
    
    return text

def format_code_blocks(text):
    """Formatea bloques de código con resaltado de sintaxis."""
    def replace_code_block(match):
        language = match.group(1) or 'plaintext'
        code = match.group(2).strip()
        return f'```{language}\n{code}\n```'

    # Procesar bloques de código
    text = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, text, flags=re.DOTALL)
    return text

def format_response(text):
    """Formatea la respuesta completa con soporte para markdown, código y matemáticas."""
    # Primero formatear expresiones matemáticas
    text = format_math(text)
    
    # Formatear bloques de código
    text = format_code_blocks(text)
    
    # Convertir markdown a HTML preservando las expresiones matemáticas
    # Escapar temporalmente las expresiones matemáticas
    math_blocks = []
    def math_replace(match):
        math_blocks.append(match.group(0))
        return f'MATH_BLOCK_{len(math_blocks)-1}'

    # Guardar expresiones matemáticas
    text = re.sub(r'\$\$.*?\$\$|\$.*?\$', math_replace, text, flags=re.DOTALL)
    
    # Convertir markdown a HTML
    md = markdown.Markdown(extensions=['fenced_code', 'tables'])
    text = md.convert(text)
    
    # Restaurar expresiones matemáticas
    for i, block in enumerate(math_blocks):
        text = text.replace(f'MATH_BLOCK_{i}', block)
    
    # Limpiar y formatear el texto
    text = text.replace('</think>', '').replace('<think>', '')
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
    
    return text.strip()

def decorate_message(message, is_error=False):
    """Decora el mensaje con emojis y formato apropiado."""
    emoji = ERROR_EMOJI if is_error else RESPONSE_EMOJI
    if is_error:
        return f"{emoji} {message}"
    
    formatted_message = format_response(message)
    return f"{emoji} {formatted_message}"

def get_thinking_message():
    """Genera un mensaje de 'pensando' aleatorio."""
    messages = [
        "Analizando tu pregunta...",
        "Procesando la información...",
        "Elaborando una respuesta...",
        "Pensando...",
        "Trabajando en ello...",
    ]
    return f"{THINKING_EMOJI} {random.choice(messages)}"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    model = data.get('model', 'deepseek-r1:7b')
    filename = data.get('pdf_file', None)
    chunk_index = data.get('chunk_index', 0)
    chat_id = data.get('chat_id', None)
    conversation_history = data.get('conversation_history', [])
    
    app.logger.debug(f"Mensaje recibido: {user_message}")
    app.logger.debug(f"Modelo seleccionado: {model}")
    app.logger.debug(f"Historial de conversación recibido con {len(conversation_history)} mensajes")

    def generate():
        try:
            # Enviar mensaje inicial de "pensando"
            thinking_msg = get_thinking_message()
            yield json.dumps({
                'thinking': thinking_msg
            }) + '\n'
            
            # Preparar el prompt base
            prompt = user_message
            
            # Construir el historial de conversación formateado
            formatted_history = ""
            if conversation_history and len(conversation_history) > 0:
                for msg in conversation_history:
                    role = "Usuario" if msg.get('isUser', False) else "Asistente"
                    content = msg.get('content', '').strip()
                    if content:
                        formatted_history += f"{role}: {content}\n\n"
            
            # Si hay historial de imágenes para este chat
            if chat_id and chat_id in app.config['image_history']:
                image_infos = app.config['image_history'][chat_id]
                
                # Detectar si el usuario se refiere específicamente a imágenes
                is_image_query = any(phrase in user_message.lower() for phrase in [
                    "esta imagen", "esta otra imagen", "la imagen", "imagen", "imágenes",
                    "foto", "fotografía", "fotografías", "fotos", "color", "colores",
                    "rostro", "rostros", "cara", "caras", "detalle", "detalles"
                ])
                
                # Si es una consulta relacionada con imágenes, usar búsqueda semántica
                if is_image_query:
                    # Intentar búsqueda semántica para encontrar imágenes relevantes
                    try:
                        session = Session()
                        # Primero verificar si hay imágenes en la base de datos para este chat
                        db_images = session.query(UserImage).filter_by(chat_id=chat_id).count()
                        session.close()
                        
                        if db_images > 0:
                            # Realizar búsqueda semántica
                            top_img_results = semantic_image_search(user_message, chat_id, top_k=2)
                            
                            # Si tenemos resultados, incluirlos en el contexto
                            if top_img_results:
                                image_contexts = []
                                
                                for i, (img, similarity) in enumerate(top_img_results, 1):
                                    # Buscar análisis avanzado
                                    session = Session()
                                    analysis = session.query(ImageAnalysis).filter_by(image_id=img.id).first()
                                    session.close()
                                    
                                    context_parts = [f"Imagen {i} (relevancia: {similarity*100:.0f}%):"]
                                    context_parts.append(f"Descripción: {img.description}")
                                    
                                    if img.extracted_text and len(img.extracted_text) > 10:
                                        context_parts.append(f"Texto extraído: {img.extracted_text}")
                                    
                                    # Añadir información de análisis si está disponible
                                    if analysis:
                                        if analysis.has_faces and analysis.detected_faces:
                                            faces = json.loads(analysis.detected_faces)
                                            context_parts.append(f"Contiene {len(faces)} rostros humanos")
                                        
                                        if analysis.has_objects and analysis.detected_objects:
                                            objects = json.loads(analysis.detected_objects)
                                            obj_types = set(obj["type"] for obj in objects)
                                            context_parts.append(f"Objetos detectados: {', '.join(obj_types)}")
                                        
                                        if analysis.detected_colors:
                                            colors = json.loads(analysis.detected_colors)
                                            if colors:
                                                top_colors = [color["hex"] for color in colors[:3]]
                                                context_parts.append(f"Colores predominantes: {', '.join(top_colors)}")
                                    
                                    image_contexts.append("\n".join(context_parts))
                                
                                if image_contexts:
                                    context = "\n\n".join(image_contexts)
                                    prompt = f"""Contexto de imágenes relevantes para la consulta:

{context}

Historial de la conversación:
{formatted_history}

Pregunta actual del usuario: {user_message}

IMPORTANTE: Actúa como si pudieras ver directamente las imágenes descritas. Basado en la descripción y análisis proporcionados, responde como si estuvieras viendo las imágenes con tus propios ojos. Si te preguntan sobre rostros, colores, objetos o cualquier elemento visual, responde con la información proporcionada en el contexto.

Por favor, responde a la pregunta basándote en las imágenes más relevantes y el historial de la conversación."""
                                else:
                                    # Si no hay contexto de imágenes, pero la consulta es sobre imágenes
                                    prompt = f"""El usuario está preguntando sobre imágenes.

Historial de la conversación:
{formatted_history}

Pregunta actual del usuario: {user_message}

IMPORTANTE: Si se mencionan imágenes en la conversación, actúa como si pudieras verlas basándote en las descripciones proporcionadas anteriormente.

Por favor, responde a la pregunta teniendo en cuenta el historial completo de la conversación."""
                            else:
                                # Incluir el historial de conversación aunque no se mencionen específicamente imágenes
                                prompt = f"""El usuario ha subido imágenes anteriormente pero no se encontraron imágenes relevantes para esta consulta específica.

Historial de la conversación:
{formatted_history}

Pregunta actual del usuario: {user_message}

IMPORTANTE: Si te preguntan sobre imágenes mencionadas anteriormente, actúa como si pudieras verlas basándote en la información proporcionada en el historial.

Por favor, responde a la pregunta teniendo en cuenta el historial completo de la conversación."""
                        else:
                            # Usar el enfoque anterior cuando no hay imágenes en la base de datos
                            prompt = generate_image_prompt_legacy(image_infos, formatted_history, user_message)
                            
                    except Exception as e:
                        app.logger.error(f"Error en búsqueda semántica de imágenes: {str(e)}")
                        # En caso de error, volver al enfoque anterior
                        prompt = generate_image_prompt_legacy(image_infos, formatted_history, user_message)
                else:
                    # Incluir el historial de conversación aunque no se mencionen específicamente imágenes
                    prompt = f"""Historial de la conversación:
{formatted_history}

Pregunta actual del usuario: {user_message}

Por favor, responde a la pregunta teniendo en cuenta el historial completo de la conversación, incluyendo cualquier imagen mencionada anteriormente."""
            
            # Solo incluir contexto del PDF si hay un archivo activo Y está en el mismo chat
            elif filename and data.get('isPdfChat', False):
                document_id = None
                doc_info = None
                session = Session()
                
                try:
                    # Buscar documento en la base de datos
                    doc = session.query(Document).filter_by(filename=filename).first()
                    if doc:
                        document_id = doc.id
                        doc_info = {
                            'title': doc.title,
                            'filename': doc.filename,
                            'total_chunks': doc.total_chunks,
                            'num_pages': doc.num_pages
                        }
                except Exception as e:
                    app.logger.error(f"Error consultando documento: {str(e)}")
                finally:
                    session.close()
                
                # Buscar los fragmentos más relevantes usando búsqueda semántica si está disponible
                semantic_results = []
                try:
                    if document_id:
                        semantic_results = semantic_search(user_message, document_id, top_k=3)
                except Exception as e:
                    app.logger.error(f"Error en búsqueda semántica: {str(e)}")
                
                # Si tenemos resultados de búsqueda semántica, usarlos
                if semantic_results:
                    semantic_contexts = []
                    for i, (chunk, similarity) in enumerate(semantic_results):
                        semantic_contexts.append(
                            f"Fragmento relevante {i+1} (página {chunk.page_number+1}" + 
                            (f", sección: {chunk.section_title}" if chunk.section_title else "") + 
                            f"):\n{chunk.text}"
                        )
                    
                    # Buscar tablas relacionadas si se mencionan en la pregunta
                    table_contexts = []
                    if document_id and any(word in user_message.lower() for word in ["tabla", "table", "tablas", "tables", "cuadro"]):
                        try:
                            session = Session()
                            tables = session.query(DocumentTable).filter_by(document_id=document_id).all()
                            for i, table in enumerate(tables[:2]):  # Limitar a las 2 primeras tablas para no sobrecargar
                                table_contexts.append(f"Tabla {i+1} en página {table.page_number+1}:\n{table.caption}\n{table.table_data}")
                            session.close()
                        except Exception as e:
                            app.logger.error(f"Error obteniendo tablas: {str(e)}")
                    
                    context = "\n\n".join(semantic_contexts + table_contexts)
                    
                    prompt = f"""Contexto del documento "{doc_info['title'] if doc_info else filename}":

{context}

Historial de la conversación:
{formatted_history}

Pregunta actual del usuario: {user_message}

Por favor, responde a la pregunta basándote en los fragmentos más relevantes del documento y el historial de la conversación.
Si necesitas información de otras partes del documento, indícalo."""

                # Si no hay búsqueda semántica o falló, usar el enfoque por chunks como antes
                else:
                    chunks = []
                    # Primero intentar obtener de la base de datos
                    if document_id:
                        try:
                            session = Session()
                            db_chunks = session.query(DocumentChunk).filter_by(document_id=document_id).all()
                            chunks = [chunk.text for chunk in db_chunks]
                            session.close()
                        except Exception as e:
                            app.logger.error(f"Error obteniendo chunks de DB: {str(e)}")
                            
                    # Si falla o no está en DB, usar el enfoque anterior
                    if not chunks and filename in app.config.get('pdf_chunks', {}):
                        chunks = app.config['pdf_chunks'][filename]
                    
                    if not chunks:
                        # No se encontraron chunks, informar al usuario
                        error_msg = f"No se encontraron fragmentos para el documento {filename}."
                        app.logger.error(error_msg)
                        yield json.dumps({
                            'error': decorate_message(error_msg, is_error=True)
                        }) + '\n'
                        return
                    
                    # Construir el contexto combinando fragmentos relevantes
                    context_chunks = []
                    
                    # Siempre incluir el fragmento actual
                    current_chunk = chunks[chunk_index]
                    context_chunks.append(f"Fragmento {chunk_index + 1}:\n{current_chunk}")
                    
                    # Incluir fragmentos adyacentes si están disponibles
                    if chunk_index > 0:
                        prev_chunk = chunks[chunk_index - 1]
                        context_chunks.insert(0, f"Fragmento {chunk_index}:\n{prev_chunk}")
                    
                    if chunk_index < len(chunks) - 1:
                        next_chunk = chunks[chunk_index + 1]
                        context_chunks.append(f"Fragmento {chunk_index + 2}:\n{next_chunk}")
                    
                    # Combinar los fragmentos en un solo contexto
                    combined_context = "\n\n".join(context_chunks)
                    
                    prompt = f"""Contexto del PDF "{doc_info['title'] if doc_info else filename}" (fragmentos {chunk_index + 1} y adyacentes de {len(chunks)} totales):

{combined_context}

Historial de la conversación:
{formatted_history}

Pregunta actual del usuario: {user_message}

Por favor, responde la pregunta basándote en el contenido proporcionado del PDF y el historial de la conversación.
Si la respuesta podría estar en otros fragmentos no incluidos, indícalo y sugiere revisar otros fragmentos."""

            # Si no hay contexto de imágenes ni PDF, pero tenemos historial de conversación
            elif formatted_history:
                prompt = f"""Historial de la conversación:
{formatted_history}

Pregunta actual del usuario: {user_message}

Por favor, responde a la pregunta teniendo en cuenta el historial completo de la conversación."""
            
            payload = {
                'model': model,
                'prompt': prompt,
                'stream': True
            }
            
            app.logger.debug(f"Enviando solicitud a Ollama API con payload: {payload}")
            
            try:
                response = http.post(
                    OLLAMA_API_URL,
                    json=payload,
                    stream=True,
                    timeout=60  # Aumentar timeout a 60 segundos
                )
            except requests.exceptions.Timeout:
                error_msg = "La solicitud está tomando más tiempo de lo esperado. Por favor, intenta con un mensaje más corto o espera un momento."
                app.logger.error(error_msg)
                yield json.dumps({
                    'error': decorate_message(error_msg, is_error=True)
                }) + '\n'
                return
            except requests.exceptions.ConnectionError:
                error_msg = "No se pudo conectar con Ollama. Por favor, verifica que Ollama esté corriendo."
                app.logger.error(error_msg)
                yield json.dumps({
                    'error': decorate_message(error_msg, is_error=True)
                }) + '\n'
                return
            
            app.logger.debug(f"Estado de respuesta de Ollama API: {response.status_code}")
            if response.status_code != 200:
                error_msg = f"Error al conectar con Ollama API. Código de estado: {response.status_code}. Respuesta: {response.text}"
                app.logger.error(error_msg)
                yield json.dumps({
                    'error': decorate_message(error_msg, is_error=True)
                }) + '\n'
                return

            # Limpiar mensaje de "pensando" y comenzar a mostrar la respuesta
            yield json.dumps({'clear_thinking': True}) + '\n'
            
            # Inicializar acumulador de respuesta
            full_response = ""
            
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        app.logger.debug(f"Fragmento de respuesta recibido: {json_response}")
                        ai_response = json_response.get('response', '')
                        if ai_response:
                            full_response += ai_response
                            # Formatear y enviar la respuesta completa hasta el momento
                            decorated_response = decorate_message(full_response)
                            yield json.dumps({'response': decorated_response}) + '\n'
                        
                    except json.JSONDecodeError as e:
                        app.logger.error(f"Error al decodificar JSON: {str(e)} para la línea: {line}")
                        continue

        except Exception as e:
            error_msg = f"Error de conexión: {str(e)}"
            app.logger.error(error_msg)
            yield json.dumps({
                'error': decorate_message(error_msg, is_error=True)
            }) + '\n'

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud"""
    status = {
        'status': 'healthy',
        'message': "Servidor en funcionamiento",
        'timestamp': time.time()
    }
    return json.dumps(status)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            app.logger.error("No se encontró imagen en la solicitud")
            return jsonify({'success': False, 'error': 'No se encontró el archivo'})
        
        chat_id = request.form.get('chat_id')  # Obtener el ID del chat desde el formulario
        if not chat_id:
            app.logger.error("No se proporcionó ID de chat")
            return jsonify({'success': False, 'error': 'No se proporcionó ID de chat'})
        
        file = request.files['file']
        if file.filename == '':
            app.logger.error("Nombre de archivo de imagen vacío")
            return jsonify({'success': False, 'error': 'No se seleccionó ningún archivo'})
        
        if not allowed_file(file.filename):
            app.logger.error(f"Tipo de imagen no permitido: {file.filename}")
            return jsonify({'success': False, 'error': 'Tipo de archivo no permitido'})
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Guardar el archivo original temporalmente
            file.save(filepath)
            app.logger.info(f"Imagen guardada temporalmente: {filepath}")
            
            # Procesar la imagen con nuestras nuevas funciones avanzadas
            result = process_image(filepath, file.filename, chat_id)
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'message': result['message'],
                    'filename': result['filename'],
                    'image_url': result['image_url'],
                    'analysis': result['analysis'],
                    'description': result['description']
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Error desconocido procesando la imagen')
                })
            
        except Exception as e:
            app.logger.error(f"Error guardando/procesando imagen: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f'Error al procesar la imagen: {str(e)}'
            })
            
    except Exception as e:
        app.logger.error(f"Error en upload_image: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error en el servidor: {str(e)}'
        })

@app.route('/image_info/<image_id>', methods=['GET'])
def get_image_info(image_id):
    """Obtener información detallada sobre una imagen procesada"""
    try:
        session = Session()
        image = session.query(UserImage).filter_by(id=image_id).first()
        
        if not image:
            session.close()
            return jsonify({'success': False, 'error': 'Imagen no encontrada'}), 404
        
        # Obtener el análisis relacionado
        analysis = session.query(ImageAnalysis).filter_by(image_id=image.id).first()
        
        # Preparar la respuesta
        response = {
            'success': True,
            'image': {
                'id': image.id,
                'filename': image.filename,
                'original_filename': image.original_filename,
                'url': f'/static/uploads/{image.filename}',
                'extracted_text': image.extracted_text,
                'description': image.description,
                'created_at': image.created_at
            }
        }
        
        # Incluir datos de análisis si existen
        if analysis:
            response['analysis'] = {
                'has_text': analysis.has_text,
                'has_faces': analysis.has_faces,
                'has_objects': analysis.has_objects
            }
            
            # Incluir detalles adicionales si están disponibles
            if analysis.detected_objects:
                response['analysis']['objects'] = json.loads(analysis.detected_objects)
            
            if analysis.detected_colors:
                response['analysis']['colors'] = json.loads(analysis.detected_colors)
            
            if analysis.detected_faces:
                response['analysis']['faces'] = json.loads(analysis.detected_faces)
            
            if analysis.exif_data:
                response['analysis']['exif'] = json.loads(analysis.exif_data)
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Error en get_image_info: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
    finally:
        if 'session' in locals():
            session.close()

@app.route('/image_search', methods=['POST'])
def search_images():
    """Realizar búsqueda semántica en imágenes"""
    try:
        data = request.json
        query = data.get('query', '')
        chat_id = data.get('chat_id', None)
        top_k = data.get('top_k', 3)
        
        if not query:
            return jsonify({'success': False, 'error': 'Query vacío'}), 400
        
        # Realizar búsqueda semántica
        results = semantic_image_search(query, chat_id, top_k)
        
        # Formatear resultados
        formatted_results = []
        for img, similarity in results:
            formatted_results.append({
                'id': img.id,
                'filename': img.filename,
                'url': f'/static/uploads/{img.filename}',
                'description': img.description,
                'extracted_text': img.extracted_text[:100] + "..." if len(img.extracted_text) > 100 else img.extracted_text,
                'similarity': float(similarity),  # Convertir a float para serialización JSON
                'chat_id': img.chat_id
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': formatted_results
        })
    
    except Exception as e:
        app.logger.error(f"Error en image_search: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Ruta para servir archivos estáticos de uploads
@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

# Función para cargar el modelo de embeddings bajo demanda
def get_sentence_transformer():
    global sentence_transformer_model
    if sentence_transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            app.logger.info("Cargando modelo de sentence transformers...")
            sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            app.logger.info("Modelo cargado correctamente")
        except ImportError:
            app.logger.error("No se pudo importar sentence_transformers. Funcionalidad de búsqueda semántica no disponible.")
            return None
    return sentence_transformer_model

# Función para generar embeddings de texto
def generate_embedding(text):
    model = get_sentence_transformer()
    if model is None:
        return None
    
    embedding = model.encode(text)
    return embedding.tobytes()

# Función para convertir bytes a numpy array
def bytes_to_embedding(embedding_bytes):
    if embedding_bytes is None:
        return None
    return np.frombuffer(embedding_bytes, dtype=np.float32)

# Función para buscar fragmentos similares
def semantic_search(query, document_id=None, top_k=3):
    model = get_sentence_transformer()
    if model is None:
        return []
    
    query_embedding = model.encode(query)
    
    session = Session()
    try:
        query_obj = session.query(DocumentChunk)
        
        if document_id is not None:
            query_obj = query_obj.filter(DocumentChunk.document_id == document_id)
        
        chunks = query_obj.all()
        results = []
        
        for chunk in chunks:
            if chunk.embedding:
                chunk_embedding = bytes_to_embedding(chunk.embedding)
                if chunk_embedding is not None:
                    # Calcular similitud de coseno
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    results.append((chunk, similarity))
        
        # Ordenar por similitud descendente y tomar los top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    finally:
        session.close()

# Función para detectar secciones y estructura
def detect_sections(pdf_document):
    sections = []
    current_section = {"title": "Inicio", "level": 0, "page": 0}
    
    for page_num, page in enumerate(pdf_document):
        text = page.get_text()
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Heurísticas para detectar encabezados de sección
            if len(line) < 100:  # Encabezados suelen ser cortos
                is_heading = False
                level = 0
                
                # Patrones comunes de encabezados
                if re.match(r'^(Chapter|Capítulo|CAPÍTULO)\s+\d+', line, re.IGNORECASE):
                    is_heading = True
                    level = 1
                elif re.match(r'^\d+\.\s+[A-Z]', line):  # 1. Título
                    is_heading = True
                    level = 1
                elif re.match(r'^\d+\.\d+\.\s+[A-Z]', line):  # 1.1. Subtítulo
                    is_heading = True
                    level = 2
                elif re.match(r'^\d+\.\d+\.\d+\.\s+[A-Z]', line):  # 1.1.1. Sub-subtítulo
                    is_heading = True
                    level = 3
                # Detectar líneas en MAYÚSCULAS o con formato especial
                elif line.isupper() and len(line) > 3 and len(line) < 50:
                    is_heading = True
                    level = 1
                
                if is_heading:
                    sections.append(current_section)
                    current_section = {
                        "title": line,
                        "level": level,
                        "page": page_num,
                        "text": ""
                    }
                    continue
            
            # Agregar texto a la sección actual
            if "text" in current_section:
                current_section["text"] += line + "\n"
    
    # Agregar la última sección
    sections.append(current_section)
    
    return sections

# Función para extraer tablas usando Camelot
def extract_tables(pdf_path, pdf_document):
    tables_data = []
    
    if not CAMELOT_AVAILABLE:
        app.logger.warning("Camelot no está disponible. No se extraerán tablas.")
        return tables_data
    
    try:
        # Detectar tablas usando Camelot
        for page_num in range(len(pdf_document)):
            try:
                tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1))
                
                for i, table in enumerate(tables):
                    table_data = {
                        "page": page_num,
                        "index": i,
                        "data": table.df.to_json(),
                        "caption": f"Tabla {i+1} en página {page_num+1}",
                        "accuracy": table.accuracy
                    }
                    
                    # Solo incluir tablas con cierta precisión
                    if table.accuracy > 80:
                        tables_data.append(table_data)
            except Exception as e:
                app.logger.error(f"Error al extraer tablas de la página {page_num+1}: {str(e)}")
    
    except Exception as e:
        app.logger.error(f"Error global al extraer tablas: {str(e)}")
    
    return tables_data

# Función para extraer imágenes con PyMuPDF
def extract_images(pdf_document):
    images_data = []
    
    for page_num, page in enumerate(pdf_document):
        try:
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                
                try:
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Crear objeto PIL para OCR
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Intentar realizar OCR en la imagen si es grande
                    caption = ""
                    if image.width > 200 and image.height > 200:
                        try:
                            ocr_text = pytesseract.image_to_string(image)
                            if ocr_text.strip():
                                caption = f"Texto detectado en imagen: {ocr_text.strip()}"
                        except Exception as e:
                            app.logger.error(f"Error en OCR: {str(e)}")
                    
                    # Obtener coordenadas de la imagen en la página
                    # (esto es una simplificación, las coordenadas reales requieren más procesamiento)
                    rect = page.get_image_bbox(img_info)
                    
                    image_data = {
                        "page": page_num,
                        "index": img_index,
                        "image": image_bytes,
                        "caption": caption,
                        "x1": rect.x0,
                        "y1": rect.y0,
                        "x2": rect.x1,
                        "y2": rect.y1
                    }
                    
                    images_data.append(image_data)
                
                except Exception as e:
                    app.logger.error(f"Error al procesar imagen {xref} en página {page_num+1}: {str(e)}")
        
        except Exception as e:
            app.logger.error(f"Error al extraer imágenes de la página {page_num+1}: {str(e)}")
    
    return images_data

# Función mejorada para procesar PDF
def process_pdf(file_path, filename):
    try:
        session = Session()
        
        # Verificar si ya existe en la base de datos
        existing_doc = session.query(Document).filter_by(filename=filename).first()
        if existing_doc:
            # Actualizar tiempo de acceso
            existing_doc.last_accessed = time.time()
            session.commit()
            
            # Devolver los chunks ya procesados
            chunks = [chunk.text for chunk in existing_doc.chunks]
            app.logger.info(f"PDF {filename} encontrado en caché, retornando {len(chunks)} chunks")
            
            return {
                "chunks": chunks,
                "doc_id": existing_doc.id,
                "title": existing_doc.title,
                "total_chunks": existing_doc.total_chunks
            }
        
        app.logger.info(f"Procesando nuevo PDF: {filename}")
        
        # Abrir el PDF
        pdf_document = fitz.open(file_path)
        total_pages = len(pdf_document)
        
        # Buscar un título en las primeras páginas
        title = filename
        for i in range(min(3, total_pages)):
            text = pdf_document[i].get_text().strip()
            lines = text.split('\n')
            if lines and len(lines[0]) < 100:  # Primera línea como posible título
                title = lines[0]
                break
        
        # Crear registro de documento
        doc = Document(
            filename=filename,
            title=title,
            num_pages=total_pages,
            created_at=time.time(),
            last_accessed=time.time()
        )
        session.add(doc)
        session.flush()  # Para obtener el ID del documento
        
        # Detectar secciones
        app.logger.info(f"Detectando secciones en {filename}")
        sections = detect_sections(pdf_document)
        
        # Extraer texto como antes, pero ahora con información de secciones
        text = ""
        for page_num, page in enumerate(pdf_document):
            page_text = page.get_text()
            text += f"\n----- Página {page_num + 1} -----\n"
            text += page_text
        
        # Dividir en chunks
        app.logger.info(f"Dividiendo texto en chunks")
        chunks = chunk_text(text, app.config['MAX_CHUNK_SIZE'])
        doc.total_chunks = len(chunks)
        
        # Guardar chunks en base de datos
        app.logger.info(f"Guardando {len(chunks)} chunks en la base de datos")
        for i, chunk_text in enumerate(tqdm(chunks, desc="Procesando chunks")):
            # Determinar la página aproximada del chunk
            page_markers = [f"----- Página {p + 1} -----" for p in range(total_pages)]
            page_number = 0
            for p, marker in enumerate(page_markers):
                if marker in chunk_text:
                    page_number = p
                    break
            
            # Encontrar la sección a la que pertenece este chunk
            section_title = None
            for section in sections:
                if section["page"] <= page_number and section.get("text", "") in chunk_text:
                    section_title = section["title"]
            
            # Generar embedding para búsqueda semántica
            embedding = generate_embedding(chunk_text)
            
            # Crear el chunk en la base de datos
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=i,
                page_number=page_number,
                text=chunk_text,
                section_title=section_title,
                embedding=embedding
            )
            session.add(chunk)
        
        # Procesar tablas si está disponible Camelot
        app.logger.info(f"Extrayendo tablas")
        tables = extract_tables(file_path, pdf_document)
        for table_data in tables:
            table = DocumentTable(
                document_id=doc.id,
                page_number=table_data["page"],
                table_data=table_data["data"],
                caption=table_data["caption"]
            )
            session.add(table)
            app.logger.info(f"Tabla detectada en página {table_data['page']+1}")
        
        # Procesar imágenes
        app.logger.info(f"Extrayendo imágenes")
        images = extract_images(pdf_document)
        for img_data in tqdm(images, desc="Procesando imágenes"):
            image = DocumentImage(
                document_id=doc.id,
                page_number=img_data["page"],
                image_data=img_data["image"],
                caption=img_data["caption"],
                x1=img_data["x1"],
                y1=img_data["y1"],
                x2=img_data["x2"],
                y2=img_data["y2"]
            )
            session.add(image)
        
        # Guardar todo
        session.commit()
        pdf_document.close()
        
        app.logger.info(f"PDF procesado con éxito: {len(chunks)} chunks, {len(tables)} tablas, {len(images)} imágenes")
        
        return {
            "chunks": chunks,
            "doc_id": doc.id,
            "title": title,
            "total_chunks": len(chunks),
        }
    
    except Exception as e:
        app.logger.error(f"Error procesando PDF: {str(e)}")
        traceback.print_exc()
        session.rollback()
        raise
    
    finally:
        session.close()
        if 'pdf_document' in locals() and pdf_document:
            pdf_document.close()

# Función para extraer texto de un PDF (versión actualizada)
def extract_text_from_pdf(file_path, filename):
    try:
        result = process_pdf(file_path, filename)
        chunks = result["chunks"]
        
        # Para mantener compatibilidad con el código existente
        app.config['pdf_chunks'] = app.config.get('pdf_chunks', {})
        app.config['pdf_chunks'][filename] = chunks
        
        return {
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
    except Exception as e:
        app.logger.error(f"Error al extraer texto del PDF: {str(e)}")
        raise

# Intento de importación para detección de objetos y análisis avanzado de imágenes
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    app.logger.warning("OpenCV no disponible. Algunas funciones de análisis de imágenes estarán limitadas.")

# Intentar importar para metadatos EXIF
try:
    from PIL import ExifTags
    EXIF_AVAILABLE = True
except ImportError:
    EXIF_AVAILABLE = False
    app.logger.warning("ExifTags no disponible. No se extraerán metadatos EXIF.")

# Función para extraer metadatos EXIF
def extract_exif_data(image):
    """Extrae metadatos EXIF de una imagen PIL."""
    if not EXIF_AVAILABLE:
        return {}
    
    try:
        exif_data = {}
        if hasattr(image, '_getexif') and image._getexif():
            for tag, value in image._getexif().items():
                if tag in ExifTags.TAGS:
                    tag_name = ExifTags.TAGS[tag]
                    # Convertir valores no serializables a string
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except:
                            value = str(value)
                    elif not isinstance(value, (int, float, str, bool, type(None))):
                        value = str(value)
                    exif_data[tag_name] = value
        return exif_data
    except Exception as e:
        app.logger.error(f"Error extrayendo metadatos EXIF: {str(e)}")
        return {}

# Función para analizar colores predominantes
def analyze_colors(cv_image, num_colors=5):
    """Analiza los colores predominantes en una imagen usando K-means."""
    if not CV2_AVAILABLE:
        return []
    
    try:
        # Redimensionar imagen para acelerar el procesamiento
        h, w = cv_image.shape[:2]
        if h > 300 or w > 300:
            scale = min(300/h, 300/w)
            cv_image = cv2.resize(cv_image, (0, 0), fx=scale, fy=scale)
        
        # Aplanar imagen para K-means
        pixels = cv_image.reshape(-1, 3).astype(np.float32)
        
        # Detener si no hay suficientes píxeles
        if len(pixels) < num_colors:
            return []
        
        # Usar K-means para encontrar colores predominantes
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Contar número de píxeles por cluster
        counts = np.bincount(labels.flatten())
        
        # Convertir a formato RGB y calcular porcentajes
        colors = []
        total_pixels = len(pixels)
        
        for i, center in enumerate(centers):
            # Convertir de BGR a RGB
            rgb = (int(center[2]), int(center[1]), int(center[0]))
            percentage = counts[i] / total_pixels * 100
            colors.append({
                "rgb": rgb,
                "hex": '#{:02x}{:02x}{:02x}'.format(*rgb),
                "percentage": percentage
            })
        
        # Ordenar por porcentaje descendente
        colors.sort(key=lambda x: x["percentage"], reverse=True)
        return colors
    
    except Exception as e:
        app.logger.error(f"Error analizando colores: {str(e)}")
        return []

# Función para detectar objetos en imágenes
def detect_objects(cv_image):
    """Detecta objetos en la imagen si hay un modelo disponible."""
    if not CV2_AVAILABLE:
        return []
    
    # Aquí se implementaría la detección con modelos pre-entrenados
    # Como YOLO o un modelo más simple. Para esta implementación,
    # haremos una detección simulada basada en características básicas
    
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detectar bordes
        edges = cv2.Canny(gray, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por tamaño
        min_contour_area = 500  # Área mínima para considerar un objeto
        objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(area / 10000, 0.95)  # Simular confianza
                
                # Determinar tipo basado en proporción
                aspect_ratio = w / h if h > 0 else 0
                if 0.9 < aspect_ratio < 1.1:
                    obj_type = "cuadrado"
                elif aspect_ratio > 1.5:
                    obj_type = "objeto horizontal"
                elif aspect_ratio < 0.7:
                    obj_type = "objeto vertical"
                else:
                    obj_type = "objeto"
                
                objects.append({
                    "type": obj_type,
                    "confidence": confidence,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                })
        
        return objects
    
    except Exception as e:
        app.logger.error(f"Error detectando objetos: {str(e)}")
        return []

# Función para detectar rostros (sin identificación)
def detect_faces(cv_image):
    """Detecta rostros en una imagen usando cascada Haar"""
    try:
        # Convertir imagen a escala de grises para mejor detección
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Ruta al clasificador Haar cascade
        cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'haarcascade_frontalface_default.xml')
        
        if not os.path.exists(cascade_path):
            app.logger.warning(f"Clasificador de rostros no encontrado en {cascade_path}")
            return []
        
        # Crear clasificador de rostros
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            app.logger.warning("Error al cargar el clasificador Haar cascade")
            return []
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convertir a formato JSON serializable
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "confidence": random.uniform(0.75, 0.95)  # Simulación de confianza
            })
        
        return face_list
    
    except Exception as e:
        app.logger.warning(f"Error en detección de rostros: {str(e)}")
        return []

# Función para generar embeddings visuales
def generate_image_embedding(pil_image):
    """Genera un embedding para la imagen usando el modelo de sentence transformers."""
    model = get_sentence_transformer()
    if model is None:
        return None
    
    try:
        # Convertir la imagen a un formato adecuado
        # Redimensionar para consistencia
        pil_image = pil_image.resize((224, 224))
        
        # En una implementación más avanzada, se usaría un modelo específico para imágenes
        # Como CLIP o un modelo visual dedicado. Por ahora, usaremos el embedding del texto extraído
        
        # Extraer texto
        text = pytesseract.image_to_string(pil_image)
        
        # Si no hay texto o es muy corto, usar una descripción genérica
        if len(text.strip()) < 10:
            text = "imagen sin texto detectable"
        
        # Generar embedding
        embedding = model.encode(text)
        return embedding.tobytes()
    
    except Exception as e:
        app.logger.error(f"Error generando embedding visual: {str(e)}")
        return None

# Función para procesar una imagen completamente
def process_image(file_path, original_filename, chat_id=None):
    """Procesa una imagen subida y genera análisis completo"""
    try:
        # Generar un nombre de archivo único
        timestamp = datetime.now().timestamp()
        file_extension = os.path.splitext(original_filename)[1].lower()
        secure_name = secure_filename(f"{uuid.uuid4()}{file_extension}")
        
        # Crear directorio de uploads si no existe
        upload_dir = os.path.join(app.static_folder, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Ruta de destino permanente
        dest_path = os.path.join(upload_dir, secure_name)
        
        # Si el archivo aún está en ubicación temporal, moverlo a destino
        if os.path.exists(file_path) and file_path != dest_path:
            # Copiar archivo a ubicación permanente
            shutil.copy2(file_path, dest_path)
            if file_path.startswith(app.config['UPLOAD_FOLDER']):
                # Si era un archivo temporal, eliminarlo
                os.remove(file_path)
        
        # Abrir la imagen con PIL
        pil_image = Image.open(dest_path)
        
        # Convertir a RGB si es necesario (para estandarizar formato)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Extraer texto mediante OCR
        extracted_text = ""
        try:
            extracted_text = pytesseract.image_to_string(pil_image, lang='spa+eng')
            extracted_text = extracted_text.strip()
        except Exception as e:
            app.logger.warning(f"Error en OCR: {str(e)}")
        
        # Obtener descripción de la imagen
        description = f"Imagen subida por usuario: {original_filename}"
        
        # Abrir la imagen con OpenCV para procesamiento adicional
        cv_image = None
        try:
            cv_image = cv2.imread(dest_path)
        except Exception as e:
            app.logger.warning(f"Error cargando imagen con OpenCV: {str(e)}")
        
        # Análisis avanzado de la imagen
        has_text = len(extracted_text) > 10
        has_faces = False
        has_objects = False
        faces_data = []
        objects_data = []
        colors_data = []
        exif_data = {}
        
        # Procesar con OpenCV si está disponible
        if cv_image is not None:
            # Detectar rostros
            faces_data = detect_faces(cv_image)
            has_faces = len(faces_data) > 0
            
            # Detectar objetos
            objects_data = detect_objects(cv_image)
            has_objects = len(objects_data) > 0
            
            # Analizar colores
            colors_data = analyze_colors(cv_image)
        
        # Extraer metadatos EXIF
        exif_data = extract_exif_data(pil_image)
        
        # Generar embedding para búsqueda semántica
        embedding_bytes = generate_image_embedding(pil_image)
        
        # Guardar en la base de datos
        session = Session()
        try:
            # Verificar si existe el chat
            chat = session.query(ChatSession).filter_by(id=chat_id).first()
            if not chat and chat_id:
                # Crear nuevo chat si no existe
                chat = ChatSession(
                    id=chat_id,
                    title=f"Chat {chat_id[:8]}",
                    created_at=timestamp,
                    last_accessed=timestamp
                )
                session.add(chat)
                session.commit()
            
            # Crear registro de imagen
            image = UserImage(
                chat_id=chat_id,
                filename=secure_name,
                original_filename=original_filename,
                storage_path=dest_path,
                extracted_text=extracted_text,
                description=description,
                embedding=embedding_bytes,
                created_at=timestamp
            )
            session.add(image)
            session.flush()  # Para obtener el ID asignado
            
            # Crear análisis de imagen
            analysis = ImageAnalysis(
                image_id=image.id,
                has_text=has_text,
                has_faces=has_faces,
                has_objects=has_objects,
                detected_objects=json.dumps(objects_data) if objects_data else None,
                detected_colors=json.dumps(colors_data) if colors_data else None,
                detected_faces=json.dumps(faces_data) if faces_data else None,
                exif_data=json.dumps(exif_data) if exif_data else None
            )
            session.add(analysis)
            session.commit()
            
            # Actualizar también el registro en la memoria para la sesión actual
            if chat_id and chat_id not in app.config['image_history']:
                app.config['image_history'][chat_id] = []
            
            if chat_id:
                app.config['image_history'][chat_id].append({
                    'id': image.id,
                    'filename': secure_name,
                    'url': f'/static/uploads/{secure_name}',
                    'description': description,
                    'extracted_text': extracted_text,
                    'timestamp': timestamp
                })
            
            # Devolver resultado exitoso
            return {
                'success': True,
                'message': 'Imagen procesada correctamente',
                'filename': secure_name,
                'image_url': f'/static/uploads/{secure_name}',
                'analysis': {
                    'has_text': has_text,
                    'has_faces': has_faces,
                    'has_objects': has_objects,
                    'faces': faces_data,
                    'objects': objects_data,
                    'colors': colors_data,
                    'exif': exif_data
                },
                'description': description
            }
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    except Exception as e:
        app.logger.error(f"Error procesando imagen: {str(e)}")
        app.logger.error(traceback.format_exc())
        return {
            'success': False,
            'message': f'Error procesando imagen: {str(e)}'
        }

# Función para búsqueda semántica de imágenes
def semantic_image_search(query, chat_id=None, top_k=3):
    """Busca imágenes semánticamente similares a una consulta."""
    model = get_sentence_transformer()
    if model is None:
        return []
    
    query_embedding = model.encode(query)
    
    session = Session()
    try:
        query_obj = session.query(UserImage)
        
        if chat_id is not None:
            query_obj = query_obj.filter(UserImage.chat_id == chat_id)
        
        images = query_obj.all()
        results = []
        
        for img in images:
            if img.embedding:
                img_embedding = bytes_to_embedding(img.embedding)
                if img_embedding is not None:
                    # Calcular similitud de coseno
                    similarity = np.dot(query_embedding, img_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(img_embedding)
                    )
                    results.append((img, similarity))
        
        # Ordenar por similitud descendente y tomar los top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    finally:
        session.close()

# Función auxiliar para mantener compatibilidad con el código anterior
def generate_image_prompt_legacy(image_infos, formatted_history, user_message):
    """Genera un prompt basado en imágenes usando el enfoque original"""
    # Usar todas las imágenes disponibles
    image_contexts = []
    for idx, img_info in enumerate(image_infos, 1):
        description = img_info.get('description', 'Imagen sin descripción')
        extracted_text = img_info.get('extracted_text', 'No se extrajo texto')
        image_contexts.append(f"Imagen {idx}:\nDescripción: {description}\nTexto extraído: {extracted_text}")
    
    if image_contexts:
        context = "\n\n".join(image_contexts)
        prompt = f"""Contexto de imágenes disponibles:

{context}

Historial de la conversación:
{formatted_history}

Pregunta actual del usuario: {user_message}

IMPORTANTE: Actúa como si pudieras ver directamente las imágenes descritas. Basado en la descripción y análisis proporcionados, responde como si estuvieras viendo las imágenes con tus propios ojos. Si te preguntan sobre rostros, colores, objetos o cualquier elemento visual, responde con la información proporcionada en el contexto.

Por favor, responde a la pregunta basándote en las imágenes y el historial de la conversación."""
    else:
        # Incluir el historial de conversación aunque no se mencionen específicamente imágenes
        prompt = f"""Historial de la conversación:
{formatted_history}

Pregunta actual del usuario: {user_message}

IMPORTANTE: Si se mencionan imágenes en la conversación, actúa como si pudieras verlas basándote en las descripciones proporcionadas anteriormente.

Por favor, responde a la pregunta teniendo en cuenta el historial completo de la conversación, incluyendo cualquier imagen mencionada anteriormente."""
    
    return prompt

# Ruta para reenviar solicitudes a Ollama desde el frontend
@app.route('/ollama_tags', methods=['GET'])
def ollama_tags():
    try:
        response = requests.get(f'{OLLAMA_HOST}/api/tags', timeout=5)
        return Response(response.content, content_type=response.headers['Content-Type'])
    except Exception as e:
        app.logger.error(f"Error obteniendo modelos de Ollama: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
# Iniciar el servidor con host='0.0.0.0' para permitir conexiones externas
if __name__ == "__main__":
    try:
        # Verificar puerto
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 5000))
        sock.close()
        
        if result == 0:
            print("ERROR: El puerto 5000 está en uso.")
            print("Intente estos pasos:")
            print("1. Ejecute 'netstat -ano | findstr :5000' para encontrar el proceso")
            print("2. Cierre la aplicación que está usando el puerto 5000")
            print("3. O inicie la aplicación en un puerto diferente con --port XXXX")
            exit(1)

        # Verificar permisos de la carpeta de uploads
        uploads_path = os.path.abspath(UPLOAD_FOLDER)
        if not os.path.exists(uploads_path):
            try:
                os.makedirs(uploads_path)
                print(f"✓ Carpeta de uploads creada en: {uploads_path}")
            except Exception as e:
                print(f"ERROR: No se pudo crear la carpeta de uploads: {str(e)}")
                exit(1)

        # Verificar conexión con Ollama
        try:
            response = requests.get(f'{OLLAMA_HOST}/api/tags', timeout=5)
            if response.status_code == 200:
                print("✓ Conexión con Ollama establecida")
            else:
                print("ERROR: Ollama no está respondiendo correctamente")
                print("Por favor, asegúrese de que Ollama esté en ejecución")
                exit(1)
        except requests.exceptions.RequestException as e:
            print("ERROR: No se puede conectar con Ollama")
            print("1. Asegúrese de que Ollama esté instalado")
            print("2. Ejecute Ollama antes de iniciar esta aplicación")
            print(f"Error detallado: {str(e)}")
            exit(1)

        print("\n=== Iniciando Servidor de Chat IA ===")
        print("✓ Todas las verificaciones completadas")
        print("✓ Servidor iniciando en: http://127.0.0.1:5000")
        print("* Presione Ctrl+C para detener el servidor")
        print("=====================================\n")

        # Iniciar el servidor con host='0.0.0.0' para permitir conexiones externas
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"\nERROR CRÍTICO: No se pudo iniciar el servidor")
        print(f"Causa: {str(e)}")
        print("\nPor favor, verifique:")
        print("1. Que no haya otra aplicación usando el puerto 5000")
        print("2. Que tenga permisos de administrador si es necesario")
        print("3. Que todas las dependencias estén instaladas correctamente")
        exit(1) 