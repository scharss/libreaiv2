# Libre AI - Chat con IA Local en Docker

Una aplicación web de chat con IA local que utiliza modelos de Ollama, con procesamiento de PDFs, OCR de imágenes y múltiples conversaciones, todo disponible en una implementación Docker multiplataforma.

![LibreAI](resources/libreai-screenshot.png)

## 🌟 Características Principales

- 💬 **Chat Interactivo con IA Local**: Sin conexión a servicios externos para máxima privacidad
- 📊 **Soporte para Múltiples Modelos**: Compatible con cualquier modelo de Ollama (Mistral, Llama, Qwen, DeepSeek, etc.)
- 📷 **OCR de Imágenes**: Extracción automática de texto de imágenes con soporte multilenguaje
- 📁 **Gestión de Conversaciones**: Organiza tus chats en diferentes sesiones
- 📄 **Procesamiento Avanzado de PDFs**: Análisis semántico y extracción de contenido
- 🌓 **Tema Claro/Oscuro**: Interfaz adaptable a tus preferencias
- 🌎 **Soporte Multiidioma**: Español e Inglés
- ✨ **Resaltado de Código**: Visualización mejorada de fragmentos de código
- 🧮 **Soporte para Fórmulas Matemáticas**: Renderizado de ecuaciones
- 🔄 **Implementación Docker**: Fácil instalación y ejecución en cualquier plataforma

## 🐳 Instalación con Docker

La aplicación utiliza Docker para funcionar de manera consistente en todas las plataformas. Sigue estas instrucciones para instalarla en tu sistema.

### Requisitos Previos

- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/) (incluido en Docker Desktop para Windows y Mac)
- Al menos 8GB de RAM disponible
- Aproximadamente 10GB de espacio en disco (variable según los modelos que descargues)

### 🪟 Instalación en Windows

1. **Instalar Docker Desktop**:
   - Descarga [Docker Desktop para Windows](https://www.docker.com/products/docker-desktop/)
   - Instala WSL 2 si es necesario (Windows 10/11):
     ```powershell
     wsl --install
     ```
   - Ejecuta el instalador de Docker Desktop y asegúrate de que la opción "Use WSL 2" esté seleccionada
   - Reinicia tu computadora

2. **Clonar/Descargar el Repositorio**:
   ```powershell
   git clone [URL_DEL_REPOSITORIO]
   cd [NOMBRE_DEL_DIRECTORIO]
   ```
   O descarga y extrae el ZIP del repositorio

3. **Iniciar los Contenedores**:
   ```powershell
   docker-compose up -d
   ```

4. **Descargar un Modelo de IA**:
   ```powershell
   docker exec -it libreimagen-4-ollama-1 bash
   ollama pull mistral
   exit
   ```

5. **Acceder a la Aplicación**:
   - Abre un navegador y ve a http://localhost:5000

### 🍎 Instalación en macOS

1. **Instalar Docker Desktop**:
   - Descarga [Docker Desktop para Mac](https://www.docker.com/products/docker-desktop/)
   - Ejecuta el instalador (asegúrate de seleccionar la versión correcta para tu Mac: Intel o Apple Silicon)

2. **Clonar/Descargar el Repositorio**:
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd [NOMBRE_DEL_DIRECTORIO]
   ```
   O descarga y extrae el ZIP del repositorio

3. **Iniciar los Contenedores**:
   ```bash
   docker-compose up -d
   ```

4. **Descargar un Modelo de IA**:
   ```bash
   docker exec -it libreimagen-4-ollama-1 bash
   ollama pull mistral
   exit
   ```

5. **Acceder a la Aplicación**:
   - Abre un navegador y ve a http://localhost:5000

### 🐧 Instalación en Linux

1. **Instalar Docker y Docker Compose**:
   - Ubuntu/Debian:
     ```bash
     sudo apt update
     sudo apt install docker.io docker-compose
     sudo systemctl enable --now docker
     ```
   - Fedora/RHEL/CentOS:
     ```bash
     sudo dnf install docker docker-compose
     sudo systemctl enable --now docker
     ```
   - Arch Linux:
     ```bash
     sudo pacman -S docker docker-compose
     sudo systemctl enable --now docker
     ```

2. **Agregar tu Usuario al Grupo Docker** (para usar Docker sin sudo):
   ```bash
   sudo usermod -aG docker $USER
   ```
   Cierra sesión y vuelve a iniciarla para que los cambios surtan efecto.

3. **Clonar/Descargar el Repositorio**:
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd [NOMBRE_DEL_DIRECTORIO]
   ```
   O descarga y extrae el ZIP del repositorio

4. **Iniciar los Contenedores**:
   ```bash
   docker-compose up -d
   ```

5. **Descargar un Modelo de IA**:
   ```bash
   docker exec -it libreimagen-4-ollama-1 bash
   ollama pull mistral
   exit
   ```

6. **Acceder a la Aplicación**:
   - Abre un navegador y ve a http://localhost:5000

## 💡 Uso Básico

1. **Seleccionar un Modelo**:
   - Haz clic en el ícono de configuración en la barra lateral
   - Selecciona un modelo de la lista de modelos disponibles

2. **Iniciar una Conversación**:
   - Haz clic en "Nuevo chat"
   - Escribe tu mensaje en el campo de texto inferior
   - Presiona Enter o haz clic en el ícono de enviar

3. **Procesar PDFs**:
   - Haz clic en el ícono de clip junto al campo de texto
   - Selecciona un archivo PDF
   - Espera a que se procese y luego realiza preguntas sobre su contenido

4. **Procesar Imágenes**:
   - Haz clic en el ícono de cámara junto al campo de texto
   - Selecciona una imagen
   - La aplicación extraerá el texto mediante OCR y podrás realizar preguntas sobre su contenido

## 🚀 Modelos Recomendados

Para un rendimiento equilibrado, recomendamos estos modelos de Ollama:

- **Mistral 7B**: Buen equilibrio entre rendimiento y velocidad, ideal para comenzar
- **DeepSeek Coder**: Excelente para tareas relacionadas con programación
- **Qwen2.5 Coder 7B**: Buen rendimiento en tareas generales y programación
- **Llama2**: Opción potente para tareas de conversación general

## ⚠️ Solución de Problemas

### No se pueden ver los modelos de Ollama
- Asegúrate de haber descargado al menos un modelo usando `docker exec -it libreimagen-4-ollama-1 bash` y luego `ollama pull mistral`
- Verifica que ambos contenedores estén en ejecución con `docker ps`
- Revisa los logs con `docker logs libreimagen-4-web-1`

### Error de conexión a la web
- Asegúrate de que no haya otro servicio usando el puerto 5000
- Comprueba si los contenedores están en ejecución con `docker ps`
- Reinicia los contenedores: `docker-compose down` y luego `docker-compose up -d`

### Problemas con PDFs
- Asegúrate de que el PDF no esté protegido
- Verifica que el PDF contenga texto seleccionable

### Problemas con el OCR
- El OCR funciona mejor con imágenes claras y texto bien definido
- El reconocimiento puede ser limitado con escritura a mano o fuentes inusuales

## 🔒 Privacidad

Toda la inferencia del modelo se realiza localmente en tu máquina. Ningún dato se envía a servicios externos, garantizando total privacidad en tus conversaciones.

## 📖 Licencia

Este proyecto está licenciado bajo los términos especificados en el archivo LICENSE.

---

Para más información o soporte, visita [el repositorio del proyecto](https://github.com/tu-usuario/tu-repositorio).