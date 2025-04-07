# Ejecutar Libre AI con Docker

Esta guía te mostrará cómo ejecutar Libre AI utilizando Docker y Docker Compose, facilitando la instalación y ejecución en cualquier sistema compatible con Docker.

## Requisitos previos

- Docker: [Instrucciones de instalación](https://docs.docker.com/get-docker/)
- Docker Compose: [Instrucciones de instalación](https://docs.docker.com/compose/install/)

## Configuración rápida

1. Clona el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd [NOMBRE_DEL_DIRECTORIO]
```

2. Construye y ejecuta los contenedores:
```bash
docker-compose up -d
```

3. Accede a la aplicación en tu navegador:
```
http://localhost:5000
```

## Descargar modelos de IA

Una vez que los contenedores estén funcionando, necesitarás descargar al menos un modelo para Ollama:

```bash
# Accede al contenedor de Ollama
docker-compose exec ollama bash

# Descarga el modelo deseado (ejemplos)
ollama pull tinyllama   # Modelo pequeño y rápido
ollama pull mistral     # Modelo balanceado (recomendado)
ollama pull llama2      # Modelo potente
ollama pull deepseek-coder # Recomendado para programación

# Sale del contenedor
exit
```

## Verificar el estado

Para verificar que los contenedores están funcionando correctamente:

```bash
docker-compose ps
```

## Detener los contenedores

Para detener la aplicación:

```bash
docker-compose down
```

Para detener y eliminar volúmenes (esto eliminará los modelos descargados):

```bash
docker-compose down -v
```

## Solución de problemas

### No puedo acceder a la aplicación

- Verifica que los contenedores estén ejecutándose: `docker-compose ps`
- Comprueba los logs: `docker-compose logs web`
- Asegúrate de que no haya otro servicio usando el puerto 5000

### Ollama no responde

- Verifica los logs de Ollama: `docker-compose logs ollama`
- Reinicia el contenedor: `docker-compose restart ollama`
- Asegúrate de haber descargado al menos un modelo

### Error al cargar un modelo

- Verifica el espacio disponible en el disco
- Comprueba que el modelo está instalado: `docker-compose exec ollama ollama list`
- Intenta descargar el modelo nuevamente

## Personalización

### Cambiar el puerto

Si deseas cambiar el puerto de la aplicación web, edita el archivo `docker-compose.yml` y modifica la línea:

```yaml
ports:
  - "5000:5000"
```

Por ejemplo, para usar el puerto 8080:

```yaml
ports:
  - "8080:5000"
```

### Persistencia de datos

Los datos se almacenan en volúmenes de Docker:
- `ollama_data`: Almacena los modelos descargados
- `./uploads`: Almacena archivos subidos (imágenes, PDFs)
- `./data`: Almacena datos procesados y chunks de PDFs

## Notas importantes

- La primera descarga de un modelo puede tardar varios minutos dependiendo de tu conexión a internet.
- El rendimiento de los modelos dependerá de los recursos asignados a Docker.
- Para un mejor rendimiento en Windows y macOS, asegúrate de asignar suficiente memoria y CPU a Docker en la configuración. 