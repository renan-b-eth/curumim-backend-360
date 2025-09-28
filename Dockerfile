# Use uma imagem base Python slim para um tamanho menor
FROM python:3.9-slim-buster

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo requirements.txt e instale as dependências
# O --no-cache-dir reduz o tamanho da imagem, e o --upgrade garante as versões mais recentes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copie o resto do seu código para o contêiner
# Isso incluirá main.py, .env (que não será usado diretamente, mas pode estar lá), etc.
# IMPORTANTE: Garanta que google-credentials.json NÃO esteja aqui.
COPY . .

# Comando para iniciar o servidor Uvicorn
# Cloud Run injeta a variável de ambiente PORT, então usamos 0.0.0.0 e $PORT
# 'main:api_app' significa que Uvicorn procurará uma variável chamada 'api_app' no arquivo 'main.py'
CMD ["uvicorn", "main:api_app", "--host", "0.0.0.0", "--port", "8080"] # Cloud Run escuta na 8080