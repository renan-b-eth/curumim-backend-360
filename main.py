# curumim-backend/main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import os
from dotenv import load_dotenv
import aiofiles
import uuid
import boto3
import logging

# --- Configurar Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Carregar variáveis de ambiente ---
load_dotenv()

# --- Credenciais Twilio ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

if not twilio_client:
    logger.warning("Credenciais Twilio incompletas. O envio de mensagens via Twilio API pode estar desativado.")
else:
    logger.info("Cliente Twilio inicializado.")

# --- Credenciais Cloudflare R2 ---
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

R2_ENDPOINT_URL_PRIVATE = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else None
R2_ENDPOINT_URL_PUBLIC = f"https://pub-{R2_ACCOUNT_ID}.r2.dev" if R2_ACCOUNT_ID else None # Para gerar URLs públicas, se necessário

s3_client = None
if all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL_PRIVATE]):
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT_URL_PRIVATE,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto'
        )
        logger.info("Conectado ao Cloudflare R2 para armazenamento de áudios.")
    except Exception as e:
        logger.error(f"Erro ao conectar ao R2: {e}")
        s3_client = None
else:
    logger.warning("Credenciais do R2 incompletas. O upload de áudios estará desativado.")

app = FastAPI()

# --- Gerenciamento de Estado do Chatbot ---
# Em um sistema de produção, use um banco de dados (ex: Redis, PostgreSQL) para persistir o estado.
# Para este MVP, vamos manter em memória (user_states), mas note que será resetado se o servidor reiniciar.
user_states = {} # { "whatsapp:+5511999999999": {"stage": "initial", "metadata": {}} }

# --- Funções Auxiliares ---
async def download_audio_from_twilio(media_url: str, content_type: str) -> str | None:
    """Faz o download do arquivo de áudio do Twilio e salva temporariamente."""
    if not twilio_client:
        logger.error("Twilio client não está configurado. Não é possível baixar áudio.")
        return None
    
    try:
        response = twilio_client.http_client.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        response.raise_for_status() # Lança exceção para status de erro (4xx ou 5xx)
        audio_data = response.content

        # Extensão do arquivo baseada no Content-Type
        ext = content_type.split('/')[-1] if '/' in content_type else 'ogg' # Twilio geralmente envia .ogg
        temp_file_name = f"/tmp/curumim_audio_{uuid.uuid4().hex}.{ext}"
        
        async with aiofiles.open(temp_file_name, "wb") as out_file:
            await out_file.write(audio_data)
        logger.info(f"Áudio baixado e salvo temporariamente como: {temp_file_name}")
        return temp_file_name
    except Exception as e:
        logger.error(f"Erro ao baixar áudio da URL {media_url}: {e}")
        return None

def upload_audio_to_r2(file_path: str, bucket_key: str, content_type: str) -> str | None:
    """Faz o upload de um arquivo de áudio para o Cloudflare R2."""
    if not s3_client:
        logger.error("S3 client (R2) não está configurado. Não é possível fazer upload de áudio.")
        return None
    try:
        s3_client.upload_file(file_path, R2_BUCKET_NAME, bucket_key, ExtraArgs={'ContentType': content_type})
        public_url = f"{R2_ENDPOINT_URL_PUBLIC}/{R2_BUCKET_NAME}/{bucket_key}"
        logger.info(f"Áudio '{bucket_key}' carregado para R2. URL pública: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Erro ao fazer upload do arquivo '{file_path}' para R2 como '{bucket_key}': {e}")
        return None

# --- Rotas da API ---
@app.get("/")
async def read_root():
    return {"message": "Curumim WhatsApp Bot is running!"}

@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(...),  # Número do usuário (ex: whatsapp:+5511...)
    Body: str = Form(None), # Mensagem de texto enviada pelo usuário
    NumMedia: str = Form(None), # Número de mídias (ex: '0', '1')
    MediaUrl0: str = Form(None), # URL da primeira mídia (se houver)
    MediaContentType0: str = Form(None) # Tipo de conteúdo da primeira mídia (se houver)
):
    twiml_response = MessagingResponse() # Objeto para construir a resposta do Twilio
    sender_id = From
    
    # --- Logging Detalhado da Requisição ---
    logger.info(f"Requisição de {sender_id}: Body='{Body}', NumMedia='{NumMedia}', MediaUrl0='{MediaUrl0}', MediaContentType0='{MediaContentType0}'")

    # --- Obter/Inicializar Estado do Usuário ---
    if sender_id not in user_states:
        user_states[sender_id] = {"stage": "initial", "metadata": {}}
        logger.info(f"Novo usuário {sender_id}. Estado inicializado: {user_states[sender_id]}")
    
    user_state = user_states[sender_id]
    current_stage = user_state["stage"]
    logger.info(f"Estado atual de {sender_id} antes da lógica: {current_stage} | Metadata: {user_state['metadata']}")

    # --- Lógica do Chatbot ---

    # ESTÁGIO 1: Initial (Primeira interação ou reinício)
    if current_stage == "initial":
        logger.info(f"[{sender_id}] Entrou no estágio 'initial'.")
        twiml_response.message("Olá! Eu sou Curumim, seu assistente para o projeto Angelia AI. Posso te ajudar a contribuir com sua voz para a pesquisa de saúde.")
        twiml_response.message("Para começar, digite 'COMEÇAR'.")
        user_state["stage"] = "waiting_start"
        logger.info(f"[{sender_id}] Transicionou para estágio 'waiting_start'.")

    # ESTÁGIO 2: Waiting for 'COMEÇAR'
    elif current_stage == "waiting_start":
        logger.info(f"[{sender_id}] Entrou no estágio 'waiting_start'.")
        if Body and Body.lower() == "começar":
            logger.info(f"[{sender_id}] Recebeu 'COMEÇAR'.")
            twiml_response.message("Ótimo! Vamos começar. Para contribuir, por favor, grave e envie um áudio com uma *vogal 'A' sustentada* por 3 a 5 segundos (ex: Aaaaaa...).")
            twiml_response.message("Em seguida, vou pedir algumas informações.")
            user_state["stage"] = "waiting_audio_a"
            user_state["metadata"]["task_type"] = "vogal_a_sustentada" # Define o tipo de tarefa
            logger.info(f"[{sender_id}] Transicionou para estágio 'waiting_audio_a'.")
        else:
            logger.info(f"[{sender_id}] Mensagem inválida em 'waiting_start': '{Body}'.")
            twiml_response.message("Entendi. Por favor, digite 'COMEÇAR' para iniciarmos.")

    # ESTÁGIO 3: Waiting for Audio 'A'
    elif current_stage == "waiting_audio_a":
        logger.info(f"[{sender_id}] Entrou no estágio 'waiting_audio_a'.")
        if NumMedia and int(NumMedia) > 0 and MediaContentType0 and MediaContentType0.startswith("audio/"):
            logger.info(f"[{sender_id}] Áudio detectado. Baixando de {MediaUrl0}...")
            temp_audio_path = await download_audio_from_twilio(MediaUrl0, MediaContentType0)
            
            if temp_audio_path:
                r2_key = f"curumim_audios/{sender_id.replace('whatsapp:', '')}/{user_state['metadata'].get('task_type', 'unknown_task')}_{uuid.uuid4().hex}.ogg"
                public_audio_url = upload_audio_to_r2(temp_audio_path, r2_key, MediaContentType0)
                
                if public_audio_url:
                    twiml_response.message(f"Áudio recebido e salvo no R2! Obrigado pela sua contribuição.")
                    logger.info(f"[{sender_id}] Áudio salvo no R2: {public_audio_url}")
                else:
                    twiml_response.message("Áudio recebido, mas houve um problema ao salvar no R2. Por favor, tente novamente.")
                    logger.error(f"[{sender_id}] Falha ao salvar áudio no R2.")
                
                # Limpar arquivo temporário
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                    logger.info(f"[{sender_id}] Arquivo temporário '{temp_audio_path}' removido.")

                # Próxima etapa: coletar metadados
                twiml_response.message("Para complementar sua contribuição, por favor, me diga sua *idade* (apenas números).")
                user_state["stage"] = "waiting_age"
                logger.info(f"[{sender_id}] Transicionou para estágio 'waiting_age'.")
            else:
                twiml_response.message("Desculpe, tive um problema ao baixar seu áudio. Poderia tentar novamente?")
                logger.error(f"[{sender_id}] Falha ao baixar áudio.")
        else:
            logger.info(f"[{sender_id}] Não recebeu áudio no estágio 'waiting_audio_a'. Mensagem: '{Body}'.")
            twiml_response.message("Não recebi um áudio. Por favor, grave e envie o áudio da vogal 'A' sustentada.")

    # ESTÁGIO 4: Waiting for Age
    elif current_stage == "waiting_age":
        logger.info(f"[{sender_id}] Entrou no estágio 'waiting_age'.")
        if Body and Body.isdigit():
            age = int(Body)
            user_state["metadata"]["age"] = age
            twiml_response.message("Idade registrada! Agora, qual é o seu *gênero*? (Ex: Masculino, Feminino, Outro)")
            user_state["stage"] = "waiting_gender"
            logger.info(f"[{sender_id}] Idade '{age}' registrada. Transicionou para estágio 'waiting_gender'.")
        else:
            logger.info(f"[{sender_id}] Idade inválida em 'waiting_age': '{Body}'.")
            twiml_response.message("Por favor, digite sua idade em números.")

    # ESTÁGIO 5: Waiting for Gender
    elif current_stage == "waiting_gender":
        logger.info(f"[{sender_id}] Entrou no estágio 'waiting_gender'.")
        if Body:
            gender = Body.strip().lower()
            user_state["metadata"]["gender"] = gender
            twiml_response.message("Gênero registrado! Sua contribuição está completa. Muito obrigado por ajudar a Angelia AI!")
            # Mensagem de depuração final
            twiml_response.message(f"Seus dados coletados: {user_state['metadata']}")
            user_state["stage"] = "finished" 
            logger.info(f"[{sender_id}] Gênero '{gender}' registrado. Transicionou para estágio 'finished'.")
        else:
            logger.info(f"[{sender_id}] Gênero inválido em 'waiting_gender': '{Body}'.")
            twiml_response.message("Por favor, digite seu gênero.")

    # ESTÁGIO 6: Finished (Oferece reiniciar)
    elif current_stage == "finished":
        logger.info(f"[{sender_id}] Entrou no estágio 'finished'.")
        if Body and Body.lower() == "reiniciar":
            logger.info(f"[{sender_id}] Recebeu 'REINICIAR'. Reiniciando conversa.")
            user_states[sender_id] = {"stage": "initial", "metadata": {}} # Reinicia o estado
            # Simula a primeira mensagem para reiniciar o fluxo de boas-vindas
            twiml_response.message("Reiniciando a conversa. " +
                                   "Olá! Eu sou Curumim, seu assistente para o projeto Angelia AI. Posso te ajudar a contribuir com sua voz para a pesquisa de saúde." +
                                   " Para começar, digite 'COMEÇAR'.")
            user_state["stage"] = "waiting_start" # Transiciona para o estágio correto após o reinício
        else:
            logger.info(f"[{sender_id}] Mensagem em 'finished': '{Body}'.")
            twiml_response.message("Já coletamos sua contribuição! Se quiser começar de novo, digite 'REINICIAR'.")


    # --- Salvar Estado do Usuário ---
    user_states[sender_id] = user_state # Garante que o estado seja salvo no dicionário (mesmo que em memória)
    logger.info(f"Estado final de {sender_id} após a lógica: {user_states[sender_id]}")

    # --- Retornar Resposta TwiML ---
    response_xml = str(twiml_response)
    logger.info(f"[{sender_id}] Resposta TwiML gerada: {response_xml}")
    return PlainTextResponse(response_xml, media_type="text/xml")