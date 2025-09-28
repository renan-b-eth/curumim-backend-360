import os
from dotenv import load_dotenv
import uuid
import boto3
import logging
import io
import tempfile
import json
import asyncio
import time # Para simular atrasos quando necessário

# --- Imports para ASR (Whisper) ---
from transformers import pipeline
import torch
import soundfile as sf # Útil para garantir que o áudio está no formato certo para o Whisper
import numpy as np # Para manipulação de arrays de áudio, se necessário

# --- Imports para Google Cloud TTS ---
from google.cloud import texttospeech
from google.oauth2 import service_account

# --- Imports para FastAPI e Cliente HTTP ---
from fastapi import FastAPI, Request, Response, Form, HTTPException
import uvicorn
import httpx

# --- Configuração FastAPI para Webhook da Meta ---
api_app = FastAPI()

# --- Configurar Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Carregar variáveis de ambiente ---
load_dotenv()

# --- Configuração FastAPI ---
# Token para verificação do webhook. Precisa ser o mesmo que você configurar no Meta for Developers.
WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN")
if not WEBHOOK_VERIFY_TOKEN:
    logger.error("WEBHOOK_VERIFY_TOKEN não encontrado no .env. Configure para segurança do webhook.")
    # Em produção, você pode querer lançar um erro ou desabilitar o bot
    # raise ValueError("WEBHOOK_VERIFY_TOKEN não configurado. Impossível verificar webhooks.")


# --- Credenciais WhatsApp Business Cloud API (Meta) ---
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
# A URL base da API da Meta. Usando v19.0. Verifique a versão mais recente na documentação da Meta.
WHATSAPP_API_BASE_URL = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"

if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
    logger.error("Credenciais da WhatsApp Business Cloud API (WHATSAPP_ACCESS_TOKEN ou WHATSAPP_PHONE_NUMBER_ID) incompletas no .env. A integração com WhatsApp estará desativada.")


# --- Configuração ASR (Whisper Local) ---
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-small")
asr_pipeline = None
try:
    logger.info(f"Tentando carregar modelo ASR '{WHISPER_MODEL_NAME}'...")
    # Tenta usar GPU (cuda:0) se disponível, senão usa CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr_pipeline = pipeline("automatic-speech-recognition", model=WHISPER_MODEL_NAME, device=device)
    logger.info(f"Modelo ASR '{WHISPER_MODEL_NAME}' carregado com sucesso no {device}.")
except ImportError:
    logger.error(f"Biblioteca 'transformers' ou 'torch' não instalada para ASR. Instale com: pip install transformers torch")
except Exception as e:
    logger.error(f"Erro CRÍTICO ao carregar modelo ASR '{WHISPER_MODEL_NAME}': {e}. A transcrição de voz será desativada.")
    asr_pipeline = None

# --- Configuração TTS (Google Cloud Text-to-Speech) ---
google_tts_client = None
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if GOOGLE_TTS_API_KEY:
    try:
        google_tts_client = texttospeech.TextToSpeechClient(api_key=GOOGLE_TTS_API_KEY)
        logger.info("Cliente Google Cloud TTS inicializado com Chave de API.")
    except Exception as e:
        logger.error(f"Erro ao inicializar cliente Google Cloud TTS com Chave de API: {e}. Verifique GOOGLE_TTS_API_KEY e se o faturamento está ativo no Google Cloud.")
        google_tts_client = None
elif GOOGLE_APPLICATION_CREDENTIALS:
    if os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        try:
            credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
            google_tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
            logger.info("Cliente Google Cloud TTS inicializado com Conta de Serviço.")
        except ImportError:
            logger.error(f"Biblioteca 'google-auth-oauthlib' não instalada para Google Cloud TTS com Conta de Serviço. Instale com: pip install google-auth-oauthlib")
        except Exception as e:
            logger.error(f"Erro ao inicializar cliente Google Cloud TTS com Conta de Serviço de {GOOGLE_APPLICATION_CREDENTIALS}: {e}. Verifique o arquivo JSON e as permissões.")
            google_tts_client = None
    else:
        logger.error(f"Arquivo de credenciais do Google Cloud TTS não encontrado no caminho: {GOOGLE_APPLICATION_CREDENTIALS}. A geração de voz será desativada.")
        google_tts_client = None
else:
    logger.error("Nenhuma credencial do Google Cloud TTS (GOOGLE_TTS_API_KEY ou GOOGLE_APPLICATION_CREDENTIALS) configurada no .env. A geração de voz será desativada.")

# --- Credenciais Cloudflare R2 ---
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

s3_client = None
R2_ENDPOINT_URL_PRIVATE = None
R2_ENDPOINT_URL_PUBLIC = None
if all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, R2_BUCKET_NAME]):
    try:
        R2_ENDPOINT_URL_PRIVATE = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
        s3_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT_URL_PRIVATE,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto' # Cloudflare R2 usa 'auto' como região
        )
        # URL pública para servir arquivos. Configurar no painel do R2 para ser pública.
        # Se usar um CNAME customizado (ex: audios.kurumim.com.br), altere aqui para ele.
        R2_ENDPOINT_URL_PUBLIC = f"https://pub-{R2_ACCOUNT_ID}.r2.dev/{R2_BUCKET_NAME}"
        logger.info("Conectado ao Cloudflare R2 para armazenamento de áudios.")
    except Exception as e:
        logger.error(f"Erro ao conectar ao R2: {e}. Verifique as credenciais e o Account ID. O upload de áudios estará desativado.")
        s3_client = None
else:
    logger.warning("Credenciais do R2 incompletas no .env. O upload de áudios estará desativado.")


# --- Gerenciamento de Estado do Chatbot ---
# Dicionário para armazenar o estado de cada usuário.
# Para um projeto de produção, isso DEVERIA ser um banco de dados (Redis, PostgreSQL, etc.)
# para persistência do estado entre reinícios do bot e para permitir escalabilidade horizontal.
user_states = {}

# --- Funções Auxiliares de R2 ---
def upload_audio_to_r2(file_path: str, bucket_key: str, content_type: str) -> str | None:
    """Faz o upload de um arquivo de áudio para o Cloudflare R2."""
    if not s3_client or not R2_ENDPOINT_URL_PUBLIC:
        logger.error("S3 client (R2) ou URL pública não estão configurados. Não é possível fazer upload de áudio.")
        return None
        
    try:
        # A URL pública do R2 já inclui o nome do bucket, então não precisamos adicioná-lo novamente aqui
        # ao formar a 'public_url'. Apenas garantimos que o s3_client usa o bucket_name corretamente.
        s3_client.upload_file(file_path, R2_BUCKET_NAME, bucket_key, ExtraArgs={'ContentType': content_type})
        public_url = f"{R2_ENDPOINT_URL_PUBLIC}/{bucket_key}"
        logger.info(f"Áudio '{bucket_key}' carregado para R2. URL pública: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Erro ao fazer upload do arquivo '{file_path}' para R2 como '{bucket_key}': {e}")
        return None

def upload_audio_bytes_to_r2(audio_bytes: bytes, bucket_key: str, content_type: str) -> str | None:
    """Faz o upload de bytes de áudio para o Cloudflare R2."""
    if not s3_client or not R2_ENDPOINT_URL_PUBLIC:
        logger.error("S3 client (R2) ou URL pública não estão configurados. Não é possível fazer upload de áudio.")
        return None
        
    try:
        s3_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=bucket_key,
            Body=audio_bytes,
            ContentType=content_type
        )
        public_url = f"{R2_ENDPOINT_URL_PUBLIC}/{bucket_key}"
        logger.info(f"Áudio (bytes) '{bucket_key}' carregado para R2. URL pública: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Erro ao fazer upload de bytes de áudio para R2 como '{bucket_key}': {e}")
        return None

# --- Funções de Voz (ASR e TTS) ---

async def transcribe_audio(audio_data_bytes: bytes, file_format: str = "ogg") -> str | None:
    """Transcreve áudio para texto usando o modelo Whisper local."""
    if not asr_pipeline:
        logger.error("ASR pipeline (Whisper) não configurado. Não é possível transcrever áudio.")
        return None
    try:
        # Whisper pipeline aceita um caminho de arquivo. Precisamos salvar o áudio temporariamente.
        with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as temp_audio_file:
            temp_audio_file.write(audio_data_bytes)
            temp_audio_file_path = temp_audio_file.name
            
        # Para pt-BR, é melhor usar o generate_kwargs={"language": "portuguese"}
        transcript_result = asr_pipeline(temp_audio_file_path, chunk_length_s=30, return_timestamps="word", generate_kwargs={"language": "portuguese"})
        transcript = transcript_result["text"]
        
        os.remove(temp_audio_file_path) # Limpa o arquivo temporário
        logger.info(f"Transcrição de áudio: '{transcript}'")
        return transcript.strip()
    except Exception as e:
        logger.error(f"Erro ao transcrever áudio com Whisper: {e}")
        if 'temp_audio_file_path' in locals() and os.path.exists(temp_audio_file_path):
            os.remove(temp_audio_file_path) # Garante que o arquivo temporário seja limpo mesmo em erro
        return None

async def generate_speech_from_text(text: str) -> bytes | None:
    """Gera áudio a partir de texto usando Google Cloud Text-to-Speech API."""
    if not google_tts_client:
        logger.error("Cliente Google Cloud TTS não inicializado. Não é possível gerar fala.")
        return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="pt-BR",
            name="pt-BR-Wavenet-C", # Experimente 'A', 'B', 'C', 'D' para diferentes vozes WaveNet
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE # ou MALE, NEUTRAL
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.OGG_OPUS # OGG_OPUS é um bom formato para WhatsApp
        )

        response = google_tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        logger.info(f"Fala gerada para o texto: '{text[:50]}...' com Google Cloud TTS.")
        return response.audio_content
    except Exception as e:
        logger.error(f"Erro ao gerar fala com Google Cloud TTS: {e}")
        return None

# --- Adaptação do Estado do Chatbot ---
def get_user_state_key(platform: str, user_id: str) -> str:
    """Gera uma chave única para o estado do usuário, combinando plataforma e ID."""
    return f"{platform}_{user_id}"

async def init_user_state(user_id: str, username: str) -> dict:
    """Inicializa ou reseta o estado de um usuário do WhatsApp."""
    state_key = get_user_state_key("whatsapp", user_id)
    user_states[state_key] = {
        "stage": "initial", # Novo estágio inicial para guiar a primeira interação
        "metadata": {
            "user_id": user_id, 
            "platform": "whatsapp", 
            "username": username,
            "name": None,
            "age": None,
            "diagnosis": None,
            "smoking_status": None,
            "emotional_state": None,
            "environment": None,
            "current_audio_task": None, # Qual tarefa de áudio está sendo solicitada no momento
            "audio_urls": {} # Para armazenar URLs dos áudios de cada tarefa específica
        },
        "tasks_queue": [], # Fila de tarefas de gravação de áudio
        "interaction_mode": "text" # Modo padrão é texto
    }
    logger.info(f"Estado inicializado/resetado para user {user_id} (key: {state_key}).")
    return user_states[state_key]

# --- Funções de Resposta para WhatsApp Business Cloud API (Meta) ---
async def send_whatsapp_response(to_number: str, user_state: dict, text: str, delay: int = 0) -> None:
    """Envia uma resposta via WhatsApp Business Cloud API da Meta."""
    if delay > 0:
        await asyncio.sleep(delay) # Adiciona um atraso antes de enviar a mensagem

    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logger.error("Credenciais da WhatsApp Business Cloud API incompletas. Não é possível enviar mensagens.")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"
    }
    
    # Payload padrão para texto
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text}
    }

    # Se o modo de interação for voz e o TTS estiver disponível, tenta enviar áudio
    # E verifica se R2 também está disponível para armazenar o áudio gerado
    if user_state["interaction_mode"] == "voice" and google_tts_client and s3_client and R2_ENDPOINT_URL_PUBLIC:
        try:
            audio_response_bytes = await generate_speech_from_text(text)
            if audio_response_bytes:
                audio_filename = f"whatsapp_tts_{uuid.uuid4().hex}.ogg"
                
                # Chave para o R2: Ex: whatsapp_audios/tts/5511987654321/tts_xyz.ogg
                # Adicione uma subpasta para TTS dentro do diretório do usuário
                r2_key = f"whatsapp_audios/{to_number}/tts/{audio_filename}"
                public_audio_url = upload_audio_bytes_to_r2(audio_response_bytes, r2_key, "audio/ogg")

                if public_audio_url:
                    payload = {
                        "messaging_product": "whatsapp",
                        "to": to_number,
                        "type": "audio",
                        "audio": {"link": public_audio_url}
                        # Opcional: "caption": text para adicionar texto ao áudio, mas WhatsApp limita o comprimento do caption em áudio.
                    }
                    logger.info(f"Resposta em voz (URL R2) enviada para WhatsApp via Meta API. URL: {public_audio_url}")
                else:
                    logger.error("Não foi possível obter URL pública para áudio do R2. Enviando como texto.")
                
        except Exception as e:
            logger.error(f"Falha ao gerar e enviar resposta em voz para WhatsApp via Meta API, retornando para texto: {e}")
            # Se falhou, o payload original de texto já está pronto e será usado
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(WHATSAPP_API_BASE_URL, headers=headers, json=payload)
            response.raise_for_status() # Levanta exceção para erros HTTP
            logger.info(f"Mensagem WhatsApp (Meta API) enviada com sucesso para {to_number}. Status: {response.status_code}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Erro HTTP ao enviar mensagem WhatsApp (Meta API) para {to_number}: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"Erro ao enviar mensagem WhatsApp (Meta API) para {to_number}: {e}")

# --- Lógica de Manipulação de Mensagens Principal ---
async def process_whatsapp_message(user_id: str, username: str, user_input_text: str = None, user_audio_media_id: str = None, raw_message_payload: str = None) -> (str, dict):
    """
    Processa uma mensagem de entrada do usuário do WhatsApp (texto ou áudio)
    e retorna a resposta em texto e o estado atualizado.
    """
    state_key = get_user_state_key("whatsapp", user_id)
    
    # Se não tem estado ou /start, inicializa/reseta
    if state_key not in user_states or (user_input_text and user_input_text.lower() == "/start"):
        user_states[state_key] = await init_user_state(user_id, username)
        # O estado inicial "initial" levará à pergunta de modo de interação
        return "Olá! Bem-vindo ao Kurumim. Como você gostaria de interagir? Por *texto* ou por *voz*?", user_states[state_key]

    user_state = user_states[state_key]
    current_stage = user_state["stage"]
    interaction_mode = user_state["interaction_mode"]
    logger.info(f"[whatsapp:{user_id}] Estado antes da lógica: {current_stage} | Modo: {interaction_mode}")

    processed_text = user_input_text # Assume que é texto inicialmente

    # --- Processar áudio se no modo voz e áudio recebido, e ASR e R2 estiverem disponíveis ---
    if interaction_mode == "voice" and user_audio_media_id and asr_pipeline and s3_client:
        logger.info(f"[whatsapp:{user_id}] Modo voz ativado e áudio Media ID recebido. Baixando e transcrevendo...")
        try:
            # 1. Obter URL de download da mídia da API da Meta
            media_info_url = f"https://graph.facebook.com/v19.0/{user_audio_media_id}"
            media_headers = {
                "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"
            }
            async with httpx.AsyncClient() as client:
                media_response = await client.get(media_info_url, headers=media_headers)
                media_response.raise_for_status()
                media_data = media_response.json()
                download_url = media_data.get("url")
                content_type = media_data.get("mime_type", "application/octet-stream") # Ex: "audio/ogg"

            if not download_url:
                logger.error(f"[whatsapp:{user_id}] Não foi possível obter a URL de download para o Media ID: {user_audio_media_id}")
                return "Desculpe, tive um problema ao obter seu áudio. Poderia tentar novamente?", user_state

            # 2. Baixar o arquivo de áudio
            async with httpx.AsyncClient() as client:
                download_file_response = await client.get(download_url, headers=media_headers)
                download_file_response.raise_for_status()
                audio_data_bytes = download_file_response.content
            
            # 3. Determinar o formato do arquivo para o Whisper
            file_format = content_type.split('/')[-1] if '/' in content_type else "ogg"
            
            # 4. Transcrever o áudio (apenas se não for a tarefa de 'silence')
            if user_state["metadata"]["current_audio_task"] != "silence":
                processed_text = await transcribe_audio(audio_data_bytes, file_format=file_format)
                if not processed_text:
                    return "Desculpe, não consegui entender o que você disse. Poderia repetir?", user_state
                logger.info(f"[whatsapp:{user_id}] Áudio transcrito para: '{processed_text}'")
            else:
                processed_text = "[Áudio de silêncio]" # Não transcrever silêncio

            # 5. Salvar áudio original no R2
            if s3_client and R2_ENDPOINT_URL_PUBLIC:
                # Usar o ID da mensagem para o nome do arquivo para garantir unicidade
                original_audio_filename = f"{message.get('id', uuid.uuid4().hex)}.{file_format}"
                
                # Categoria de áudio (e.g., "received_message", "task_silence", "task_vogal_a")
                audio_category = "received_message"
                if user_state["metadata"]["current_audio_task"]:
                    audio_category = f"task_{user_state['metadata']['current_audio_task']}"

                r2_key_original = f"whatsapp_audios/{user_id}/{audio_category}/{original_audio_filename}"
                
                public_original_audio_url = upload_audio_bytes_to_r2(audio_data_bytes, r2_key_original, content_type)

                if public_original_audio_url:
                    logger.info(f"[whatsapp:{user_id}] Áudio original salvo no R2: {public_original_audio_url}")
                    # Armazenar a URL no estado do usuário sob a tarefa atual
                    if user_state["metadata"]["current_audio_task"]:
                        task_name = user_state["metadata"]["current_audio_task"]
                        user_state["metadata"]["audio_urls"][task_name] = public_original_audio_url
                else:
                    logger.warning(f"[whatsapp:{user_id}] Falha ao obter URL pública para áudio original Media ID: {user_audio_media_id}")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"[whatsapp:{user_id}] Erro HTTP ao baixar mídia da Meta API: {e.response.status_code} - {e.response.text}")
            return "Desculpe, tive um problema ao baixar seu áudio. Poderia tentar novamente?", user_state
        except Exception as e:
            logger.error(f"[whatsapp:{user_id}] Erro ao processar áudio da URL '{user_audio_media_id}': {e}", exc_info=True)
            return "Desculpe, tive um problema ao processar seu áudio. Poderia tentar novamente?", user_state
        
    elif interaction_mode == "voice" and not user_audio_media_id and user_input_text:
        # Modo voz mas recebeu texto - tratar como texto
        response_text = "Detectei que você está no modo de *voz*, mas enviou uma mensagem de *texto*. Processando sua mensagem de texto."
        # Se for um estágio de áudio, informar que esperava áudio
        if current_stage.startswith("awaiting_audio_"):
            task_type = user_state["metadata"]["current_audio_task"]
            response_text += f"\nMas esperava um *áudio* para a tarefa '{task_type.replace('_',' ')}'. Você gostaria de tentar novamente enviando um áudio?"
        return response_text, user_state
    
    elif interaction_mode == "text" and user_audio_media_id:
        # Modo texto mas recebeu áudio - pedir para usar texto
        return "Detectei que você está no modo de *texto*, mas enviou uma mensagem de *voz*. Por favor, use texto para interagir.", user_state
    
    elif not processed_text and not user_audio_media_id:
        # Nenhuma entrada válida
        return "Não entendi sua mensagem. Poderia tentar novamente?", user_state
    
    # --- Lógica do Chatbot Baseada no Estado ---
    response_text = ""

    if current_stage == "initial":
        # Este é o primeiro estágio após /start ou primeira mensagem.
        # Já enviamos a pergunta sobre o modo de interação, agora aguardamos a resposta.
        if processed_text and processed_text.lower() == "texto":
            user_state["interaction_mode"] = "text"
            user_state["stage"] = "awaiting_consent"
            response_text = "Modo de interação definido para *texto*."
            consent_msg = (
                f"Olá, {username}! Sua voz pode nos ajudar a desenvolver novas formas de monitorar a saúde. "
                "As gravações serão usadas anonimamente e exclusivamente para pesquisa científica. "
                "Você gostaria de participar e contribuir com sua voz? (Sim/Não)"
            )
            response_text += "\n" + consent_msg
        elif processed_text and processed_text.lower() == "voz":
            # Verificar se todos os componentes de voz estão operacionais
            if not asr_pipeline:
                response_text = "Desculpe, a *transcrição de voz (ASR)* está desativada devido a um erro na inicialização do modelo Whisper. Por favor, use o modo de texto."
                user_state["interaction_mode"] = "text"
                user_state["stage"] = "awaiting_interaction_mode" # Volta para a escolha, mas agora com texto padrão
                logger.warning(f"[whatsapp:{user_id}] Tentativa de ativar modo voz falhou: ASR não disponível.")
            elif not google_tts_client:
                response_text = "Desculpe, a *geração de voz (TTS)* está desativada devido a um erro nas credenciais ou inicialização do Google Cloud TTS. Por favor, use o modo de texto."
                user_state["interaction_mode"] = "text"
                user_state["stage"] = "awaiting_interaction_mode"
                logger.warning(f"[whatsapp:{user_id}] Tentativa de ativar modo voz falhou: TTS não disponível.")
            elif not s3_client or not R2_ENDPOINT_URL_PUBLIC:
                response_text = "Desculpe, o serviço de *armazenamento de áudio (R2)* não está configurado para o upload de áudios. Por favor, use o modo de texto."
                user_state["interaction_mode"] = "text"
                user_state["stage"] = "awaiting_interaction_mode"
                logger.warning(f"[whatsapp:{user_id}] Tentativa de ativar modo voz falhou: R2 não disponível.")
            else:
                user_state["interaction_mode"] = "voice"
                user_state["stage"] = "awaiting_consent"
                response_text = "Modo de interação definido para *voz*."
                consent_msg = (
                    f"Olá, {username}! Sua voz pode nos ajudar a desenvolver novas formas de monitorar a saúde. "
                    "As gravações serão usadas anonimamente e exclusivamente para pesquisa científica. "
                    "Você gostaria de participar e contribuir com sua voz? (Sim/Não)"
                )
                response_text += "\n" + consent_msg
        else:
            response_text = "Por favor, escolha 'texto' ou 'voz' para definir como vamos interagir."
        
        logger.info(f"[whatsapp:{user_id}] Resposta para modo de interação: '{response_text}'.")

    elif current_stage == "awaiting_consent":
        if processed_text and processed_text.lower() in ["sim", "s"]:
            response_text = "Ótimo! Sua participação é muito importante. Para começar, qual o seu *nome* ou um *apelido* que gostaria de usar para esta pesquisa?"
            user_state["stage"] = "awaiting_name"
            logger.info(f"[whatsapp:{user_id}] Consentimento aceito. Transicionou para 'awaiting_name'.")
        elif processed_text and processed_text.lower() in ["não", "n", "nao"]:
            response_text = "Entendi. Agradecemos seu interesse. Se mudar de ideia, pode digitar /start a qualquer momento."
            user_state["stage"] = "finished" # Não coletou dados, então "finished"
            logger.info(f"[whatsapp:{user_id}] Consentimento recusado. Finalizou.")
        else:
            response_text = "Por favor, responda 'Sim' ou 'Não' para indicar seu consentimento."
            logger.info(f"[whatsapp:{user_id}] Resposta inválida para consentimento: '{processed_text}'.")

    elif current_stage == "awaiting_name":
        if processed_text:
            user_state["metadata"]["name"] = processed_text
            response_text = f"Obrigado, {user_state['metadata']['name']}! Agora, qual a sua *idade* (apenas números)?"
            user_state["stage"] = "awaiting_age"
            logger.info(f"[whatsapp:{user_id}] Nome '{processed_text}' registrado. Transicionou para 'awaiting_age'.")
        else:
            response_text = "Por favor, me diga seu nome ou apelido."

    elif current_stage == "awaiting_age":
        if processed_text and processed_text.isdigit() and 5 <= int(processed_text) <= 120:
            user_state["metadata"]["age"] = int(processed_text)
            response_text = "Idade registrada! Você se considera *Fumante*, *Ex-fumante* ou *Não fumante*?"
            user_state["stage"] = "awaiting_smoking_status"
            logger.info(f"[whatsapp:{user_id}] Idade '{processed_text}' registrada. Transicionou para 'awaiting_smoking_status'.")
        else:
            response_text = "Por favor, digite sua idade em números (entre 5 e 120 anos)."

    elif current_stage == "awaiting_smoking_status":
        if processed_text and processed_text.lower() in ["fumante", "ex-fumante", "não fumante", "nao fumante"]:
            user_state["metadata"]["smoking_status"] = processed_text.lower()
            response_text = "Obrigado. Você tem algum *diagnóstico de saúde* relevante (ex: Parkinson, Diabetes, Hipertensão, Saudável)?"
            user_state["stage"] = "awaiting_diagnosis"
            logger.info(f"[whatsapp:{user_id}] Status de fumante '{processed_text}' registrado. Transicionou para 'awaiting_diagnosis'.")
        else:
            response_text = "Por favor, responda com 'Fumante', 'Ex-fumante' ou 'Não fumante'."

    elif current_stage == "awaiting_diagnosis":
        if processed_text:
            user_state["metadata"]["diagnosis"] = processed_text
            response_text = "Entendido. Em uma escala de 1 a 5 (onde 1 é muito calmo e 5 é muito estressado), como você se sente *emocionalmente* agora?"
            user_state["stage"] = "awaiting_emotional_state"
            logger.info(f"[whatsapp:{user_id}] Diagnóstico '{processed_text}' registrado. Transicionou para 'awaiting_emotional_state'.")
        else:
            response_text = "Por favor, digite seu diagnóstico de saúde (ou 'Saudável' se não tiver)."

    elif current_stage == "awaiting_emotional_state":
        if processed_text and processed_text.isdigit() and 1 <= int(processed_text) <= 5:
            user_state["metadata"]["emotional_state"] = int(processed_text)
            response_text = "Quase lá! Por favor, descreva o *ambiente* onde você está gravando agora: (Ex: Silencioso, Pouco ruído, Barulhento)"
            user_state["stage"] = "awaiting_environment"
            logger.info(f"[whatsapp:{user_id}] Estado emocional '{processed_text}' registrado. Transicionou para 'awaiting_environment'.")
        else:
            response_text = "Por favor, use um número de 1 a 5 para descrever seu estado emocional."

    elif current_stage == "awaiting_environment":
        if processed_text:
            user_state["metadata"]["environment"] = processed_text
            # Definindo a fila de tarefas de áudio com as novas tarefas
            user_state["tasks_queue"] = ["silence", "vogal_a", "vogal_e", "vogal_i", "vogal_o", "fricativo_s", "fricativo_z", "sentence_read"]
            response_text = (
                "Perfeito! Seus dados iniciais foram registrados. Agora, vamos para a parte mais importante: a sua voz. "
                "Por favor, encontre um local o mais silencioso possível. "
                "Quando estiver pronto, vamos começar."
            )
            user_state["stage"] = "requesting_first_audio_task" # Novo estágio para a transição para a primeira tarefa de áudio
            logger.info(f"[whatsapp:{user_id}] Ambiente '{processed_text}' registrado. Iniciando fila de tarefas de áudio.")
        else:
            response_text = "Por favor, descreva o ambiente da gravação."

    # Lógica para solicitar e processar as tarefas de áudio
    elif current_stage.startswith("awaiting_audio_"):
        # Se recebemos um áudio para uma tarefa específica
        if user_audio_media_id:
            # O processamento e upload do áudio para o R2 já ocorreu no início de process_whatsapp_message
            # e a URL do áudio da tarefa já foi salva em user_state["metadata"]["audio_urls"]
            task_type = user_state["metadata"]["current_audio_task"]
            response_text = f"Áudio da tarefa '{task_type.replace('_',' ')}' recebido e salvo! Obrigado."
            
            # Se há mais tarefas na fila, pede a próxima
            if user_state["tasks_queue"]:
                next_task = user_state["tasks_queue"].pop(0) # Remove a próxima tarefa da fila
                user_state["metadata"]["current_audio_task"] = next_task # Define a tarefa atual
                user_state["stage"] = f"awaiting_audio_{next_task}" # Atualiza o estágio
                response_text += "\n" + get_task_prompt(next_task)
            else:
                # Todas as tarefas foram concluídas
                response_text += "\n" + get_completion_message(user_state)
                user_state["stage"] = "finished_tasks" # Novo estágio para indicar que as tarefas foram concluídas
                user_state["metadata"]["current_audio_task"] = None # Reseta a tarefa atual

        else: # Recebeu texto no estágio de áudio
            task_type = user_state["metadata"]["current_audio_task"]
            response_text = f"Não recebi um áudio para a tarefa '{task_type.replace('_',' ')}'. Por favor, grave e envie o áudio solicitado."

    elif current_stage == "requesting_first_audio_task":
        # Este estágio é uma transição para pedir a primeira tarefa.
        # Ele ocorre logo após o registro do ambiente.
        if user_state["tasks_queue"]:
            next_task = user_state["tasks_queue"].pop(0) # Remove a primeira tarefa da fila
            user_state["metadata"]["current_audio_task"] = next_task # Define a tarefa atual
            user_state["stage"] = f"awaiting_audio_{next_task}" # Atualiza o estágio para aguardar o áudio desta tarefa
            response_text = get_task_prompt(next_task)
            logger.info(f"[whatsapp:{user_id}] Solicitando primeira tarefa de áudio: '{next_task}'.")
        else:
            response_text = get_completion_message(user_state)
            user_state["stage"] = "finished_tasks" # Nenhuma tarefa para iniciar

    elif current_stage == "finished_tasks": # Novo estágio após todas as tarefas de áudio serem concluídas
        if processed_text and processed_text.lower() == "/start":
            user_states[state_key] = await init_user_state(user_id, username)
            return "Olá! Bem-vindo ao Kurumim. Como você gostaria de interagir? Por *texto* ou por *voz*?", user_states[state_key]
        else:
            response_text = "Sua contribuição está completa! Muito obrigado por ajudar o Kurumim. Seus dados e áudios foram salvos com sucesso. Se quiser começar de novo, digite /start."
            user_state["stage"] = "finished" # Finaliza completamente

    elif current_stage == "finished": # Estado final de uma sessão concluída (seja por recusa de consentimento ou conclusão de tarefas)
        if processed_text and processed_text.lower() == "/start":
            user_states[state_key] = await init_user_state(user_id, username)
            return "Olá! Bem-vindo ao Kurumim. Como você gostaria de interagir? Por *texto* ou por *voz*?", user_states[state_key]
        else:
            response_text = "Sua sessão já foi concluída. Muito obrigado por sua participação. Para iniciar uma nova sessão, digite /start."


    user_states[state_key] = user_state
    logger.debug(f"[whatsapp:{user_id}] Estado final após a lógica: {user_states[state_key]}")
    return response_text, user_state


def get_task_prompt(task_type: str) -> str:
    """Retorna o texto da instrução para cada tarefa de áudio."""
    if task_type == "silence":
        return "Para a primeira gravação, por favor, inspire fundo e grave cerca de 5 segundos de *silêncio* no ambiente onde você está. Isso nos ajuda a analisar o ruído de fundo. Pode começar quando estiver pronto!"
    elif task_type == "vogal_a":
        return "Ótimo! Inspire fundo. Quando estiver pronto, por favor, diga 'Aaaaaa' por cerca de 5 segundos em um tom normal e envie o áudio."
    elif task_type == "vogal_e":
        return "Excelente. Agora, vamos fazer o mesmo com a vogal 'E'. Inspire fundo. Quando estiver pronto, por favor, diga 'Eeeee' por cerca de 5 segundos em um tom normal e envie o áudio."
    elif task_type == "vogal_i":
        return "Perfeito! Para a próxima, inspire fundo. Quando estiver pronto, por favor, diga 'Iiiiiii' por cerca de 5 segundos em um tom normal e envie o áudio."
    elif task_type == "vogal_o":
        return "Quase lá! Para a vogal 'O', inspire fundo. Quando estiver pronto, por favor, diga 'Oooooo' por cerca de 5 segundos em um tom normal e envie o áudio."
    elif task_type == "fricativo_s":
        return "Muito bem! Agora, faremos um som diferente. Inspire fundo e, quando estiver pronto, emita o som de 'Sssssss' de forma contínua pelo máximo de tempo que conseguir, e envie o áudio."
    elif task_type == "fricativo_z":
        return "Perfeito. Agora, faremos o mesmo com o som de 'Z'. Inspire fundo e, quando estiver pronto, emita o som de 'Zzzzzzz' de forma contínua pelo máximo de tempo que conseguir, e envie o áudio."
    elif task_type == "sentence_read":
        # Exemplo de frase foneticamente balanceada em português (pode ser ajustada)
        sentence = "O peito do pé do Pedro é preto. A aranha arranha a jarra."
        return f"Para a última tarefa, por favor, inspire fundo e leia a seguinte frase em voz alta de forma natural: \"{sentence}\". Envie o áudio quando terminar."
    return "Tarefa de áudio desconhecida. Por favor, tente novamente."

def get_completion_message(user_state: dict) -> str:
    """Gera a mensagem final de conclusão com os dados coletados."""
    collected_data = user_state['metadata']
    
    # Formata as URLs dos áudios coletados
    audio_urls_formatted = []
    for task, url in collected_data.get("audio_urls", {}).items():
        audio_urls_formatted.append(f"- {task.replace('_', ' ').title()}: {url}")
    audio_list_str = "\n".join(audio_urls_formatted) if audio_urls_formatted else "Nenhum áudio de tarefa coletado."

    return (
        "Fantástico! Coletamos todas as suas amostras de voz. Sua contribuição é extremamente valiosa para a pesquisa de saúde do Kurumim."
        "\nMuito obrigado por participar! Seus dados e áudios foram salvos com sucesso."
        "\n\nDetalhes da sua sessão (anonimizados para pesquisa):"
        f"\nNome/ID: {collected_data.get('name', 'N/A')}"
        f"\nIdade: {collected_data.get('age', 'N/A')}"
        f"\nDiagnóstico: {collected_data.get('diagnosis', 'N/A')}"
        f"\nStatus de Fumante: {collected_data.get('smoking_status', 'N/A').title()}"
        f"\nEstado Emocional (1-5): {collected_data.get('emotional_state', 'N/A')}"
        f"\nAmbiente da Gravação: {collected_data.get('environment', 'N/A')}"
        f"\n\nÁudios de Tarefa Coletados (R2):\n{audio_list_str}"
        "\n\nPara iniciar uma nova sessão, digite /start."
    )




@api_app.get("/")
async def root():
    return {"message": "Kurumim bot server is alive!"}

# Endpoint para o Webhook do WhatsApp Business Cloud API (Meta)
@api_app.get("/whatsapp/webhook")
async def whatsapp_verify_webhook(request: Request):
    """
    Endpoint de verificação do Webhook da Meta.
    A Meta envia uma requisição GET para verificar a URL do seu webhook.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == WEBHOOK_VERIFY_TOKEN:
            logger.info("WEBHOOK_VERIFY_TOKEN verificado com sucesso. Assinando.")
            return Response(content=challenge, media_type="text/plain", status_code=200)
        else:
            logger.warning("Falha na verificação do token do Webhook. Token incorreto.")
            raise HTTPException(status_code=403, detail="Falha na verificação do token.")
    logger.warning("Parâmetros de verificação ausentes na requisição GET do webhook.")
    raise HTTPException(status_code=400, detail="Parâmetros de verificação ausentes.")


@api_app.post("/whatsapp/webhook")
async def whatsapp_receive_message(request: Request):
    """
    Endpoint para receber mensagens do WhatsApp Business Cloud API.
    A Meta envia requisições POST com o payload da mensagem.
    """
    try:
        payload = await request.json()
        logger.info(f"Payload do WhatsApp (Meta API) recebido: {json.dumps(payload, indent=2)}")

        # A estrutura do payload da Meta é complexa. Precisamos navegar nela.
        # Exemplo de estrutura: {"entry": [{"changes": [{"value": {"messages": [...]}}]}]}
        if "entry" not in payload or not payload["entry"]:
            logger.warning("Payload inválido: 'entry' ausente ou vazio.")
            return Response(status_code=200) # Sempre responda 200 OK para evitar reenvios

        for entry in payload["entry"]:
            if "changes" not in entry or not entry["changes"]: continue
            for change in entry["changes"]:
                if "value" not in change or change["field"] != "messages": continue
                value = change["value"]

                # Verificar se é uma mensagem de usuário e não um status ou outra notificação
                if "messages" not in value or not value["messages"]: continue
                for message in value["messages"]:
                    # Ignorar mensagens de sistema ou de outro tipo que não queremos processar
                    if message["type"] not in ["text", "audio"]:
                        logger.info(f"Ignorando mensagem do tipo: {message['type']}")
                        continue

                    from_number = message["from"]
                    # O nome do usuário não vem diretamente no webhook. Usamos o número como ID e um nome genérico.
                    # Para obter o nome real, você precisaria usar a API de perfil do usuário (requer permissão adicional
                    # e configuração de usuários Meta).
                    username = from_number # Pode ser substituído por um nome de verdade se você implementar a busca no perfil

                    user_text = None
                    user_audio_media_id = None # Para a Meta API, o áudio_url é o ID da mídia para posterior download

                    if message["type"] == "text":
                        user_text = message["text"]["body"]
                        logger.info(f"[{from_number}] Mensagem de Texto: '{user_text}'")
                    elif message["type"] == "audio":
                        # Para áudios, o payload vem com um 'id' da mídia.
                        # Precisamos chamar a API da Meta para obter a URL de download real.
                        user_audio_media_id = message["audio"]["id"]
                        logger.info(f"[{from_number}] Mensagem de Áudio. Media ID: {user_audio_media_id}")

                    # Processa a mensagem e envia a resposta de forma assíncrona
                    # Usa asyncio.create_task para não bloquear o loop de eventos da FastAPI
                    # enquanto as funções assíncronas de processamento e envio estão rodando.
                    asyncio.create_task(
                        handle_incoming_whatsapp_message(from_number, username, user_text, user_audio_media_id, message)
                    )

    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON do Webhook: {e}")
        raise HTTPException(status_code=400, detail="Requisição JSON inválida.")
    except Exception as e:
        logger.error(f"Erro no processamento do webhook do WhatsApp: {e}", exc_info=True)
        # Não lançar HTTPException aqui, apenas registrar o erro e retornar 200
        # para evitar que a Meta tente reenviar o webhook várias vezes.
    
    return Response(status_code=200) # Meta espera 200 OK para evitar reenvios

async def handle_incoming_whatsapp_message(from_number: str, username: str, user_text: str | None, user_audio_media_id: str | None, raw_message_payload: dict):
    """
    Função auxiliar para processar a mensagem do WhatsApp e enviar a resposta.
    Rodada como uma tarefa assíncrona para não bloquear o endpoint do webhook.
    """
    try:
        response_text, new_state = await process_whatsapp_message(from_number, username, user_text, user_audio_media_id, raw_message_payload, )
        await send_whatsapp_response(from_number, new_state, response_text)
    except Exception as e:
        logger.error(f"Erro ao manipular mensagem do WhatsApp para {from_number}: {e}", exc_info=True)
        # Tentar enviar uma mensagem de erro genérica ao usuário
        try:
            error_message = "Desculpe, ocorreu um erro interno. Por favor, tente novamente mais tarde."
            current_user_state = user_states.get(get_user_state_key("whatsapp", from_number), {})
            await send_whatsapp_response(from_number, current_user_state, error_message)
        except Exception as e_send_error:
            logger.error(f"Falha ao enviar mensagem de erro ao usuário {from_number}: {e_send_error}")

# --- Execução do Servidor FastAPI (para quando rodar main.py diretamente) ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Iniciando servidor FastAPI na porta {port}...")
    # host="0.0.0.0" permite que o servidor seja acessível externamente (necessário para ngrok/deploy)
    #uvicorn.run(api_app, host="0.0.0.0", port=port)