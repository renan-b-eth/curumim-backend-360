import os
import asyncio
import logging
import json
from dotenv import load_dotenv
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from starlette.responses import Response

from google.cloud import texttospeech
from google.oauth2 import service_account

# Importações para Whisper e R2, se você as estiver usando
from transformers import pipeline
import torch
import soundfile as sf
import io
import boto3
from botocore.exceptions import NoCredentialsError

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Carregar variáveis de ambiente ---
load_dotenv()

# --- Instância FastAPI ---
api_app = FastAPI() # A instância do FastAPI é chamada de 'api_app' para ser reconhecida pelo uvicorn main:api_app

# --- Variáveis de ambiente ---
WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

# --- Configuração Whisper ASR ---
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-small")
asr_pipeline = None
if WHISPER_MODEL_NAME:
    try:
        # Tenta carregar o modelo ASR. Se não houver GPU, usará a CPU.
        device = 0 if torch.cuda.is_available() else -1
        asr_pipeline = pipeline("automatic-speech-recognition", model=WHISPER_MODEL_NAME, device=device)
        logger.info(f"Modelo ASR '{WHISPER_MODEL_NAME}' carregado com sucesso no {'cuda' if device == 0 else 'cpu'}.")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo ASR '{WHISPER_MODEL_NAME}': {e}")
        asr_pipeline = None
else:
    logger.warning("Variável de ambiente WHISPER_MODEL_NAME não definida. A transcrição de áudio não estará disponível.")


# --- Configuração Google Cloud Text-to-Speech ---
google_tts_client = None

### INÍCIO DA MODIFICAÇÃO PARA RENDER (Google Cloud TTS Credenciais) ###
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
    try:
        creds_json_content = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        creds_json = json.loads(creds_json_content)
        credentials = service_account.Credentials.from_service_account_info(creds_json)
        google_tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        logger.info("Cliente Google Cloud TTS inicializado com credenciais JSON da variável de ambiente.")
    except Exception as e:
        logger.error(f"Erro ao inicializar cliente Google Cloud TTS com JSON da variável de ambiente: {e}")
        google_tts_client = None
elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    # Esta parte é para compatibilidade se você ainda rodar localmente com o arquivo JSON,
    # mas no Render, o JSON_CONTENT_VAR é o que será usado.
    if os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
        try:
            credentials = service_account.Credentials.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
            google_tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
            logger.info("Cliente Google Cloud TTS inicializado com Conta de Serviço do arquivo.")
        except Exception as e:
            logger.error(f"Erro ao inicializar cliente Google Cloud TTS com arquivo de credenciais: {e}")
            google_tts_client = None
    else:
        logger.warning(f"Arquivo de credenciais do Google Cloud TTS não encontrado no caminho: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
else:
    logger.warning("Nenhuma credencial Google Cloud TTS encontrada. A funcionalidade de voz pode estar desativada.")
### FIM DA MODIFICAÇÃO PARA RENDER ###


# --- Configuração Cloudflare R2 ---
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

r2_client = None
if R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_ACCOUNT_ID and R2_BUCKET_NAME:
    try:
        r2_client = boto3.client(
            's3',
            endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto' # R2 usa 'auto' ou qualquer string, não uma região AWS real
        )
        # Testar a conexão listando objetos (pode ser ajustado para um teste mais leve)
        # r2_client.list_objects_v2(Bucket=R2_BUCKET_NAME, MaxKeys=1)
        logger.info("Conectado ao Cloudflare R2 para armazenamento de áudios.")
    except NoCredentialsError:
        logger.error("Credenciais do Cloudflare R2 não configuradas corretamente.")
        r2_client = None
    except Exception as e:
        logger.error(f"Erro ao conectar ao Cloudflare R2: {e}")
        r2_client = None
else:
    logger.warning("Credenciais Cloudflare R2 incompletas ou ausentes. O armazenamento de áudio pode não funcionar.")


# --- Armazenamento de estado da sessão (simples, em memória) ---
# Em um ambiente de produção real, você usaria um banco de dados (Redis, Firestore, DynamoDB)
# para armazenar o estado de forma persistente e escalável.
session_states: Dict[str, Dict[str, Any]] = {}

# Estados possíveis
INITIAL_STATE = "initial"
WAITING_FOR_INTERACTION_MODE = "waiting_for_interaction_mode"
INTERACTION_MODE_TEXT = "text"
INTERACTION_MODE_VOICE = "voice"


# --- Funções Auxiliares ---

async def transcribe_audio(audio_data: bytes) -> Optional[str]:
    if not asr_pipeline:
        logger.error("ASR pipeline não inicializado. Não é possível transcrever áudio.")
        return None

    # O pipeline espera um arquivo ou objeto tipo arquivo
    # soundfile.read é usado para carregar o áudio e garantir o formato correto (array numpy e sample_rate)
    try:
        audio_array, sampling_rate = sf.read(io.BytesIO(audio_data))
        # O pipeline Whisper geralmente espera 16kHz, então resampling pode ser necessário
        # Se seu áudio já estiver em 16kHz, esta etapa pode ser omitida ou otimizada
        
        # O pipeline pode aceitar diretamente o array numpy
        # print(f"Audio array shape: {audio_array.shape}, Sample rate: {sampling_rate}") # Para debug
        transcription = asr_pipeline(audio_array.copy(), sampling_rate=sampling_rate, chunk_length_s=30, return_timestamps=True)
        logger.info(f"Transcrição: {transcription['text']}")
        return transcription['text']
    except Exception as e:
        logger.error(f"Erro ao transcrever áudio: {e}")
        return None

async def synthesize_speech(text: str) -> Optional[bytes]:
    if not google_tts_client:
        logger.error("Google TTS client não inicializado. Não é possível sintetizar fala.")
        return None

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="pt-BR",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE # ou MALE, NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.OGG_OPUS # OGG_OPUS é bom para WhatsApp
    )

    try:
        response = google_tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content
    except Exception as e:
        logger.error(f"Erro ao sintetizar fala: {e}")
        return None

async def upload_audio_to_r2(audio_bytes: bytes, filename: str) -> Optional[str]:
    if not r2_client:
        logger.error("Cliente R2 não inicializado. Não é possível fazer upload de áudio.")
        return None
    try:
        r2_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=filename,
            Body=audio_bytes,
            ContentType='audio/ogg' # Ou o tipo de conteúdo correto do seu áudio
        )
        public_url = f"https://pub-{R2_ACCOUNT_ID}.r2.dev/{filename}" # URL pública do R2
        logger.info(f"Áudio enviado para R2: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Erro ao fazer upload para R2: {e}")
        return None


async def send_whatsapp_message(to_number: str, text: Optional[str] = None, audio_url: Optional[str] = None):
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    
    payload: Dict[str, Any] = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
    }

    if text:
        payload["type"] = "text"
        payload["text"] = {"body": text}
    elif audio_url:
        payload["type"] = "audio"
        payload["audio"] = {"link": audio_url}
    else:
        logger.error("Tentativa de enviar mensagem WhatsApp sem texto ou URL de áudio.")
        return

    try:
        # Usar httpx para requisições assíncronas
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status() # Lança exceção para erros HTTP
            logger.info(f"Mensagem WhatsApp (Meta API) enviada com sucesso para {to_number}. Status: {response.status_code}")
            logger.info(f"Resposta da API: {response.json()}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Erro HTTP ao enviar mensagem WhatsApp (Meta API) para {to_number}: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Erro de requisição ao enviar mensagem WhatsApp (Meta API) para {to_number}: {e}")
    except Exception as e:
        logger.error(f"Erro inesperado ao enviar mensagem WhatsApp (Meta API) para {to_number}: {e}")


# --- Lógica do Bot ---
async def process_whatsapp_message(
    from_number: str,
    username: str,
    user_input: Optional[str] = None,
    user_audio_media_id: Optional[str] = None,
    raw_message_payload: Optional[Dict[str, Any]] = None # Adicionado para compatibilidade, mas não usado diretamente aqui
) -> tuple[str, str]: # Retorna (response_text, new_state)
    
    user_state = session_states.get(from_number, {"state": INITIAL_STATE, "interaction_mode": None})
    current_state = user_state["state"]
    interaction_mode = user_state["interaction_mode"]
    
    response_text = ""
    new_state = current_state

    # --- Lógica de Reset/Início ---
    if user_input and user_input.lower() == "/start":
        response_text = "Olá! Bem-vindo ao Kurumim. Como você gostaria de interagir? Por *texto* ou por *voz*?"
        new_state = WAITING_FOR_INTERACTION_MODE
        interaction_mode = None # Reseta o modo de interação
        session_states[from_number] = {"state": new_state, "interaction_mode": interaction_mode}
        logger.info(f"Estado inicializado/resetado para user {from_number} (key: whatsapp_{from_number}).")
        return response_text, new_state

    # --- Lógica de Seleção de Modo de Interação ---
    if current_state == WAITING_FOR_INTERACTION_MODE:
        if user_input and user_input.lower() == "texto":
            interaction_mode = INTERACTION_MODE_TEXT
            response_text = "Ótimo! Estamos no modo texto. Como posso ajudar você hoje?"
            new_state = INTERACTION_MODE_TEXT # O estado agora reflete o modo de interação
            session_states[from_number] = {"state": new_state, "interaction_mode": interaction_mode}
            return response_text, new_state
        elif user_input and user_input.lower() == "voz":
            interaction_mode = INTERACTION_MODE_VOICE
            response_text = "Excelente! Estamos no modo voz. Envie-me uma mensagem de áudio ou diga algo."
            new_state = INTERACTION_MODE_VOICE # O estado agora reflete o modo de interação
            session_states[from_number] = {"state": new_state, "interaction_mode": interaction_mode}
            return response_text, new_state
        else:
            response_text = "Por favor, digite 'texto' ou 'voz' para escolher seu modo de interação."
            # O estado permanece WAITING_FOR_INTERACTION_MODE
            return response_text, current_state

    # --- Lógica de Interação Geral ---
    if interaction_mode == INTERACTION_MODE_TEXT:
        if user_input:
            if "olá" in user_input.lower():
                response_text = f"Olá, {username}! Em que posso ser útil no modo texto?"
            elif "como vai" in user_input.lower():
                response_text = "Vou muito bem, obrigado! Estou pronto para ajudar. O que você gostaria de saber ou fazer?"
            else:
                response_text = f"Você disse por texto: '{user_input}'. Estou aprendendo, mas ainda não consigo processar isso complexamente. Tente algo mais simples ou pergunte 'ajuda'."
            new_state = INTERACTION_MODE_TEXT # Permanece no modo texto
        else:
            response_text = "Por favor, envie uma mensagem de texto."
            new_state = INTERACTION_MODE_TEXT

    elif interaction_mode == INTERACTION_MODE_VOICE:
        if user_audio_media_id:
            logger.info(f"Recebido ID de mídia de áudio: {user_audio_media_id}")
            # Aqui você faria o download do áudio, transcreveria e processaria.
            # Por enquanto, vamos apenas confirmar o recebimento.
            response_text = "Recebi sua mensagem de voz. A funcionalidade de processamento de voz está em desenvolvimento. Por enquanto, só posso confirmar que recebi seu áudio."
            new_state = INTERACTION_MODE_VOICE # Permanece no modo voz
        elif user_input: # Se o usuário enviar texto enquanto está no modo voz
            response_text = "Você está no modo voz. Por favor, envie uma mensagem de áudio, ou digite '/start' para mudar o modo."
            new_state = INTERACTION_MODE_VOICE
        else:
            response_text = "Por favor, envie uma mensagem de voz."
            new_state = INTERACTION_MODE_VOICE

    else: # Estado inicial ou desconhecido, ou se interaction_mode não foi setado
        response_text = "Olá! Bem-vindo ao Kurumim. Como você gostaria de interagir? Por *texto* ou por *voz*?"
        new_state = WAITING_FOR_INTERACTION_MODE

    session_states[from_number] = {"state": new_state, "interaction_mode": interaction_mode}
    return response_text, new_state


# --- Rotas FastAPI ---

@api_app.get("/")
async def root():
    return {"message": "Kurumim Bot está online!"}

@api_app.get("/whatsapp/webhook")
async def verify_webhook(request: Request):
    """
    Verifica o webhook para o WhatsApp Business API.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == WEBHOOK_VERIFY_TOKEN:
            logger.info("Webhook verificado com sucesso!")
            return Response(content=challenge, media_type="text/plain")
        else:
            raise HTTPException(status_code=403, detail="Falha na verificação. Token inválido.")
    else:
        raise HTTPException(status_code=400, detail="Parâmetros de verificação ausentes.")


@api_app.post("/whatsapp/webhook")
async def handle_incoming_whatsapp_message(request: Request):
    """
    Manipula mensagens recebidas do WhatsApp.
    """
    payload = await request.json()
    logger.info(f"Payload do WhatsApp (Meta API) recebido: {json.dumps(payload, indent=2)}")

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") == "messages":
                value = change.get("value", {})
                if "messages" in value:
                    for message in value["messages"]:
                        from_number = message["from"]  # Número do remetente
                        message_type = message["type"]
                        
                        # Tentar obter o username se disponível (pode não vir em todas as mensagens)
                        username = from_number # Fallback para o número se o nome não for encontrado
                        if "contacts" in value and value["contacts"]:
                            username = value["contacts"][0].get("profile", {}).get("name", from_number)
                        
                        user_text = None
                        user_audio_media_id = None

                        if message_type == "text":
                            user_text = message["text"]["body"]
                            logger.info(f"[{from_number}] Mensagem de Texto: '{user_text}'")
                        elif message_type == "audio":
                            user_audio_media_id = message["audio"]["id"]
                            logger.info(f"[{from_number}] Mensagem de Áudio (ID): '{user_audio_media_id}'")
                        else:
                            logger.info(f"[{from_number}] Tipo de mensagem não suportado: {message_type}")
                            await send_whatsapp_message(from_number, text="Desculpe, só consigo processar mensagens de texto e áudio por enquanto.")
                            continue # Pula para a próxima mensagem

                        try:
                            response_text, new_state = await process_whatsapp_message(from_number, username, user_text, user_audio_media_id, payload)
                            session_states[from_number]["state"] = new_state # Atualiza o estado da sessão

                            # Lógica para responder de acordo com o modo de interação
                            interaction_mode = session_states[from_number].get("interaction_mode")

                            if interaction_mode == INTERACTION_MODE_VOICE and google_tts_client:
                                logger.info(f"Sintetizando resposta de voz para {from_number}: '{response_text}'")
                                audio_bytes = await synthesize_speech(response_text)
                                if audio_bytes:
                                    filename = f"response_{from_number}_{asyncio.current_task().get_name()}.ogg"
                                    audio_url = await upload_audio_to_r2(audio_bytes, filename)
                                    if audio_url:
                                        await send_whatsapp_message(from_number, audio_url=audio_url)
                                    else:
                                        logger.error(f"Falha ao obter URL pública do áudio para {from_number}. Enviando texto.")
                                        await send_whatsapp_message(from_number, text=response_text)
                                else:
                                    logger.error(f"Falha ao sintetizar áudio para {from_number}. Enviando texto.")
                                    await send_whatsapp_message(from_number, text=response_text)
                            else: # Modo texto ou modo voz com falha na síntese/upload
                                await send_whatsapp_message(from_number, text=response_text)

                        except Exception as e:
                            logger.error(f"Erro ao manipular mensagem do WhatsApp para {from_number}: {e}", exc_info=True)
                            error_response_text = "Desculpe, ocorreu um erro interno. Por favor, tente novamente mais tarde."
                            # Tentar enviar uma mensagem de erro em texto, pois a lógica de modo pode ter falhado
                            await send_whatsapp_message(from_number, text=error_response_text)

    return Response(status_code=200)