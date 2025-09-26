#!/usr/bin/env bash
# Inicia o servidor uvicorn com Gunicorn (melhor para produção)
# O Render setará a variável de ambiente PORT automaticamente
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT