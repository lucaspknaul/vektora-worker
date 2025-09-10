# vektora-worker

Rodando em ~4s.

# Possível Otimização

Usar o vektora manager para baixar modelos direto do hugginface no network-volume e apontar worker para network volume.

# Desenvolvimento local

## Instalação

Clonar esse repositório.

Criar venv com requirements.txt

Comentar código de produção e descomentar de testes

## Execução

Dentro do venv:

```python
python handler.py
```

# Build

## Build
```bash
sudo docker build --platform linux/amd64 -t lucaspknaul/vektora-worker:v0.1.0 .
```

## Run locally
```bash
sudo docker run -it lucaspknaul/vektora-worker:v0.1.0
```

## Push
```bash
# Log in to Docker Hub
sudo docker login

# Push the image
sudo docker push lucaspknaul/vektora-worker:v0.1.0
```