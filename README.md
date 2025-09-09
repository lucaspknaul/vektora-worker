# vektora-worker

# Idéia

Custos de execução de serverless-comfyui-worker:
- comfyui-worker images: 1
- comfyui-worker execution time: 93s
- comfyui-worker serverless price: $0.032
- comfyui-worker GPU price: $0.032

comfyui-worker está muito caro por precisar inicializar todo o comfyui a cada restart.

Iniciando desenvolvimento do meu próprio backend diretamente em python.


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
sudo docker build --platform linux/amd64 -t lucaspknaul/vektora-worker:v0.0.1 .
```

## Run locally
```bash
sudo docker run -it lucaspknaul/vektora-worker:v0.0.1
```

## Push
```bash
# Log in to Docker Hub
sudo docker login

# Push the image
sudo docker push lucaspknaul/vektora-worker:v0.0.1
```