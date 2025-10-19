from flask import Flask, request, jsonify, Response
import requests
import json
import time

app = Flask(__name__)

# NVIDIA NIM Configuration
NVIDIA_API_KEY = "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"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Model mapping
MODEL_MAPPING = {
    "gpt-3.5-turbo": "deepseek-ai/deepseek-v3.1-terminus",
    "gpt-4": "deepseek-ai/deepseek-v3.1-terminus",
    "gpt-4-turbo": "deepseek-ai/deepseek-v3.1-terminus",
    "deepseek-v3.1": "deepseek-ai/deepseek-v3.1-terminus"
}

def convert_to_nvidia_format(openai_request):
    model = openai_request.get("model", "gpt-3.5-turbo")
    nvidia_model = MODEL_MAPPING.get(model, "deepseek-ai/deepseek-v3.1-terminus")
    
    nvidia_request = {
        "model": nvidia_model,
        "messages": openai_request.get("messages", []),
        "temperature": openai_request.get("temperature", 0.7),
        "top_p": openai_request.get("top_p", 1.0),
        "max_tokens": openai_request.get("max_tokens", 1024),
        "stream": openai_request.get("stream", False)
    }
    
    return nvidia_request

def convert_to_openai_format(nvidia_response, model):
    openai_response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": nvidia_response["choices"][0]["message"]["content"]
                },
                "finish_reason": nvidia_response["choices"][0].get("finish_reason", "stop")
            }
        ],
        "usage": nvidia_response.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
    }
    
    return openai_response

def convert_stream_chunk(nvidia_chunk, model):
    openai_chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": nvidia_chunk["choices"][0].get("delta", {}),
                "finish_reason": nvidia_chunk["choices"][0].get("finish_reason")
            }
        ]
    }
    
    return openai_chunk

@app.route('/')
def index():
    return jsonify({
        "status": "ok",
        "message": "NVIDIA NIM Proxy is running",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    })

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
        
    try:
        openai_request = request.json
        nvidia_request = convert_to_nvidia_format(openai_request)
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if nvidia_request.get("stream", False):
            def generate():
                response = requests.post(
                    f"{NVIDIA_BASE_URL}/chat/completions",
                    headers=headers,
                    json=nvidia_request,
                    stream=True
                )
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data.strip() == '[DONE]':
                                yield f"data: [DONE]\n\n"
                            else:
                                try:
                                    nvidia_chunk = json.loads(data)
                                    openai_chunk = convert_stream_chunk(
                                        nvidia_chunk, 
                                        openai_request.get("model", "gpt-3.5-turbo")
                                    )
                                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                                except json.JSONDecodeError:
                                    continue
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=nvidia_request
            )
            
            if response.status_code == 200:
                nvidia_response = response.json()
                openai_response = convert_to_openai_format(
                    nvidia_response, 
                    openai_request.get("model", "gpt-3.5-turbo")
                )
                return jsonify(openai_response)
            else:
                return jsonify({"error": response.text}), response.status_code
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    models = {
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nvidia-nim"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nvidia-nim"
            },
            {
                "id": "deepseek-v3.1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nvidia-nim"
            }
        ]
    }
    return jsonify(models)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# For Vercel serverless
def handler(request):
    with app.request_context(request.environ):
        try:
            rv = app.preprocess_request()
            if rv is None:
                rv = app.dispatch_request()
        except Exception as e:
            rv = app.handle_user_exception(e)
        response = app.make_response(rv)
        return app.process_response(response)
