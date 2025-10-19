from flask import Flask, request, jsonify, Response
import requests
import json
import os
from datetime import datetime

app = Flask(__name__)

# Configuration
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', 'your-nvidia-api-key-here')
NVIDIA_BASE_URL = os.environ.get('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')

# Model mapping (OpenAI model names to NVIDIA NIM model names)
MODEL_MAPPING = {
    'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
    'gpt-4': 'meta/llama-3.1-70b-instruct',
    'gpt-4-turbo': 'meta/llama-3.1-70b-instruct',
    'gpt-4o': 'meta/llama-3.1-405b-instruct',
}

def convert_to_nvidia_format(openai_request):
    """Convert OpenAI format to NVIDIA NIM format"""
    nvidia_request = {
        'model': MODEL_MAPPING.get(openai_request.get('model', 'gpt-3.5-turbo'), 
                                   'meta/llama-3.1-8b-instruct'),
        'messages': openai_request.get('messages', []),
        'temperature': openai_request.get('temperature', 1.0),
        'top_p': openai_request.get('top_p', 1.0),
        'max_tokens': openai_request.get('max_tokens', 1024),
        'stream': openai_request.get('stream', False)
    }
    return nvidia_request

def convert_to_openai_format(nvidia_response, model):
    """Convert NVIDIA NIM response to OpenAI format"""
    openai_response = {
        'id': f"chatcmpl-{nvidia_response.get('id', 'nvidia')}",
        'object': 'chat.completion',
        'created': int(datetime.now().timestamp()),
        'model': model,
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': nvidia_response['choices'][0]['message']['content']
                },
                'finish_reason': nvidia_response['choices'][0].get('finish_reason', 'stop')
            }
        ],
        'usage': nvidia_response.get('usage', {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        })
    }
    return openai_response

def stream_response(nvidia_response, model):
    """Convert NVIDIA NIM streaming response to OpenAI streaming format"""
    for line in nvidia_response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    yield f"data: [DONE]\n\n"
                    break
                
                try:
                    nvidia_chunk = json.loads(data)
                    openai_chunk = {
                        'id': f"chatcmpl-{nvidia_chunk.get('id', 'nvidia')}",
                        'object': 'chat.completion.chunk',
                        'created': int(datetime.now().timestamp()),
                        'model': model,
                        'choices': [
                            {
                                'index': 0,
                                'delta': nvidia_chunk['choices'][0].get('delta', {}),
                                'finish_reason': nvidia_chunk['choices'][0].get('finish_reason')
                            }
                        ]
                    }
                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                except json.JSONDecodeError:
                    continue

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        openai_request = request.get_json()
        nvidia_request = convert_to_nvidia_format(openai_request)
        
        headers = {
            'Authorization': f'Bearer {NVIDIA_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        is_stream = nvidia_request.get('stream', False)
        
        # Make request to NVIDIA NIM
        response = requests.post(
            f'{NVIDIA_BASE_URL}/chat/completions',
            headers=headers,
            json=nvidia_request,
            stream=is_stream
        )
        
        if response.status_code != 200:
            return jsonify({
                'error': {
                    'message': f'NVIDIA API error: {response.text}',
                    'type': 'api_error',
                    'code': response.status_code
                }
            }), response.status_code
        
        if is_stream:
            return Response(
                stream_response(response, openai_request.get('model', 'gpt-3.5-turbo')),
                mimetype='text/event-stream'
            )
        else:
            nvidia_response = response.json()
            openai_response = convert_to_openai_format(
                nvidia_response, 
                openai_request.get('model', 'gpt-3.5-turbo')
            )
            return jsonify(openai_response)
            
    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'server_error'
            }
        }), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models in OpenAI format"""
    models = [
        {
            'id': model_name,
            'object': 'model',
            'created': int(datetime.now().timestamp()),
            'owned_by': 'nvidia'
        }
        for model_name in MODEL_MAPPING.keys()
    ]
    return jsonify({
        'object': 'list',
        'data': models
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting OpenAI-compatible NVIDIA NIM Proxy...")
    print(f"Proxy URL: http://localhost:5000")
    print(f"Chat endpoint: http://localhost:5000/v1/chat/completions")
    print("\nSet your NVIDIA_API_KEY environment variable before running!")
    app.run(host='0.0.0.0', port=5000, debug=False)
