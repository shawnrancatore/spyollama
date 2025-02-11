#!/usr/bin/env python3
import os
import json
import logging
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
from functools import wraps
from flask import Flask, request, jsonify, g, Response, stream_with_context
from flask_cors import CORS


# Configuration using environment variables with defaults
@dataclass
class Config:
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', 11434))
    RESPONSE_FILE: Path = Path(os.getenv('RESPONSE_FILE', 'response.txt'))
    REQUESTS_DIR: Path = Path(os.getenv('REQUESTS_DIR', 'requests'))
    MAX_REQUEST_SIZE: int = int(os.getenv('MAX_REQUEST_SIZE', 1024 * 1024))  # 1MB
    LOG_FILE: str = os.getenv('LOG_FILE', 'llm_requests.log')
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_TO_STDOUT: bool = os.getenv('VERBOSE', 'true').lower() == 'true'


config = Config()

# Constants
DEFAULT_MODEL = "llama3.3:latest"
API_VERSION = "0.5.7"
AVAILABLE_MODELS = [{
    "name": DEFAULT_MODEL,
    "model": DEFAULT_MODEL,
    "modified_at": "2025-02-04T12:02:18.3666057-05:00",
    "size": 42520413916,
    "digest": "a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa",
    "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "70.6B",
        "quantization_level": "Q4_K_M"
    }
}]

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Setup logging
def get_request_id():
    if not hasattr(g, 'request_id'):
        try:
            g.request_id = str(uuid.uuid4())
        except RuntimeError:
            return 'no-request'
    return g.request_id


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        try:
            record.request_id = get_request_id()
        except Exception:
            record.request_id = 'no-request'
        return True


def setup_logging():
    # Create formatters - use simpler format without request_id for general logging
    base_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Configure root logger with basic format
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=base_format,
        force=True
    )

    # Create handlers for Flask app specifically
    handlers = []

    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(logging.Formatter(base_format))
    file_handler.addFilter(RequestIdFilter())
    handlers.append(file_handler)

    # Optional stdout handler
    if config.LOG_TO_STDOUT:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(logging.Formatter(base_format))
        stdout_handler.addFilter(RequestIdFilter())
        handlers.append(stdout_handler)

    def init_app_logging(app):
        # Configure Flask app logger
        app.logger.handlers = []
        for handler in handlers:
            app.logger.addHandler(handler)
        app.logger.setLevel(getattr(logging, config.LOG_LEVEL))

    return init_app_logging


def log_request(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        logging.info(f"Received {request.method} request to {request.path}",
                     extra={'request_id': get_request_id()})
        return f(*args, **kwargs)

    return decorated


class RequestProcessor:
    @staticmethod
    def validate_request_size(request_data: bytes) -> None:
        if len(request_data) > config.MAX_REQUEST_SIZE:
            raise ValueError("Request size exceeds maximum allowed size")

    @staticmethod
    def process_request() -> Dict[str, Any]:
        try:
            RequestProcessor.validate_request_size(request.get_data())
            details = {}

            if request.is_json:
                try:
                    payload = request.get_json()
                    details["json_payload"] = payload
                except Exception as e:
                    details["json_payload_error"] = str(e)
            else:
                raw_text = request.data.decode('utf-8') if request.data else ""
                if raw_text:
                    details["raw_text"] = raw_text

            if request.files:
                files_info = {}
                for key, file in request.files.items():
                    try:
                        content = file.read()
                        text_content = content.decode('utf-8')
                        files_info[key] = {
                            "filename": file.filename,
                            "content": text_content
                        }
                    except UnicodeDecodeError:
                        files_info[key] = {
                            "filename": file.filename,
                            "content": "<binary data>"
                        }
                    except Exception as e:
                        files_info[key] = {"error": str(e)}
                details["files"] = files_info

            return details

        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            raise

    @staticmethod
    def save_request_to_file(details: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        requests_dir = Path(config.REQUESTS_DIR)
        requests_dir.mkdir(exist_ok=True)

        filename = requests_dir / f"{timestamp}.md"

        lines = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append("# Request Details\n")
        lines.append(f"**Timestamp:** {current_time}\n\n")

        if "json_payload" in details:
            payload = details["json_payload"]
            if isinstance(payload, dict) and "messages" in payload:
                messages = payload["messages"]

                # Add all messages with their roles
                lines.append("## Messages\n")
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "").strip()
                    if content:
                        lines.append(f"### {role.title()}\n")
                        lines.append(f"{content}\n\n")
            else:
                lines.append("## JSON Payload\n")
                lines.append("```json\n")
                lines.append(json.dumps(payload, indent=2) + "\n")
                lines.append("```\n\n")

        try:
            filename.write_text('\n'.join(lines), encoding='utf-8')
            logging.info(f"Saved request details to {filename}")
            return filename
        except Exception as e:
            logging.error(f"Error saving request details: {str(e)}")
            raise


# Routes
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
@log_request
def catch_all(path):
    return jsonify({
        "error": "endpoint_not_found",
        "message": f"The requested endpoint '/{path}' does not exist",
        "method": request.method
    }), 404


@app.route('/api/tags', methods=['GET'])
@log_request
def get_tags():
    return jsonify({"models": AVAILABLE_MODELS}), 200


@app.route('/api/generate', methods=['POST'])
@log_request
def generate_text():
    try:
        details = RequestProcessor.process_request()
        RequestProcessor.save_request_to_file(details)

        if not config.RESPONSE_FILE.exists():
            return jsonify({"error": "Response file not found"}), 500

        response_text = config.RESPONSE_FILE.read_text(encoding='utf-8')
        return response_text, 200
    except Exception as e:
        logging.error(f"Error in generate_text: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/api/chat/completions', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
@log_request
def chat_completions():
    try:
        details = RequestProcessor.process_request()
        RequestProcessor.save_request_to_file(details)
        response_text = config.RESPONSE_FILE.read_text(encoding='utf-8') if config.RESPONSE_FILE.exists() else "received."
        req_json = request.get_json(silent=True) or {}
        is_stream = req_json.get("stream", False)

        def build_chunk(delta, finish_reason=None):
            return {
                "id": f"chatcmpl-{random.randint(0, 99999):05d}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": DEFAULT_MODEL,
                "system_fingerprint": "fp_ollama",
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason
                }]
            }

        if request.path == "/v1/chat/completions":
            if is_stream:
                def generate():
                    yield "data: " + json.dumps(build_chunk({"role": "assistant"})) + "\n\n"
                    for chunk in [response_text[i:i+4] for i in range(0, len(response_text), 4)]:
                        yield "data: " + json.dumps(build_chunk({"content": chunk})) + "\n\n"
                    yield "data: " + json.dumps(build_chunk({}, "stop")) + "\n\n"
                    yield "data: [DONE]\n\n"
                return Response(stream_with_context(generate()), mimetype="text/event-stream")

            return jsonify({
                "id": f"chatcmpl-{random.randint(0, 99999):05d}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": DEFAULT_MODEL,
                "system_fingerprint": "fp_ollama",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                        "images": None
                    },
                    "finish_reason": "length"
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 50, "total_tokens": 70}
            }), 200

        return jsonify({
            "model": DEFAULT_MODEL,
            "created_at": datetime.now().isoformat(),
            "message": {
                "role": "assistant",
                "content": response_text,
                "images": None
            },
            "done": True
        }), 200

    except Exception as e:
        logging.error(f"Error in chat_completions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
@log_request
def index():
    return "Ollama is running", 200


@app.route('/api/version', methods=['GET'])
@log_request
def version():
    return jsonify({
        "version": API_VERSION
    }), 200


@app.route('/api/health', methods=['GET'])
@log_request
def health():
    health_status = "ok"
    status_code = 200

    # Basic health checks
    try:
        # Check if response file is accessible
        if not config.RESPONSE_FILE.exists():
            health_status = "warning: response file not found"

        # Check if requests directory is writable
        requests_dir = Path(config.REQUESTS_DIR)
        requests_dir.mkdir(exist_ok=True)
        test_file = requests_dir / ".test_write"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        health_status = f"error: {str(e)}"
        status_code = 500
        logging.error(f"Health check failed: {str(e)}")

    return jsonify({
        "status": health_status
    }), status_code


def cleanup_old_requests(max_age_days=7):
    """Optional cleanup of old request files"""
    try:
        requests_dir = Path(config.REQUESTS_DIR)
        if not requests_dir.exists():
            return

        current_time = datetime.now()
        for file in requests_dir.glob("*.md"):
            file_age = datetime.fromtimestamp(file.stat().st_mtime)
            if (current_time - file_age).days > max_age_days:
                file.unlink()
                logging.info(f"Cleaned up old request file: {file}")
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")


if __name__ == '__main__':
    try:
        init_app_logging = setup_logging()
        Path(config.REQUESTS_DIR).mkdir(exist_ok=True)

        # optional
        # cleanup_old_requests()

        # Log startup information
        logging.info(f"Starting server on {config.HOST}:{config.PORT}")
        logging.info(f"Response file: {config.RESPONSE_FILE}")
        logging.info(f"Requests directory: {config.REQUESTS_DIR}")
        logging.info(f"Logging to stdout: {config.LOG_TO_STDOUT}")

        # Initialize request-specific logging
        init_app_logging(app)

        # Start the Flask application
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=False
        )
    except Exception as e:
        logging.error(f"Server startup failed: {str(e)}")
        raise