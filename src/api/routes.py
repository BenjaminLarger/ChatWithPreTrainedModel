"""
FastAPI routes for the Multi-Model Chat Interface
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi import UploadFile, File

from src.utils.config import get_settings, get_model_config
from src.models.openai_client import OpenAIClient
from src.models.huggingface_client import HuggingFaceClient
from src.models.inference_api_client import InferenceAPIClient
from src.models.distilbert_model import DistilBERTModel, DistilBERTFineTuner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Model Chat Interface",
    description="A chat interface that supports OpenAI, Hugging Face, and DistilBERT models",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000, description="User message")
    model_type: str = Field(default="openai", description="Model type: openai, huggingface, inference_api, or distilbert")
    conversation_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation messages")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Model-specific parameters")

class ChatResponse(BaseModel):
    message: str = Field(..., description="Generated response")
    model_type: str = Field(..., description="Model used for generation")
    usage: Optional[Dict[str, Any]] = Field(default={}, description="Usage statistics")
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = Field(default=None, description="Error message if any")

class ModelStatus(BaseModel):
    model_type: str
    status: str
    message: str

class FineTuningRequest(BaseModel):
    num_labels: int = Field(default=2, description="Number of classification labels")
    text_column: str = Field(default="text", description="Name of text column in dataset")
    label_column: str = Field(default="label", description="Name of label column in dataset")
    test_size: float = Field(default=0.2, description="Proportion for validation split")
    epochs: Optional[int] = Field(default=None, description="Number of training epochs")
    learning_rate: Optional[float] = Field(default=None, description="Learning rate")
    batch_size: Optional[int] = Field(default=None, description="Training batch size")
    output_name: str = Field(default="fine_tuned_model", description="Name for saved model")

class FineTuningResponse(BaseModel):
    status: str
    message: str
    model_path: Optional[str] = None
    training_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class PredictionRequest(BaseModel):
    texts: Union[str, List[str]] = Field(..., description="Text(s) to classify")
    model_path: Optional[str] = Field(default=None, description="Path to fine-tuned model")
    return_probabilities: bool = Field(default=False, description="Return class probabilities")

class PredictionResponse(BaseModel):
    predictions: Union[Dict[str, Any], List[Dict[str, Any]]]
    model_used: str
    timestamp: datetime

class ConnectionManager:
    """WebSocket connection manager for real-time chat"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)

# Global connection manager
manager = ConnectionManager()

# Model clients cache
_model_clients = {}

def get_model_client(model_type: str):
    """Get or create model client instance"""
    if model_type not in _model_clients:
        try:
            settings = get_settings()
            
            if model_type == "openai":
                _model_clients[model_type] = OpenAIClient(
                    api_key=settings.openai_api_key,
                    model="gpt-3.5-turbo",
                    max_tokens=settings.max_tokens,
                    temperature=settings.temperature
                )
            elif model_type == "huggingface":
                _model_clients[model_type] = HuggingFaceClient()
            elif model_type == "inference_api":
                _model_clients[model_type] = InferenceAPIClient()
            elif model_type == "distilbert":
                _model_clients[model_type] = DistilBERTModel()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize {model_type} client: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize {model_type} client")
    
    return _model_clients[model_type]

# Routes

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/models/status", response_model=List[ModelStatus])
async def get_models_status():
    """Get status of all available models"""
    model_types = ["openai", "huggingface", "inference_api", "distilbert"]
    statuses = []
    
    for model_type in model_types:
        try:
            client = get_model_client(model_type)
            # Try to validate the client
            if hasattr(client, 'validate_connection'):
                await client.validate_connection()
            statuses.append(ModelStatus(
                model_type=model_type,
                status="available",
                message="Model is ready"
            ))
        except Exception as e:
            statuses.append(ModelStatus(
                model_type=model_type,
                status="error",
                message=str(e)
            ))
    
    return statuses

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message and return response"""
    try:
        # Validate message length
        settings = get_settings()
        if len(request.message) > settings.max_message_length:
            raise HTTPException(
                status_code=400,
                detail=f"Message too long. Maximum length: {settings.max_message_length}"
            )
        
        # Get model client
        client = get_model_client(request.model_type)
        
        # Prepare conversation context
        conversation_context = []
        if request.conversation_history:
            conversation_context = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history[-10:]  # Keep last 10 messages
            ]
        
        # Generate response based on model type
        response_text = ""
        usage_info = {}
        
        if request.model_type == "openai":
            conversation_context.append({"role": "user", "content": request.message})
            result = client.generate_text(request.message, conversation_context)
            response_text = result["text"]
            usage_info = result["usage"]
        
        elif request.model_type == "huggingface":
            # Combine conversation context into a single prompt
            if conversation_context:
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_context])
                prompt = f"{context}\nuser: {request.message}\nassistant:"
            else:
                prompt = f"user: {request.message}\nassistant:"
            
            result = client.generate_text(prompt, **request.parameters)
            response_text = result.get("text", result) if isinstance(result, dict) else result
        
        elif request.model_type == "inference_api":
            # Use the inference API client
            if conversation_context:
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_context])
                prompt = f"{context}\nuser: {request.message}\nassistant:"
            else:
                prompt = f"user: {request.message}\nassistant:"
            
            result = client.generate_text(prompt, **request.parameters)
            response_text = result.get("text", result) if isinstance(result, dict) else result
        
        elif request.model_type == "distilbert":
            # DistilBERT is primarily for classification, but we can adapt it for simple responses
            sentiment = client.classify_text(request.message)
            response_text = f"I analyzed your message and detected a {sentiment} sentiment. How can I help you further?"
        
        return ChatResponse(
            message=response_text.strip(),
            model_type=request.model_type,
            usage=usage_info,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error with {request.model_type}: {str(e)}")
        return ChatResponse(
            message="I apologize, but I encountered an error processing your message.",
            model_type=request.model_type,
            error=str(e),
            timestamp=datetime.now()
        )

@app.post("/chat/classify", response_model=Dict[str, Any])
async def classify_text(request: Dict[str, str]):
    """Classify text using available classification models"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        results = {}
        
        # Try DistilBERT classification
        try:
            distilbert_client = get_model_client("distilbert")
            results["distilbert"] = distilbert_client.classify_text(text)
        except Exception as e:
            results["distilbert"] = {"error": str(e)}
        
        # Try Inference API classification
        try:
            inference_client = get_model_client("inference_api")
            results["inference_api"] = inference_client.classify_text(text)
        except Exception as e:
            results["inference_api"] = {"error": str(e)}
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process the chat request
            chat_request = ChatRequest(**data)
            
            # Send acknowledgment
            await manager.send_personal_message({
                "type": "status",
                "message": "Processing your message...",
                "timestamp": datetime.now().isoformat()
            }, websocket)
            
            try:
                # Get model client and generate response
                client = get_model_client(chat_request.model_type)
                
                # Generate response (simplified for WebSocket)
                if chat_request.model_type == "openai":
                    conversation = [{"role": "user", "content": chat_request.message}]
                    result = client.generate_text(chat_request.message, conversation)
                    response_text = result["text"]
                    usage_info = result["usage"]
                elif chat_request.model_type in ["huggingface", "inference_api"]:
                    client = get_model_client(chat_request.model_type)
                    result = client.generate_text(f"user: {chat_request.message}\nassistant:")
                    response_text = result.get("text", result) if isinstance(result, dict) else result
                    usage_info = {}
                elif chat_request.model_type == "distilbert":
                    client = get_model_client(chat_request.model_type)
                    sentiment = client.classify_text(chat_request.message)
                    response_text = f"Sentiment: {sentiment}"
                    usage_info = {}
                
                # Send response back to client
                await manager.send_personal_message({
                    "type": "response",
                    "message": response_text.strip(),
                    "model_type": chat_request.model_type,
                    "usage": usage_info,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
            except Exception as e:
                logger.error(f"WebSocket chat error: {str(e)}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Sorry, I encountered an error processing your message.",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

@app.get("/api/models")
async def list_available_models():
    """List all available models and their configurations"""
    settings = get_settings()
    model_config = get_model_config()
    
    return {
        "models": {
            "openai": {
                "name": "OpenAI GPT-3.5-turbo",
                "type": "text_generation",
                "status": "available" if settings.openai_api_key else "requires_api_key",
                "config": model_config["openai"]
            },
            "huggingface": {
                "name": f"Hugging Face {settings.huggingface_model_name}",
                "type": "text_generation",
                "status": "available",
                "config": model_config["huggingface"]
            },
            "inference_api": {
                "name": "Hugging Face Inference API",
                "type": "text_generation_classification",
                "status": "available" if settings.huggingface_api_token else "requires_api_key",
                "config": {"api_token_required": True}
            },
            "distilbert": {
                "name": f"DistilBERT {settings.distilbert_model_name}",
                "type": "text_classification",
                "status": "available",
                "config": model_config["distilbert"]
            }
        },
        "settings": {
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "max_message_length": settings.max_message_length,
            "max_context_length": settings.max_context_length
        }
    }

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Multi-Model Chat Interface starting up...")
    settings = get_settings()
    logger.info(f"Server will run on {settings.host}:{settings.port}")
    
    # Pre-warm model clients if needed
    try:
        # Only initialize OpenAI if API key is available
        if settings.openai_api_key:
            get_model_client("openai")
            logger.info("OpenAI client initialized")
    except Exception as e:
        logger.warning(f"Failed to pre-warm OpenAI client: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Multi-Model Chat Interface shutting down...")
    # Clean up model clients if needed
    global _model_clients
    _model_clients.clear()


# Fine-tuning endpoints

@app.post("/finetune/distilbert/upload", response_model=FineTuningResponse)
async def fine_tune_distilbert_upload(
    file: UploadFile = File(...),
    request: FineTuningRequest = Depends()
):
    """Fine-tune DistilBERT with uploaded dataset"""
    try:
        # Save uploaded file
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Initialize fine-tuner
        fine_tuner = DistilBERTFineTuner(num_labels=request.num_labels)
        
        # Load and prepare dataset
        data = fine_tuner.load_data_from_file(tmp_file_path)
        train_dataset, val_dataset = fine_tuner.prepare_dataset(
            data=data,
            text_column=request.text_column,
            label_column=request.label_column,
            test_size=request.test_size
        )
        
        # Setup training arguments with custom parameters
        settings = get_settings()
        training_args = fine_tuner.setup_training_args(
            output_dir=f"{settings.fine_tune_output_dir}/{request.output_name}",
            num_train_epochs=request.epochs or settings.fine_tune_epochs,
            per_device_train_batch_size=request.batch_size or settings.fine_tune_batch_size,
            learning_rate=request.learning_rate or settings.fine_tune_learning_rate
        )
        
        # Run fine-tuning
        results = fine_tuner.fine_tune(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_args=training_args
        )
        
        # Save the model
        model_save_path = f"{settings.fine_tune_output_dir}/{request.output_name}"
        fine_tuner.save_model(model_save_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return FineTuningResponse(
            status="success",
            message=f"Fine-tuning completed successfully. Model saved to {model_save_path}",
            model_path=model_save_path,
            training_results=results
        )
        
    except Exception as e:
        logger.error(f"Fine-tuning error: {str(e)}")
        return FineTuningResponse(
            status="error",
            message="Fine-tuning failed",
            error=str(e)
        )

@app.post("/finetune/distilbert/sample", response_model=FineTuningResponse)
async def fine_tune_distilbert_sample(request: FineTuningRequest):
    """Fine-tune DistilBERT with sample dataset for testing"""
    try:
        # Initialize fine-tuner
        fine_tuner = DistilBERTFineTuner(num_labels=request.num_labels)
        
        # Create sample dataset
        data = fine_tuner.create_sample_dataset(num_samples=200)
        train_dataset, val_dataset = fine_tuner.prepare_dataset(
            data=data,
            text_column=request.text_column,
            label_column=request.label_column,
            test_size=request.test_size
        )
        
        # Setup training arguments
        settings = get_settings()
        training_args = fine_tuner.setup_training_args(
            output_dir=f"{settings.fine_tune_output_dir}/{request.output_name}",
            num_train_epochs=request.epochs or 2,  # Use fewer epochs for sample data
            per_device_train_batch_size=request.batch_size or 8,
            learning_rate=request.learning_rate or settings.fine_tune_learning_rate
        )
        
        # Run fine-tuning
        results = fine_tuner.fine_tune(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_args=training_args
        )
        
        # Save the model
        model_save_path = f"{settings.fine_tune_output_dir}/{request.output_name}"
        fine_tuner.save_model(model_save_path)
        
        return FineTuningResponse(
            status="success",
            message=f"Sample fine-tuning completed successfully. Model saved to {model_save_path}",
            model_path=model_save_path,
            training_results=results
        )
        
    except Exception as e:
        logger.error(f"Sample fine-tuning error: {str(e)}")
        return FineTuningResponse(
            status="error",
            message="Sample fine-tuning failed",
            error=str(e)
        )

@app.post("/predict/distilbert", response_model=PredictionResponse)
async def predict_with_distilbert(request: PredictionRequest):
    """Make predictions using DistilBERT (fine-tuned or default)"""
    try:
        if request.model_path:
            # Load fine-tuned model
            model = DistilBERTModel(model_path=request.model_path)
            model_used = f"Fine-tuned: {request.model_path}"
        else:
            # Use default model
            model = get_model_client("distilbert")
            model_used = "Default DistilBERT"
        
        # Make predictions
        if isinstance(request.texts, str):
            predictions = model.classify_text(request.texts, return_probabilities=request.return_probabilities)
        else:
            predictions = []
            for text in request.texts:
                pred = model.classify_text(text, return_probabilities=request.return_probabilities)
                predictions.append(pred)
        
        return PredictionResponse(
            predictions=predictions,
            model_used=model_used,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/finetune/models")
async def list_fine_tuned_models():
    """List all available fine-tuned models"""
    try:
        settings = get_settings()
        models_dir = settings.fine_tune_output_dir
        
        if not os.path.exists(models_dir):
            return {"fine_tuned_models": []}
        
        models = []
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if os.path.isdir(model_path):
                # Check if it's a valid model directory
                config_path = os.path.join(model_path, "config.json")
                history_path = os.path.join(model_path, "training_history.json")
                
                model_info = {
                    "name": model_name,
                    "path": model_path,
                    "created": datetime.fromtimestamp(os.path.getctime(model_path)).isoformat(),
                    "has_config": os.path.exists(config_path),
                    "has_training_history": os.path.exists(history_path)
                }
                
                # Load training history if available
                if os.path.exists(history_path):
                    try:
                        with open(history_path, 'r') as f:
                            history = json.load(f)
                        if history:
                            latest_result = history[-1]
                            model_info["last_training_accuracy"] = latest_result.get("eval_metrics", {}).get("eval_accuracy")
                            model_info["training_completed"] = latest_result.get("training_completed", False)
                    except Exception as e:
                        logger.warning(f"Could not load training history for {model_name}: {e}")
                
                models.append(model_info)
        
        return {"fine_tuned_models": models}
        
    except Exception as e:
        logger.error(f"Error listing fine-tuned models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))