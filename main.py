"""
AI-Powered Trading Agent
FastAPI backend with endpoints for model training, data fetching, and predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import sys
import os
import uvicorn
import traceback

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_trainer import RobustModelTrainer
from data.data_ingestion import DataIngestion

app = FastAPI(
    title="IntelliTradeAI API",
    description="AI-Powered Trading Agent with ML-based prediction capabilities",
    version="1.0.0"
)

# Initialize components
trainer = RobustModelTrainer()
ingestor = DataIngestion()

@app.get("/")
def root():
    """Root endpoint - API status"""
    return {
        "message": "IntelliTradeAI API is running",
        "version": "1.0.0",
        "status": "active",
        "features": [
            "Model training and retraining",
            "Real-time data fetching",
            "ML-based predictions",
            "Multi-asset support (stocks, crypto)"
        ]
    }

@app.post("/retrain")
def retrain_model(symbol: str = "BTC", algorithms: str = "random_forest"):
    """Retrain model for a specific symbol"""
    try:
        # Parse algorithms
        algorithm_list = [alg.strip() for alg in algorithms.split(",")]
        
        # Fetch data for training
        if symbol in ["BTC", "ETH", "LTC"]:
            # Cryptocurrency data
            crypto_data = ingestor.fetch_crypto_data([symbol])
            if not crypto_data or symbol not in crypto_data:
                raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
            data = crypto_data[symbol]
        else:
            # Stock data
            stock_data = ingestor.fetch_stock_data([symbol])
            if not stock_data or symbol not in stock_data:
                raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
            data = stock_data[symbol]
        
        # Run training
        results = trainer.run_comprehensive_training(
            data,
            symbol,
            algorithms=algorithm_list,
            optimize_hyperparams=False
        )
        
        return {
            "status": "Retraining complete",
            "symbol": symbol,
            "algorithms": algorithm_list,
            "best_model": results["best_model"],
            "data_shape": results["data_shape"],
            "models": {k: v["status"] for k, v in results["models"].items()}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/data")
def get_data(symbols: str = "BTC,AAPL", data_type: str = "mixed"):
    """Fetch market data for specified symbols"""
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        
        result = {}
        
        if data_type in ["crypto", "mixed"]:
            # Fetch crypto data
            crypto_symbols = [s for s in symbol_list if s in ["BTC", "ETH", "LTC"]]
            if crypto_symbols:
                crypto_data = ingestor.fetch_crypto_data(crypto_symbols)
                if crypto_data:
                    for symbol, data in crypto_data.items():
                        result[symbol] = {
                            "type": "cryptocurrency",
                            "rows": len(data),
                            "latest_price": float(data['close'].iloc[-1]) if len(data) > 0 else None,
                            "date_range": f"{data.index[0]} to {data.index[-1]}" if len(data) > 0 else None
                        }
        
        if data_type in ["stock", "mixed"]:
            # Fetch stock data
            stock_symbols = [s for s in symbol_list if s not in ["BTC", "ETH", "LTC"]]
            if stock_symbols:
                stock_data = ingestor.fetch_stock_data(stock_symbols)
                if stock_data:
                    for symbol, data in stock_data.items():
                        result[symbol] = {
                            "type": "stock",
                            "rows": len(data),
                            "latest_price": float(data['close'].iloc[-1]) if len(data) > 0 else None,
                            "date_range": f"{data.index[0]} to {data.index[-1]}" if len(data) > 0 else None
                        }
        
        return {
            "status": "Data fetched successfully",
            "symbols": list(result.keys()),
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")

@app.get("/predict")
def predict(symbol: str = "BTC"):
    """Generate predictions for a symbol using trained models"""
    try:
        # Get model summary to check available models
        summary = trainer.get_model_summary()
        
        # Find available models for the symbol
        available_models = [key for key in summary["models"].keys() if key.startswith(symbol)]
        
        if not available_models:
            raise HTTPException(
                status_code=404, 
                detail=f"No trained models found for {symbol}. Train a model first using /retrain"
            )
        
        # Get latest data for prediction
        if symbol in ["BTC", "ETH", "LTC"]:
            data_dict = ingestor.fetch_crypto_data([symbol])
        else:
            data_dict = ingestor.fetch_stock_data([symbol])
        
        if not data_dict or symbol not in data_dict:
            raise HTTPException(status_code=404, detail=f"No current data available for {symbol}")
        
        data = data_dict[symbol]
        
        # Generate features for prediction
        features = trainer.engineer_features(data, symbol)
        
        # Make predictions with available models
        predictions = {}
        
        for model_key in available_models:
            try:
                algorithm = summary["models"][model_key]["algorithm"]
                # Extract algorithm name from model key (more reliable)
                algorithm_name = model_key.split('_', 1)[1]  # e.g., "AAPL_random_forest" -> "random_forest"
                model_predictions = trainer.make_predictions(features, symbol, algorithm_name)
                
                if model_predictions is not None:
                    predictions[algorithm] = {
                        "signal": int(model_predictions[-1]) if len(model_predictions) > 0 else None,
                        "confidence": float(max(model_predictions)) if len(model_predictions) > 0 else None,
                        "recent_signals": [int(x) for x in model_predictions[-5:]] if len(model_predictions) >= 5 else []
                    }
            except Exception as model_error:
                algorithm = summary["models"][model_key]["algorithm"]  # Ensure algorithm is defined
                predictions[algorithm] = {"error": str(model_error)}
        
        return {
            "symbol": symbol,
            "latest_price": float(data['close'].iloc[-1]),
            "predictions": predictions,
            "available_models": len(available_models),
            "timestamp": str(data.index[-1])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models")
def get_models():
    """Get information about trained models"""
    try:
        summary = trainer.get_model_summary()
        return {
            "total_models": summary["total_models"],
            "models": summary["models"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api": "running",
        "components": {
            "model_trainer": "ready",
            "data_ingestor": "ready"
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
