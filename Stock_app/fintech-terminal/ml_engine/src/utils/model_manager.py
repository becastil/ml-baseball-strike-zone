"""
Model Manager

This module handles model versioning, loading, saving, and deployment
for the ML engine.
"""

import os
import json
import pickle
import joblib
import shutil
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch


class ModelManager:
    """Comprehensive model management system"""
    
    def __init__(self, 
                 base_path: str = "models",
                 use_mlflow: bool = False,
                 mlflow_uri: Optional[str] = None):
        """
        Initialize the model manager
        
        Args:
            base_path: Base directory for model storage
            use_mlflow: Whether to use MLflow for tracking
            mlflow_uri: MLflow tracking URI
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.use_mlflow = use_mlflow
        
        if self.use_mlflow:
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("fintech-terminal")
        
        self.model_registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load or create model registry"""
        registry_path = self.base_path / "model_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        else:
            return {"models": {}}
    
    def _save_registry(self):
        """Save model registry"""
        registry_path = self.base_path / "model_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def save_model(self,
                   model: Any,
                   model_name: str,
                   version: Optional[str] = None,
                   metadata: Optional[Dict] = None,
                   artifacts: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a model with versioning and metadata
        
        Args:
            model: Model object to save
            model_name: Name of the model
            version: Version string (auto-generated if None)
            metadata: Additional metadata
            artifacts: Additional artifacts to save
            
        Returns:
            Model ID
        """
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        model_dir = self.base_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine model type and save accordingly
        model_path = model_dir / "model.pkl"
        
        if hasattr(model, 'save'):  # Keras/TensorFlow models
            model_path = model_dir / "model.h5"
            model.save(str(model_path))
            model_type = "tensorflow"
        elif hasattr(model, 'save_pretrained'):  # Transformers models
            model.save_pretrained(str(model_dir))
            model_type = "transformers"
        else:  # Sklearn and other models
            joblib.dump(model, model_path)
            model_type = "sklearn"
        
        # Save metadata
        model_metadata = {
            "model_name": model_name,
            "version": version,
            "model_type": model_type,
            "created_at": datetime.now().isoformat(),
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            **(metadata or {})
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Save additional artifacts
        if artifacts:
            artifacts_dir = model_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            
            for name, artifact in artifacts.items():
                if isinstance(artifact, pd.DataFrame):
                    artifact.to_csv(artifacts_dir / f"{name}.csv")
                elif isinstance(artifact, dict):
                    with open(artifacts_dir / f"{name}.json", 'w') as f:
                        json.dump(artifact, f, indent=2)
                elif isinstance(artifact, np.ndarray):
                    np.save(artifacts_dir / f"{name}.npy", artifact)
                else:
                    joblib.dump(artifact, artifacts_dir / f"{name}.pkl")
        
        # Generate model ID
        model_id = f"{model_name}_{version}"
        
        # Update registry
        if model_name not in self.model_registry["models"]:
            self.model_registry["models"][model_name] = {}
        
        self.model_registry["models"][model_name][version] = {
            "model_id": model_id,
            "path": str(model_dir),
            "metadata": model_metadata
        }
        
        self._save_registry()
        
        # MLflow tracking
        if self.use_mlflow:
            with mlflow.start_run():
                mlflow.log_params(metadata or {})
                
                if model_type == "sklearn":
                    mlflow.sklearn.log_model(model, model_name)
                elif model_type == "tensorflow":
                    mlflow.tensorflow.log_model(model, model_name)
                
                mlflow.set_tag("model_version", version)
                mlflow.set_tag("model_id", model_id)
        
        return model_id
    
    def load_model(self,
                   model_name: str,
                   version: Optional[str] = None) -> Tuple[Any, Dict]:
        """
        Load a model and its metadata
        
        Args:
            model_name: Name of the model
            version: Specific version (latest if None)
            
        Returns:
            Tuple of (model, metadata)
        """
        if model_name not in self.model_registry["models"]:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Get version
        if version is None:
            # Get latest version
            versions = list(self.model_registry["models"][model_name].keys())
            version = sorted(versions)[-1]
        
        if version not in self.model_registry["models"][model_name]:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        model_info = self.model_registry["models"][model_name][version]
        model_dir = Path(model_info["path"])
        
        # Load metadata
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load model based on type
        model_type = metadata["model_type"]
        
        if model_type == "tensorflow":
            import tensorflow as tf
            model = tf.keras.models.load_model(model_dir / "model.h5")
        elif model_type == "transformers":
            from transformers import AutoModel
            model = AutoModel.from_pretrained(str(model_dir))
        else:  # sklearn and others
            model = joblib.load(model_dir / "model.pkl")
        
        return model, metadata
    
    def load_artifacts(self,
                      model_name: str,
                      version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load artifacts associated with a model
        
        Args:
            model_name: Name of the model
            version: Specific version
            
        Returns:
            Dictionary of artifacts
        """
        if version is None:
            versions = list(self.model_registry["models"][model_name].keys())
            version = sorted(versions)[-1]
        
        model_info = self.model_registry["models"][model_name][version]
        artifacts_dir = Path(model_info["path"]) / "artifacts"
        
        if not artifacts_dir.exists():
            return {}
        
        artifacts = {}
        
        for file_path in artifacts_dir.iterdir():
            name = file_path.stem
            
            if file_path.suffix == ".csv":
                artifacts[name] = pd.read_csv(file_path, index_col=0)
            elif file_path.suffix == ".json":
                with open(file_path, 'r') as f:
                    artifacts[name] = json.load(f)
            elif file_path.suffix == ".npy":
                artifacts[name] = np.load(file_path)
            elif file_path.suffix == ".pkl":
                artifacts[name] = joblib.load(file_path)
        
        return artifacts
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all available models and versions"""
        return {
            model_name: list(versions.keys())
            for model_name, versions in self.model_registry["models"].items()
        }
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict:
        """Get detailed information about a model"""
        if version is None:
            versions = list(self.model_registry["models"][model_name].keys())
            version = sorted(versions)[-1]
        
        return self.model_registry["models"][model_name][version]
    
    def compare_models(self,
                      model_names: List[Tuple[str, Optional[str]]]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            model_names: List of (model_name, version) tuples
            
        Returns:
            DataFrame with model comparison
        """
        comparisons = []
        
        for model_name, version in model_names:
            info = self.get_model_info(model_name, version)
            metadata = info["metadata"]
            
            comparison = {
                "model_name": model_name,
                "version": version or "latest",
                "created_at": metadata["created_at"],
                "model_type": metadata["model_type"],
                "model_class": metadata["model_class"]
            }
            
            # Add custom metadata fields
            for key, value in metadata.items():
                if key not in comparison and isinstance(value, (str, int, float)):
                    comparison[key] = value
            
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)
    
    def delete_model(self, model_name: str, version: Optional[str] = None):
        """Delete a model version or all versions"""
        if model_name not in self.model_registry["models"]:
            raise ValueError(f"Model '{model_name}' not found")
        
        if version:
            # Delete specific version
            if version in self.model_registry["models"][model_name]:
                model_path = Path(self.model_registry["models"][model_name][version]["path"])
                shutil.rmtree(model_path)
                del self.model_registry["models"][model_name][version]
                
                # Remove model entry if no versions left
                if not self.model_registry["models"][model_name]:
                    del self.model_registry["models"][model_name]
        else:
            # Delete all versions
            for version_info in self.model_registry["models"][model_name].values():
                model_path = Path(version_info["path"])
                shutil.rmtree(model_path)
            
            del self.model_registry["models"][model_name]
        
        self._save_registry()
    
    def export_model(self,
                    model_name: str,
                    version: Optional[str] = None,
                    export_path: str = "exports",
                    format: str = "pkl") -> str:
        """
        Export model for deployment
        
        Args:
            model_name: Name of the model
            version: Model version
            export_path: Export directory
            format: Export format
            
        Returns:
            Path to exported model
        """
        model, metadata = self.load_model(model_name, version)
        artifacts = self.load_artifacts(model_name, version)
        
        export_dir = Path(export_path)
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{model_name}_{version or 'latest'}_{timestamp}"
        export_file = export_dir / f"{export_name}.{format}"
        
        if format == "pkl":
            # Pickle format with all components
            export_data = {
                "model": model,
                "metadata": metadata,
                "artifacts": artifacts
            }
            joblib.dump(export_data, export_file)
        
        elif format == "onnx":
            # ONNX format for interoperability
            try:
                import onnx
                import skl2onnx
                
                if metadata["model_type"] == "sklearn":
                    # Convert sklearn model to ONNX
                    initial_type = [('float_input', skl2onnx.common.data_types.FloatTensorType([None, 10]))]
                    onx = skl2onnx.convert_sklearn(model, initial_types=initial_type)
                    
                    export_file = export_dir / f"{export_name}.onnx"
                    with open(export_file, "wb") as f:
                        f.write(onx.SerializeToString())
            except ImportError:
                print("ONNX conversion requires onnx and skl2onnx packages")
                return None
        
        # Save deployment config
        config = {
            "model_name": model_name,
            "version": version or "latest",
            "export_format": format,
            "export_date": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        config_file = export_dir / f"{export_name}_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        return str(export_file)
    
    def create_ensemble(self,
                       models: List[Tuple[str, Optional[str]]],
                       ensemble_name: str,
                       weights: Optional[List[float]] = None) -> str:
        """
        Create an ensemble from multiple models
        
        Args:
            models: List of (model_name, version) tuples
            ensemble_name: Name for the ensemble
            weights: Weights for each model
            
        Returns:
            Ensemble ID
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Load all models
        loaded_models = []
        for model_name, version in models:
            model, metadata = self.load_model(model_name, version)
            loaded_models.append({
                "model": model,
                "metadata": metadata,
                "weight": weights[models.index((model_name, version))]
            })
        
        # Create ensemble wrapper
        class EnsembleModel:
            def __init__(self, models_info):
                self.models_info = models_info
            
            def predict(self, X):
                predictions = []
                for info in self.models_info:
                    pred = info["model"].predict(X)
                    predictions.append(pred * info["weight"])
                
                return np.sum(predictions, axis=0)
            
            def predict_proba(self, X):
                if hasattr(self.models_info[0]["model"], 'predict_proba'):
                    probas = []
                    for info in self.models_info:
                        proba = info["model"].predict_proba(X)
                        probas.append(proba * info["weight"])
                    
                    return np.sum(probas, axis=0)
                else:
                    raise AttributeError("Component models don't support predict_proba")
        
        ensemble = EnsembleModel(loaded_models)
        
        # Save ensemble
        ensemble_metadata = {
            "ensemble_type": "weighted_average",
            "component_models": [
                {"name": name, "version": version, "weight": weight}
                for (name, version), weight in zip(models, weights)
            ]
        }
        
        return self.save_model(ensemble, ensemble_name, metadata=ensemble_metadata)
    
    def version_control(self, model_name: str) -> Dict:
        """
        Get version control information for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Version control information
        """
        if model_name not in self.model_registry["models"]:
            return {"error": "Model not found"}
        
        versions = self.model_registry["models"][model_name]
        
        # Create version tree
        version_info = []
        for version, info in sorted(versions.items()):
            metadata = info["metadata"]
            
            version_info.append({
                "version": version,
                "created_at": metadata["created_at"],
                "model_type": metadata["model_type"],
                "performance_metrics": metadata.get("performance_metrics", {}),
                "training_data": metadata.get("training_data", "unknown"),
                "hyperparameters": metadata.get("hyperparameters", {})
            })
        
        return {
            "model_name": model_name,
            "total_versions": len(versions),
            "latest_version": sorted(versions.keys())[-1],
            "versions": version_info
        }
    
    def deploy_model(self,
                    model_name: str,
                    version: Optional[str] = None,
                    deployment_type: str = "api") -> Dict:
        """
        Deploy a model (generates deployment configuration)
        
        Args:
            model_name: Name of the model
            version: Model version
            deployment_type: Type of deployment
            
        Returns:
            Deployment configuration
        """
        model_info = self.get_model_info(model_name, version)
        
        if deployment_type == "api":
            # Generate FastAPI deployment code
            api_code = f'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("{model_info['path']}/model.pkl")

class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str = "{version or 'latest'}"

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return PredictionResponse(prediction=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {{"status": "healthy", "model": "{model_name}"}}
'''
            
            deployment_config = {
                "deployment_type": "fastapi",
                "model_name": model_name,
                "version": version or "latest",
                "api_code": api_code,
                "requirements": [
                    "fastapi",
                    "uvicorn",
                    "joblib",
                    "numpy",
                    "scikit-learn"
                ],
                "dockerfile": f'''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
            }
        
        elif deployment_type == "batch":
            deployment_config = {
                "deployment_type": "batch",
                "model_name": model_name,
                "version": version or "latest",
                "model_path": model_info["path"],
                "batch_config": {
                    "input_format": "csv",
                    "output_format": "csv",
                    "batch_size": 1000
                }
            }
        
        else:
            raise ValueError(f"Unknown deployment type: {deployment_type}")
        
        return deployment_config