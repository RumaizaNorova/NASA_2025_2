"""
OpenAI Integration Service for Shark Habitat Prediction
Provides AI-powered insights and natural language queries
"""

import openai
import os
import json
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.model = "gpt-4"
        
    async def generate_insights(self, prediction_data: Dict, shark_data: Dict) -> Dict:
        """Generate AI-powered insights about shark habitat predictions"""
        try:
            prompt = self._create_insights_prompt(prediction_data, shark_data)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a marine biologist and data scientist specializing in shark behavior and habitat prediction. Provide scientific, accurate insights based on the data provided."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return {
                "insights": response.choices[0].message.content,
                "model": self.model,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "insights": "Unable to generate insights at this time.",
                "model": self.model,
                "status": "error",
                "error": str(e)
            }
    
    async def answer_question(self, question: str, context_data: Dict) -> Dict:
        """Answer natural language questions about shark data"""
        try:
            prompt = self._create_question_prompt(question, context_data)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a marine biologist assistant. Answer questions about shark behavior, habitat prediction, and environmental factors using the provided data. Be scientific and accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5
            )
            
            return {
                "answer": response.choices[0].message.content,
                "model": self.model,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": "I'm sorry, I couldn't process your question at this time.",
                "model": self.model,
                "status": "error",
                "error": str(e)
            }
    
    async def generate_report(self, analysis_data: Dict) -> Dict:
        """Generate a comprehensive analysis report"""
        try:
            prompt = self._create_report_prompt(analysis_data)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a marine biologist writing a scientific report about shark habitat prediction analysis. Write in a professional, scientific tone with clear sections and actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.6
            )
            
            return {
                "report": response.choices[0].message.content,
                "model": self.model,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "report": "Unable to generate report at this time.",
                "model": self.model,
                "status": "error",
                "error": str(e)
            }
    
    def _create_insights_prompt(self, prediction_data: Dict, shark_data: Dict) -> str:
        """Create prompt for insights generation"""
        return f"""
        Based on the following shark habitat prediction data, provide scientific insights:
        
        Prediction Data:
        - Foraging Probability: {prediction_data.get('foraging_probability', 'N/A')}
        - Confidence: {prediction_data.get('confidence', 'N/A')}
        - Location: {prediction_data.get('location', 'N/A')}
        - Model: {prediction_data.get('model_info', {}).get('model_type', 'N/A')}
        - AUC Score: {prediction_data.get('model_info', {}).get('auc_score', 'N/A')}
        
        Shark Data Context:
        - Total Records: {shark_data.get('total_records', 'N/A')}
        - Species: {shark_data.get('species_count', 'N/A')}
        - Foraging Rate: {shark_data.get('foraging_distribution', {}).get('foraging', 'N/A')}
        
        Please provide:
        1. What this prediction tells us about shark habitat suitability
        2. Key environmental factors influencing this prediction
        3. Scientific implications for shark conservation
        4. Recommendations for further study
        
        Keep the response concise and scientific.
        """
    
    def _create_question_prompt(self, question: str, context_data: Dict) -> str:
        """Create prompt for question answering"""
        return f"""
        Question: {question}
        
        Context Data:
        - Dataset: {context_data.get('total_records', 'N/A')} shark tracking records
        - Species: {context_data.get('species_count', 'N/A')} different species
        - Model Performance: AUC {context_data.get('best_auc', 'N/A')}
        - Date Range: {context_data.get('date_range', 'N/A')}
        - Foraging Distribution: {context_data.get('foraging_distribution', 'N/A')}
        
        Please answer the question based on this context and your knowledge of shark biology and habitat prediction.
        """
    
    def _create_report_prompt(self, analysis_data: Dict) -> str:
        """Create prompt for report generation"""
        return f"""
        Generate a scientific report based on the following shark habitat prediction analysis:
        
        Analysis Data:
        - Model Performance: {analysis_data.get('model_performance', 'N/A')}
        - Dataset Statistics: {analysis_data.get('dataset_stats', 'N/A')}
        - Feature Importance: {analysis_data.get('feature_importance', 'N/A')}
        - Temporal Analysis: {analysis_data.get('temporal_analysis', 'N/A')}
        - Species Distribution: {analysis_data.get('species_distribution', 'N/A')}
        
        Please structure the report with:
        1. Executive Summary
        2. Methodology
        3. Key Findings
        4. Environmental Factors
        5. Conservation Implications
        6. Recommendations
        
        Write in a professional, scientific tone suitable for marine biology research.
        """

# Global instance
openai_service = OpenAIService()

