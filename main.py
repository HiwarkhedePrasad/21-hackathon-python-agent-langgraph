"""
Enhanced TalentFlow Pro - Complete CV Processing System with Advanced Ranking

Complete implementation with comprehensive ranking system, tier classification, 
and detailed candidate evaluation matching the frontend interface.
"""

import os
import json
import re
import uuid
import tempfile
import logging
import asyncio
from typing import Dict, Any, TypedDict, List, Optional, Annotated
from datetime import datetime
from pathlib import Path
import zipfile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Groq integration
from langchain_groq import ChatGroq

# Document processing
import pdfplumber
from docx import Document
import mysql.connector

# Email functionality
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# FastAPI for API endpoints
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =================================================================
# LOGGING CONFIGURATION
# =================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =================================================================
# STATE DEFINITIONS
# =================================================================

class CandidateProcessingState(TypedDict):
    """State for individual candidate processing"""
    file_path: str
    file_name: str
    raw_cv_text: str
    structured_data: Optional[Dict[str, Any]]
    candidate_id: Optional[int]
    skill_matches: Dict[str, Any]
    qualified: bool
    match_score: float
    error_message: Optional[str]
    processing_status: str
    email_sent: bool

class BulkCVProcessingState(TypedDict):
    """State for the complete bulk CV processing workflow"""
    # Job posting information
    job_title: str
    job_description: str
    required_skills: List[str]
    preferred_skills: List[str]
    min_experience: float
    department: str
    min_match_threshold: float
    job_id: Optional[int]
    batch_id: Optional[int]
    
    # Batch processing
    resume_files: List[str]
    total_resumes: int
    processed_resumes: int
    
    # Individual candidate states
    candidate_states: List[CandidateProcessingState]
    
    # Results aggregation with enhanced ranking
    qualified_candidates: List[Dict[str, Any]]
    all_candidates: List[Dict[str, Any]]
    candidate_tiers: Dict[str, List[Dict[str, Any]]]
    processing_summary: Dict[str, Any]
    
    # Email notifications
    emails_sent: int
    email_results: List[Dict[str, Any]]
    
    # Error handling and status
    error_message: Optional[str]
    processing_status: str
    db_connection: Any
    messages: List[Dict[str, Any]]
    
    # Processing metadata
    processing_time: Optional[float]
    workflow_stage: str
    start_time: float

# =================================================================
# PYDANTIC MODELS
# =================================================================

class JobPostingModel(BaseModel):
    job_title: str
    job_description: str
    required_skills: List[str]
    preferred_skills: List[str] = []
    min_experience: float = 0.0
    department: str = "General"
    min_match_threshold: float = 50.0

# =================================================================
# CONFIGURATION CLASSES
# =================================================================

class DatabaseConfig:
    """Database configuration management"""
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.database = database

    @classmethod
    def from_env(cls):
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", 3306),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "root"),
            database=os.getenv("DB_NAME", "CIH2")
        )

class EmailConfig:
    """Email configuration management"""
    def __init__(self, smtp_server, smtp_port, email_user, email_password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email_user = email_user
        self.email_password = email_password

    @classmethod
    def from_env(cls):
        return cls(
            smtp_server=os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('SMTP_PORT', 587)),
            email_user=os.getenv('SENDER_EMAIL', ''),
            email_password=os.getenv('SENDER_PASSWORD', '')
        )

class GroqConfig:
    """Groq API configuration"""
    def __init__(self, api_key: str, model_name: str = "llama3-70b-8192"):
        self.api_key = api_key
        self.model_name = model_name
    
    @classmethod
    def from_env(cls):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        model_name = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        return cls(api_key, model_name)

# =================================================================
# DOCUMENT PROCESSING UTILITIES
# =================================================================

class DocumentProcessor:
    """Enhanced document processing with error handling"""
    
    @staticmethod
    def read_word_file(path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(path)
            text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            logger.info(f"Successfully extracted {len(text)} characters from Word file")
            return text
        except Exception as e:
            logger.error(f"Error reading Word file {path}: {e}")
            raise

    @staticmethod
    def read_pdf_file(path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"--- Page {page_num + 1} ---\n{page_text}\n"
            logger.info(f"Successfully extracted {len(text)} characters from PDF file")
            return text
        except Exception as e:
            logger.error(f"Error reading PDF file {path}: {e}")
            raise

    @staticmethod
    def parse_file(path: str) -> str:
        """Parse file based on extension"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        path_lower = path.lower()
        if path_lower.endswith(".docx"):
            return DocumentProcessor.read_word_file(path)
        elif path_lower.endswith(".pdf"):
            return DocumentProcessor.read_pdf_file(path)
        else:
            raise ValueError(f"Unsupported file type: {path}. Must be .pdf or .docx")

    @staticmethod
    def extract_files_from_zip(zip_path: str, extract_dir: str) -> List[str]:
        """Extract resume files from ZIP archive"""
        extracted_files = []
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if not file_info.is_dir():
                        filename = file_info.filename
                        # Only extract PDF and DOCX files
                        if filename.lower().endswith(('.pdf', '.docx')):
                            # Extract with safe filename
                            safe_filename = os.path.basename(filename)
                            extract_path = os.path.join(extract_dir, safe_filename)
                            
                            # Handle duplicate filenames
                            counter = 1
                            base_name, ext = os.path.splitext(safe_filename)
                            while os.path.exists(extract_path):
                                safe_filename = f"{base_name}_{counter}{ext}"
                                extract_path = os.path.join(extract_dir, safe_filename)
                                counter += 1
                            
                            with zip_ref.open(file_info) as source, open(extract_path, 'wb') as target:
                                target.write(source.read())
                            
                            extracted_files.append(extract_path)
                            logger.info(f"Extracted: {safe_filename}")
            
            logger.info(f"Extracted {len(extracted_files)} resume files from ZIP")
            return extracted_files
        except Exception as e:
            logger.error(f"Error extracting ZIP file: {e}")
            raise

# =================================================================
# JSON EXTRACTION UTILITIES
# =================================================================

class JSONExtractor:
    """Enhanced JSON extraction from AI responses"""
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from AI response text with multiple strategies"""
        # Clean up the text
        text = re.sub(r'//.*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Try different JSON extraction patterns
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            r'\{.*?\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match.strip())
                    if JSONExtractor._is_valid_cv_data(parsed):
                        logger.info(f"Successfully extracted JSON with {len(parsed)} fields")
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        # Fallback: manual extraction
        return JSONExtractor._manual_extraction(text)
    
    @staticmethod
    def _is_valid_cv_data(data: Dict) -> bool:
        """Check if extracted data looks like valid CV data"""
        required_fields = ['name', 'email', 'skills']
        return any(field in data for field in required_fields)
    
    @staticmethod
    def _manual_extraction(text: str) -> Optional[Dict[str, Any]]:
        """Manual key-value extraction as fallback"""
        manual_data = {}
        patterns = {
            'name': r'"name"\s*:\s*"([^"]+)"',
            'email': r'"email"\s*:\s*"([^"]+)"',
            'phone': r'"phone"\s*:\s*"([^"]+)"',
            'role': r'"role"\s*:\s*"([^"]+)"',
            'summary': r'"summary"\s*:\s*"([^"]+)"',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                manual_data[key] = match.group(1)
        
        return manual_data if manual_data else None

# =================================================================
# ENHANCED SKILL MATCHING WITH COMPREHENSIVE RANKING
# =================================================================

class EnhancedSkillMatcher:
    """Enhanced skill matching with comprehensive ranking system"""
    def __init__(self):
        self.skill_aliases = {
            "js": "javascript", "reactjs": "react", "nodejs": "node.js",
            "expressjs": "express.js", "k8s": "kubernetes", "ai": "artificial intelligence",
            "ml": "machine learning", "ui/ux": "ui/ux design", "qa": "quality assurance",
            "rest": "rest api", "gcp": "google cloud platform", "aws": "amazon web services"
        }

    def _normalize_skill(self, skill):
        """Normalize skill string"""
        normalized = re.sub(r'[^\w\s]', '', skill).lower().strip()
        return self.skill_aliases.get(normalized, normalized)

    def find_skill_matches(self, candidate_skills, required_skills):
        """Find matches between candidate and required skills"""
        matches = []
        normalized_candidate = [self._normalize_skill(s) for s in candidate_skills]
        normalized_required = [self._normalize_skill(s) for s in required_skills]

        for req_skill_orig, req_skill_norm in zip(required_skills, normalized_required):
            for cand_skill_orig, cand_skill_norm in zip(candidate_skills, normalized_candidate):
                if req_skill_norm == cand_skill_norm:
                    matches.append({
                        'candidate_skill': cand_skill_orig,
                        'required_skill': req_skill_orig,
                        'match_type': 'exact'
                    })
                    break
        return matches

    def calculate_match_score(self, matches, total_required_skills):
        """Calculate match score"""
        if total_required_skills == 0:
            return 100.0, "No required skills specified"

        matched_count = len(matches)
        score = (matched_count / total_required_skills) * 100.0
        
        if matches:
            matched_names = [m['required_skill'] for m in matches]
            details = f"Matched {matched_count}/{total_required_skills}: {', '.join(matched_names)}"
        else:
            details = "No required skills matched"
        
        return score, details

    def calculate_comprehensive_ranking(self, candidate_data: Dict, job_requirements: Dict) -> Dict[str, Any]:
        """Calculate comprehensive ranking with multiple factors"""
        candidate_skills = candidate_data.get('skills', [])
        candidate_experience = float(candidate_data.get('total_experience', 0))
        
        required_skills = job_requirements['required_skills']
        preferred_skills = job_requirements.get('preferred_skills', [])
        min_experience = job_requirements.get('min_experience', 0.0)
        min_threshold = job_requirements.get('min_match_threshold', 50.0)
        
        # Required skills matching (70% weight)
        required_matches = self.find_skill_matches(candidate_skills, required_skills)
        required_score, required_details = self.calculate_match_score(
            required_matches, len(required_skills)
        )
        
        # Preferred skills matching (20% weight)
        preferred_matches = self.find_skill_matches(candidate_skills, preferred_skills)
        preferred_score, preferred_details = self.calculate_match_score(
            preferred_matches, len(preferred_skills)
        ) if preferred_skills else (0.0, "No preferred skills specified")
        
        # Experience factor (10% weight)
        experience_score = min(100.0, (candidate_experience / max(min_experience, 1.0)) * 100.0)
        
        # Overall weighted score calculation
        overall_score = (
            required_score * 0.7 +
            preferred_score * 0.2 +
            experience_score * 0.1
        )
        
        # Qualification determination
        meets_experience = candidate_experience >= min_experience
        meets_skill_threshold = required_score >= min_threshold
        is_qualified = meets_skill_threshold and meets_experience
        
        # Ranking tier determination
        ranking_tier = self._determine_ranking_tier(overall_score, is_qualified)
        
        # Additional scoring from candidate data
        ai_scores = candidate_data.get('scoring', {})
        
        evaluation = {
            "candidate_name": candidate_data.get("name", "Unknown"),
            "candidate_email": candidate_data.get("email", "No email"),
            "candidate_experience": candidate_experience,
            "candidate_role": candidate_data.get("role", "Not specified"),
            "candidate_location": candidate_data.get("location", "Not specified"),
            
            # Skill matching scores
            "required_skill_score": round(required_score, 1),
            "required_skill_details": required_details,
            "preferred_skill_score": round(preferred_score, 1),
            "preferred_skill_details": preferred_details,
            
            # Experience evaluation
            "experience_score": round(experience_score, 1),
            "meets_experience_requirement": meets_experience,
            
            # Overall evaluation
            "overall_match_score": round(overall_score, 1),
            "is_qualified": is_qualified,
            "ranking_tier": ranking_tier,
            
            # Matched skills lists
            "matched_required_skills": [m['candidate_skill'] for m in required_matches],
            "matched_preferred_skills": [m['candidate_skill'] for m in preferred_matches],
            "total_matched_skills_count": len(required_matches) + len(preferred_matches),
            
            # Additional AI scores if available
            "tech_score": ai_scores.get('tech_score'),
            "communication_score": ai_scores.get('communication_score'),
            "ai_fit_score": ai_scores.get('ai_fit_score'),
            "overall_ai_score": ai_scores.get('overall_score'),
            
            # Summary for recruiters
            "recruiter_summary": self._generate_recruiter_summary(
                candidate_data, required_matches, preferred_matches,
                overall_score, is_qualified
            )
        }
        
        return evaluation

    def _determine_ranking_tier(self, score: float, is_qualified: bool) -> str:
        """Determine ranking tier based on score and qualification status"""
        if not is_qualified:
            return "Not Qualified"
        elif score >= 90:
            return "Excellent Match"
        elif score >= 80:
            return "Very Good Match"
        elif score >= 70:
            return "Good Match"
        elif score >= 60:
            return "Fair Match"
        else:
            return "Minimal Match"

    def _generate_recruiter_summary(self, candidate_data: Dict, required_matches: List,
                                    preferred_matches: List, overall_score: float,
                                    is_qualified: bool) -> str:
        """Generate a summary for recruiters"""
        name = candidate_data.get("name", "Candidate")
        experience = candidate_data.get("total_experience", 0)
        role = candidate_data.get("role", "")
        
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"{name} is a {role} with {experience} years of experience.")
        
        # Skill matching summary
        req_count = len(required_matches)
        pref_count = len(preferred_matches)
        
        if req_count > 0:
            req_skills = [m['required_skill'] for m in required_matches[:3]]  # Show top 3
            skill_text = ", ".join(req_skills)
            if req_count > 3:
                skill_text += f" and {req_count - 3} more"
            summary_parts.append(f"Strong match in required skills: {skill_text}.")
        
        if pref_count > 0:
            pref_skills = [m['required_skill'] for m in preferred_matches[:2]]  # Show top 2
            summary_parts.append(f"Also has preferred skills: {', '.join(pref_skills)}.")
        
        # Overall assessment
        if is_qualified:
            if overall_score >= 85:
                summary_parts.append("This is a top-tier candidate recommended for immediate interview.")
            elif overall_score >= 70:
                summary_parts.append("This candidate shows strong potential and is recommended for consideration.")
            else:
                summary_parts.append("This candidate meets basic requirements.")
        else:
            summary_parts.append("This candidate does not meet the minimum requirements.")
        
        return " ".join(summary_parts)

    def rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank candidates with position numbers and tier grouping"""
        # Sort by overall match score (descending)
        ranked_candidates = sorted(candidates, key=lambda x: x['overall_match_score'], reverse=True)
        
        # Add ranking information
        for i, candidate in enumerate(ranked_candidates, 1):
            candidate['rank_position'] = i
            candidate['rank_display'] = f"#{i}"
            
            # Add percentile ranking
            total_candidates = len(ranked_candidates)
            percentile = ((total_candidates - i + 1) / total_candidates) * 100
            candidate['percentile_rank'] = round(percentile, 1)
        
        return ranked_candidates

    def group_candidates_by_tier(self, ranked_candidates: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group candidates by ranking tier"""
        tiers = {
            "Excellent Match": [],
            "Very Good Match": [],
            "Good Match": [],
            "Fair Match": [],
            "Minimal Match": [],
            "Not Qualified": []
        }
        
        for candidate in ranked_candidates:
            tier = candidate.get('ranking_tier', 'Not Qualified')
            tiers[tier].append(candidate)
        
        # Remove empty tiers
        return {tier: candidates for tier, candidates in tiers.items() if candidates}

# =================================================================
# DATABASE UTILITIES
# =================================================================

class DatabaseManager:
    """Enhanced database operations for bulk processing"""
    
    @staticmethod
    def get_connection():
        """Get database connection"""
        config = DatabaseConfig.from_env()
        return mysql.connector.connect(
            host=config.host,
            user=config.user,
            password=config.password,
            database=config.database,
            port=config.port
        )
    
    @staticmethod
    def create_job_posting(job_data: Dict[str, Any], db_connection) -> int:
        """Create job posting entry"""
        cursor = db_connection.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_postings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    job_title VARCHAR(255) NOT NULL,
                    job_description TEXT,
                    required_skills JSON,
                    preferred_skills JSON,
                    min_experience FLOAT DEFAULT 0,
                    department VARCHAR(100),
                    min_match_threshold FLOAT DEFAULT 50.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'active'
                )
            """)
            
            cursor.execute("""
                INSERT INTO job_postings 
                (job_title, job_description, required_skills, preferred_skills, 
                 min_experience, department, min_match_threshold)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                job_data['job_title'],
                job_data['job_description'],
                json.dumps(job_data['required_skills']),
                json.dumps(job_data.get('preferred_skills', [])),
                job_data.get('min_experience', 0.0),
                job_data.get('department', 'General'),
                job_data.get('min_match_threshold', 50.0)
            ))
            
            job_id = cursor.lastrowid
            db_connection.commit()
            logger.info(f"Created job posting with ID: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Failed to create job posting: {e}")
            db_connection.rollback()
            raise
        finally:
            cursor.close()

    @staticmethod
    def create_bulk_processing_batch(job_id: int, total_resumes: int, db_connection) -> int:
        """Create bulk processing batch record"""
        cursor = db_connection.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bulk_processing_batches (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    job_id INT,
                    total_resumes INT,
                    processed_resumes INT DEFAULT 0,
                    qualified_candidates INT DEFAULT 0,
                    emails_sent INT DEFAULT 0,
                    processing_status VARCHAR(50) DEFAULT 'started',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME NULL,
                    FOREIGN KEY (job_id) REFERENCES job_postings(id)
                )
            """)
            
            cursor.execute("""
                INSERT INTO bulk_processing_batches (job_id, total_resumes)
                VALUES (%s, %s)
            """, (job_id, total_resumes))
            
            batch_id = cursor.lastrowid
            db_connection.commit()
            logger.info(f"Created bulk processing batch with ID: {batch_id}")
            return batch_id
        except Exception as e:
            logger.error(f"Failed to create bulk processing batch: {e}")
            db_connection.rollback()
            raise
        finally:
            cursor.close()

    @staticmethod
    def update_batch_progress(batch_id: int, processed: int, qualified: int, emails_sent: int, status: str, db_connection):
        """Update bulk processing batch progress"""
        cursor = db_connection.cursor()
        try:
            if status == 'completed':
                cursor.execute("""
                    UPDATE bulk_processing_batches 
                    SET processed_resumes = %s, qualified_candidates = %s, 
                        emails_sent = %s, processing_status = %s,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (processed, qualified, emails_sent, status, batch_id))
            else:
                cursor.execute("""
                    UPDATE bulk_processing_batches 
                    SET processed_resumes = %s, qualified_candidates = %s, 
                        emails_sent = %s, processing_status = %s
                    WHERE id = %s
                """, (processed, qualified, emails_sent, status, batch_id))
            db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to update batch progress: {e}")
        finally:
            cursor.close()

    @staticmethod
    def insert_candidate_data(data: Dict[str, Any], db_connection, job_id: int = None, batch_id: int = None) -> int:
        """Insert structured CV data into database with job association"""
        if not data:
            raise ValueError("No data provided for insertion")
        
        cursor = db_connection.cursor()
        
        try:
            # Clean and prepare data
            name = data.get("name", "").replace("  ", " ").strip() if data.get("name") else None
            
            # Create candidates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255),
                    role VARCHAR(255),
                    summary TEXT,
                    email VARCHAR(255),
                    phone VARCHAR(50),
                    location VARCHAR(255),
                    portfolio_url VARCHAR(500),
                    github_url VARCHAR(500),
                    linkedin_url VARCHAR(500),
                    total_experience FLOAT DEFAULT 0,
                    education_gap BOOLEAN DEFAULT FALSE,
                    work_gap BOOLEAN DEFAULT FALSE,
                    job_id INT,
                    batch_id INT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATE,
                    FOREIGN KEY (job_id) REFERENCES job_postings(id) ON DELETE SET NULL,
                    FOREIGN KEY (batch_id) REFERENCES bulk_processing_batches(id) ON DELETE SET NULL
                )
            """)
            
            # Insert candidate basic info with job association
            candidate_query = """
                INSERT INTO candidates (name, role, summary, email, phone, location,
                                        portfolio_url, github_url, linkedin_url, total_experience,
                                        education_gap, work_gap, last_updated, job_id, batch_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURDATE(), %s, %s)
            """
            
            candidate_data = (
                name,
                data.get("role"),
                data.get("summary"),
                data.get("email"),
                data.get("phone"),
                data.get("location"),
                data.get("portfolio_url"),
                data.get("github_url"),
                data.get("linkedin_url"),
                data.get("total_experience"),
                data.get("education_gap", False),
                data.get("work_gap", False),
                job_id,
                batch_id
            )
            
            cursor.execute(candidate_query, candidate_data)
            candidate_id = cursor.lastrowid
            
            # Insert related data
            DatabaseManager._insert_experience(cursor, candidate_id, data.get("experience", []))
            DatabaseManager._insert_education(cursor, candidate_id, data.get("education", []))
            DatabaseManager._insert_skills(cursor, candidate_id, data.get("skills", []))
            DatabaseManager._insert_projects(cursor, candidate_id, data.get("projects", []))
            DatabaseManager._insert_soft_skills(cursor, candidate_id, data.get("soft_skills", []))
            DatabaseManager._insert_scoring(cursor, candidate_id, data.get("scoring", {}))
            
            db_connection.commit()
            logger.info(f"Successfully inserted candidate data with ID: {candidate_id}")
            return candidate_id
            
        except Exception as e:
            logger.error(f"Database insertion error: {e}")
            db_connection.rollback()
            raise
        finally:
            cursor.close()

    @staticmethod
    def _insert_experience(cursor, candidate_id: int, experiences: List[Dict]):
        """Insert experience data"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experience (
                id INT AUTO_INCREMENT PRIMARY KEY,
                candidate_id INT,
                title VARCHAR(255),
                company VARCHAR(255),
                start_date DATE,
                end_date DATE,
                description TEXT,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        """)
        
        for exp in experiences:
            try:
                start_date = DatabaseManager._parse_date(exp.get("start_date"))
                end_date = DatabaseManager._parse_date(exp.get("end_date")) if exp.get("end_date") != "Present" else None
                
                cursor.execute("""
                    INSERT INTO experience (candidate_id, title, company, start_date, end_date, description)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    candidate_id,
                    exp.get("title"),
                    exp.get("company") if exp.get("company") else "Freelance/Self-employed",
                    start_date,
                    end_date,
                    exp.get("description")
                ))
            except Exception as e:
                logger.warning(f"Error inserting experience {exp.get('title', 'Unknown')}: {e}")

    @staticmethod
    def _insert_education(cursor, candidate_id: int, educations: List[Dict]):
        """Insert education data"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS education (
                id INT AUTO_INCREMENT PRIMARY KEY,
                candidate_id INT,
                institute VARCHAR(255),
                degree VARCHAR(255),
                start_date DATE,
                end_date DATE,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        """)
        
        for edu in educations:
            try:
                start_date = DatabaseManager._parse_date(edu.get("start_date"))
                end_date = DatabaseManager._parse_date(edu.get("end_date"))
                
                cursor.execute("""
                    INSERT INTO education (candidate_id, institute, degree, start_date, end_date)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    candidate_id,
                    edu.get("institute"),
                    edu.get("degree"),
                    start_date,
                    end_date
                ))
            except Exception as e:
                logger.warning(f"Error inserting education {edu.get('institute', 'Unknown')}: {e}")

    @staticmethod
    def _insert_skills(cursor, candidate_id: int, skills: List[str]):
        """Insert skills data"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id INT AUTO_INCREMENT PRIMARY KEY,
                candidate_id INT,
                skill VARCHAR(255),
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        """)
        
        for skill in skills:
            try:
                cursor.execute("""
                    INSERT INTO skills (candidate_id, skill)
                    VALUES (%s, %s)
                """, (candidate_id, skill))
            except Exception as e:
                logger.warning(f"Error inserting skill {skill}: {e}")

    @staticmethod
    def _insert_projects(cursor, candidate_id: int, projects: List[Dict]):
        """Insert projects data"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INT AUTO_INCREMENT PRIMARY KEY,
                candidate_id INT,
                title VARCHAR(255),
                description TEXT,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        """)
        
        for project in projects:
            try:
                title = DatabaseManager._clean_text(project.get("title", ""))[:150]
                description = DatabaseManager._clean_text(project.get("description", ""))
                
                cursor.execute("""
                    INSERT INTO projects (candidate_id, title, description)
                    VALUES (%s, %s, %s)
                """, (candidate_id, title, description))
            except Exception as e:
                logger.warning(f"Error inserting project {project.get('title', 'Unknown')}: {e}")

    @staticmethod
    def _insert_soft_skills(cursor, candidate_id: int, soft_skills: List[Dict]):
        """Insert soft skills data"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS soft_skills (
                id INT AUTO_INCREMENT PRIMARY KEY,
                candidate_id INT,
                skill VARCHAR(255),
                strength_level VARCHAR(50),
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        """)
        
        for soft_skill in soft_skills:
            try:
                cursor.execute("""
                    INSERT INTO soft_skills (candidate_id, skill, strength_level)
                    VALUES (%s, %s, %s)
                """, (
                    candidate_id,
                    soft_skill.get("skill"),
                    soft_skill.get("strength_level")
                ))
            except Exception as e:
                logger.warning(f"Error inserting soft skill {soft_skill.get('skill', 'Unknown')}: {e}")

    @staticmethod
    def _insert_scoring(cursor, candidate_id: int, scoring: Dict):
        """Insert scoring metrics"""
        if scoring and any(scoring.values()):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scoring_metrics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    candidate_id INT,
                    tech_score FLOAT,
                    communication_score FLOAT,
                    ai_fit_score FLOAT,
                    overall_score FLOAT,
                    evaluated_by_ai BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
                )
            """)
            
            try:
                cursor.execute("""
                    INSERT INTO scoring_metrics (candidate_id, tech_score, communication_score,
                                                 ai_fit_score, overall_score, evaluated_by_ai)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    candidate_id,
                    scoring.get("tech_score"),
                    scoring.get("communication_score"),
                    scoring.get("ai_fit_score"),
                    scoring.get("overall_score"),
                    True
                ))
            except Exception as e:
                logger.warning(f"Error inserting scoring metrics: {e}")

    @staticmethod
    def _parse_date(date_str):
        """Parse various date formats"""
        if not date_str or date_str in ["Present", "Current", None]:
            return None
        
        date_formats = [
            "%Y-%m-%d", "%Y-%m", "%Y", "%d/%m/%Y", "%m/%d/%Y"
        ]
        
        date_str = str(date_str).strip()
        
        for date_format in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, date_format)
                return parsed_date.date()
            except ValueError:
                continue
        
        # Try to extract year only
        try:
            year = int(date_str)
            if 1900 <= year <= 2100:
                return datetime(year, 1, 1).date()
        except (ValueError, TypeError):
            pass
        
        logger.warning(f"Could not parse date: {date_str}")
        return None

    @staticmethod
    def _clean_text(text):
        """Clean text by removing problematic Unicode characters"""
        if not text:
            return ""
        
        text = str(text)
        # Replace common Unicode characters
        replacements = {
            '\ud83c\udfc6': '🏆', '\u2013': '-', '\u2014': '--',
            '\u2019': "'", '\u201c': '"', '\u201d': '"'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()

    @staticmethod
    def store_assessment_link(db_connection, assessment_uuid, candidate_id, job_title, candidate_email):
        """Store assessment link in database"""
        cursor = db_connection.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assessment (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    assessment_uuid VARCHAR(255) UNIQUE NOT NULL,
                    candidate_id VARCHAR(255) NOT NULL,
                    candidate_email VARCHAR(255) NOT NULL,
                    job_title VARCHAR(255) NOT NULL,
                    status VARCHAR(50) DEFAULT 'created',
                    screening_email_sent BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    start_time DATETIME NULL,
                    end_time DATETIME NULL,
                    score DECIMAL(5,2) NULL
                )
            """)
            
            cursor.execute("""
                INSERT INTO assessment 
                (assessment_uuid, candidate_id, candidate_email, job_title)
                VALUES (%s, %s, %s, %s)
            """, (assessment_uuid, candidate_id, candidate_email, job_title))
            
            db_connection.commit()
            logger.info(f"Assessment link stored: {assessment_uuid}")
            return True
        except Exception as e:
            logger.error(f"Failed to store assessment link: {e}")
            db_connection.rollback()
            return False
        finally:
            cursor.close()

# =================================================================
# EMAIL UTILITIES
# =================================================================

class EmailSender:
    """Enhanced email sending functionality for bulk operations"""
    
    def __init__(self, config: EmailConfig):
        self.config = config
    
    @classmethod
    def from_env(cls):
        return cls(EmailConfig.from_env())
    
    def send_bulk_test_invitations(self, qualified_candidates: List[Dict], job_info: Dict) -> List[Dict[str, Any]]:
        """Send test invitations to multiple qualified candidates"""
        email_results = []
        
        if not self.config.email_user or not self.config.email_password:
            logger.warning("Email credentials not configured")
            return email_results
        
        for candidate in qualified_candidates:
            try:
                assessment_uuid = str(uuid.uuid4())
                success = self.send_test_invitation(candidate, job_info, assessment_uuid)
                
                email_results.append({
                    'candidate_name': candidate.get('candidate_name'),
                    'candidate_email': candidate.get('candidate_email'),
                    'assessment_uuid': assessment_uuid,
                    'email_sent': success,
                    'job_title': job_info['job_title']
                })
                
            except Exception as e:
                logger.error(f"Failed to send email to {candidate.get('candidate_email', 'unknown')}: {e}")
                email_results.append({
                    'candidate_name': candidate.get('candidate_name'),
                    'candidate_email': candidate.get('candidate_email'),
                    'assessment_uuid': None,
                    'email_sent': False,
                    'error': str(e),
                    'job_title': job_info['job_title']
                })
        
        return email_results
    
    def send_test_invitation(self, candidate_data, job_info, assessment_uuid):
        """Send test invitation email to individual candidate"""
        if not self.config.email_user or not self.config.email_password:
            logger.warning("Email credentials not configured")
            return False
        
        recipient_email = candidate_data.get("candidate_email")
        candidate_name = candidate_data.get("candidate_name", "Candidate")
        job_title = job_info.get("job_title", "Position")
        
        # Create unique test link
        unique_test_link = f"http://localhost:4000/assessment/{assessment_uuid}"
        
        subject = f"Technical Assessment Invitation - {job_title}"
        
        body = f"""
Dear {candidate_name},

Thank you for your interest in the {job_title} position.

Based on your qualifications, we would like to invite you to complete a technical assessment.

Please click the link below to start your assessment:
{unique_test_link}

This link is unique to your assessment. Please do not share it.

Best regards,
The Hiring Team
"""
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_user
            msg['To'] = recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_user, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent to {recipient_email} for {job_title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {recipient_email}: {e}")
            return False

# =================================================================
# ENHANCED LANGGRAPH WORKFLOW NODES
# =================================================================

class EnhancedBulkCVWorkflowNodes:
    """Enhanced LangGraph workflow nodes with comprehensive ranking"""
    
    def __init__(self):
        # Initialize Groq model
        groq_config = GroqConfig.from_env()
        
        self.llm = ChatGroq(
            groq_api_key=groq_config.api_key,
            model_name=groq_config.model_name,
            temperature=0.1,
            max_tokens=4000,
            timeout=60
        )
        
        self.skill_matcher = EnhancedSkillMatcher()
        self.email_sender = EmailSender.from_env()
        
        logger.info(f"Initialized EnhancedBulkCVWorkflowNodes with ranking system")
    
    def initialize_batch_node(self, state: BulkCVProcessingState) -> BulkCVProcessingState:
        """Node: Initialize batch processing"""
        try:
            logger.info(f"Initializing bulk processing for {state['total_resumes']} resumes")
            state['workflow_stage'] = "initializing_batch"
            state['processed_resumes'] = 0
            state['candidate_states'] = []
            state['qualified_candidates'] = []
            state['all_candidates'] = []
            state['candidate_tiers'] = {}
            state['emails_sent'] = 0
            state['email_results'] = []
            state['processing_status'] = "batch_initialized"
            
            # Create job posting in database
            job_data = {
                'job_title': state['job_title'],
                'job_description': state['job_description'],
                'required_skills': state['required_skills'],
                'preferred_skills': state['preferred_skills'],
                'min_experience': state['min_experience'],
                'department': state['department'],
                'min_match_threshold': state['min_match_threshold']
            }
            
            job_id = DatabaseManager.create_job_posting(job_data, state['db_connection'])
            state['job_id'] = job_id
            
            # Create batch record
            batch_id = DatabaseManager.create_bulk_processing_batch(
                job_id, state['total_resumes'], state['db_connection']
            )
            state['batch_id'] = batch_id
            
            state['messages'].append({
                'type': 'system',
                'content': f"Initialized batch processing for {state['total_resumes']} resumes against job: {state['job_title']}"
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Batch initialization failed: {e}")
            state['error_message'] = str(e)
            state['processing_status'] = "failed"
            return state
    
    def process_resumes_batch_node(self, state: BulkCVProcessingState) -> BulkCVProcessingState:
        """Node: Process all resumes in parallel batches"""
        try:
            logger.info(f"Processing {len(state['resume_files'])} resume files")
            state['workflow_stage'] = "processing_resumes"
            
            # Process resumes in smaller batches to avoid overwhelming the system
            batch_size = int(os.getenv("PARALLEL_BATCH_SIZE", 3))  # Reduced for Groq rate limits
            resume_files = state['resume_files']
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                # Process files in batches
                for i in range(0, len(resume_files), batch_size):
                    batch_files = resume_files[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_files)} files")
                    
                    # Submit batch for processing
                    futures = []
                    for file_path in batch_files:
                        future = executor.submit(self._process_single_resume, file_path, state)
                        futures.append(future)
                    
                    # Collect results
                    for future in futures:
                        try:
                            candidate_state = future.result(timeout=180)  # 3 minute timeout per resume
                            if candidate_state:
                                state['candidate_states'].append(candidate_state)
                                state['processed_resumes'] += 1
                                
                                # Update progress in database
                                DatabaseManager.update_batch_progress(
                                    state['batch_id'],
                                    state['processed_resumes'],
                                    len(state['qualified_candidates']),
                                    0,  # emails not sent yet
                                    "processing",
                                    state['db_connection']
                                )
                                
                        except Exception as e:
                            logger.error(f"Failed to process resume: {e}")
                            # Create failed candidate state
                            state['candidate_states'].append({
                                'file_path': 'unknown',
                                'file_name': 'unknown',
                                'error_message': str(e),
                                'processing_status': 'failed',
                                'qualified': False
                            })
                            state['processed_resumes'] += 1
            
            state['processing_status'] = "resumes_processed"
            state['messages'].append({
                'type': 'system',
                'content': f"Processed {state['processed_resumes']}/{state['total_resumes']} resumes"
            })
            
            logger.info(f"Completed processing {state['processed_resumes']} resumes")
            return state
            
        except Exception as e:
            logger.error(f"Batch resume processing failed: {e}")
            state['error_message'] = str(e)
            state['processing_status'] = "failed"
            return state
    
    def _process_single_resume(self, file_path: str, batch_state: BulkCVProcessingState) -> CandidateProcessingState:
        """Process a single resume file"""
        candidate_state: CandidateProcessingState = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'raw_cv_text': '',
            'structured_data': None,
            'candidate_id': None,
            'skill_matches': {},
            'qualified': False,
            'match_score': 0.0,
            'error_message': None,
            'processing_status': 'started',
            'email_sent': False
        }
        
        try:
            # Step 1: Parse document
            logger.info(f"Processing resume: {candidate_state['file_name']}")
            raw_text = DocumentProcessor.parse_file(file_path)
            candidate_state['raw_cv_text'] = raw_text
            
            # Step 2: Extract structured data using Groq AI
            structured_data = self._extract_structured_data(raw_text)
            if not structured_data:
                candidate_state['error_message'] = "Failed to extract structured data"
                candidate_state['processing_status'] = "failed"
                return candidate_state
            
            candidate_state['structured_data'] = structured_data
            
            # Step 3: Store in database
            candidate_id = DatabaseManager.insert_candidate_data(
                structured_data,
                batch_state['db_connection'],
                batch_state.get('job_id'),
                batch_state.get('batch_id')
            )
            candidate_state['candidate_id'] = candidate_id
            
            # Step 4: Evaluate against job requirements (basic evaluation here, enhanced in aggregate)
            job_requirements = {
                'job_title': batch_state['job_title'],
                'required_skills': batch_state['required_skills'],
                'preferred_skills': batch_state['preferred_skills'],
                'min_experience': batch_state['min_experience'],
                'min_match_threshold': batch_state.get('min_match_threshold', 50.0)
            }
            
            evaluation = self.skill_matcher.calculate_comprehensive_ranking(
                structured_data, job_requirements
            )
            
            candidate_state['skill_matches'] = evaluation
            candidate_state['qualified'] = evaluation['is_qualified']
            candidate_state['match_score'] = evaluation['overall_match_score']
            candidate_state['processing_status'] = 'completed'
            
            logger.info(f"Successfully processed {candidate_state['file_name']} - Qualified: {candidate_state['qualified']}")
            return candidate_state
            
        except Exception as e:
            logger.error(f"Error processing {candidate_state['file_name']}: {e}")
            candidate_state['error_message'] = str(e)
            candidate_state['processing_status'] = 'failed'
            return candidate_state
    
    def _extract_structured_data(self, cv_text: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from CV text using Groq AI"""
        try:
            # Create extraction prompt
            prompt = self._create_extraction_prompt(cv_text)
            
            # Call Groq with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    messages = [
                        SystemMessage(content="You are an expert CV parser. Extract information and return ONLY valid JSON with no additional text or explanations."),
                        HumanMessage(content=prompt)
                    ]
                    
                    response = self.llm.invoke(messages)
                    ai_response = response.content
                    
                    # Extract JSON from response
                    structured_data = JSONExtractor.extract_json_from_text(ai_response)
                    if structured_data:
                        return structured_data
                    
                    logger.warning(f"Attempt {attempt + 1}: Failed to extract valid JSON")
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}: Groq API error: {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2)  # Wait before retry
                        continue
            
            logger.error("Failed to extract structured data after all retries")
            return None
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return None
    
    def _get_most_common_skills(self, ranked_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get most common skills across all candidates"""
        all_skills = []
        for candidate in ranked_candidates:
            all_skills.extend(candidate.get('matched_required_skills', []))
            all_skills.extend(candidate.get('matched_preferred_skills', []))
        
        skill_counts = Counter(all_skills)
        return [{'skill': skill, 'count': count} for skill, count in skill_counts.most_common(10)]
    
    def _generate_recruiter_recommendations(self, qualified_candidates: List[Dict], 
                                            candidate_tiers: Dict, uncovered_skills: List) -> List[str]:
        """Generate actionable recommendations for recruiters"""
        recommendations = []
        
        if not qualified_candidates:
            recommendations.append("No qualified candidates found. Consider reviewing job requirements or expanding the candidate pool.")
        else:
            top_tier_count = len(candidate_tiers.get("Excellent Match", []))
            if top_tier_count > 0:
                recommendations.append(f"Schedule immediate interviews with {top_tier_count} top-tier candidate(s).")
            
            good_matches = len(candidate_tiers.get("Very Good Match", [])) + len(candidate_tiers.get("Good Match", []))
            if good_matches > 0:
                recommendations.append(f"Consider {good_matches} additional candidate(s) for second round evaluation.")
        
        if uncovered_skills:
            recommendations.append(f"Skills gap identified: {', '.join(uncovered_skills[:3])}. Consider skills training or adjusted requirements.")
        
        return recommendations
    
    def aggregate_results_node(self, state: BulkCVProcessingState) -> BulkCVProcessingState:
        """Enhanced Node: Aggregate and rank processing results"""
        try:
            logger.info("Aggregating and ranking processing results")
            state['workflow_stage'] = "aggregating_results"
            
            # Calculate comprehensive rankings for all candidates
            enhanced_candidates = []
            job_requirements = {
                'job_title': state['job_title'],
                'required_skills': state['required_skills'],
                'preferred_skills': state['preferred_skills'],
                'min_experience': state['min_experience'],
                'min_match_threshold': state.get('min_match_threshold', 50.0)
            }
            
            for candidate_state in state['candidate_states']:
                if candidate_state.get('structured_data') and candidate_state['processing_status'] == 'completed':
                    # Calculate comprehensive ranking
                    evaluation = self.skill_matcher.calculate_comprehensive_ranking(
                        candidate_state['structured_data'],
                        job_requirements
                    )
                    enhanced_candidates.append(evaluation)
            
            # Rank all candidates
            ranked_candidates = self.skill_matcher.rank_candidates(enhanced_candidates)
            
            # Separate qualified and all candidates (both ranked)
            qualified_candidates = [c for c in ranked_candidates if c['is_qualified']]
            
            # Group candidates by tier
            candidate_tiers = self.skill_matcher.group_candidates_by_tier(ranked_candidates)
            
            # Update state with ranked results
            state['qualified_candidates'] = qualified_candidates
            state['all_candidates'] = ranked_candidates
            state['candidate_tiers'] = candidate_tiers
            
            # Calculate enhanced summary statistics
            total_processed = len(state['candidate_states'])
            successful_processing = len(ranked_candidates)
            failed_processing = total_processed - successful_processing
            qualified_count = len(qualified_candidates)
            
            # Top performer analysis
            top_candidate = ranked_candidates[0] if ranked_candidates else None
            top_tier_count = len(candidate_tiers.get("Excellent Match", []))
            
            # Skill gap analysis
            all_required_skills = set(state['required_skills'])
            covered_skills = set()
            for candidate in ranked_candidates:
                covered_skills.update(candidate['matched_required_skills'])
            uncovered_skills = all_required_skills - covered_skills
            
            # Create enhanced processing summary
            state['processing_summary'] = {
                'total_resumes_submitted': state['total_resumes'],
                'total_resumes_processed': total_processed,
                'successful_processing': successful_processing,
                'failed_processing': failed_processing,
                'qualified_candidates': qualified_count,
                'qualification_rate': round((qualified_count / successful_processing * 100), 1) if successful_processing > 0 else 0,
                
                # Ranking statistics
                'top_tier_candidates': top_tier_count,
                'average_match_score': round(sum([c['overall_match_score'] for c in ranked_candidates]) / len(ranked_candidates), 1) if ranked_candidates else 0,
                'highest_score': ranked_candidates[0]['overall_match_score'] if ranked_candidates else 0,
                'lowest_score': ranked_candidates[-1]['overall_match_score'] if ranked_candidates else 0,
                'median_score': round(ranked_candidates[len(ranked_candidates)//2]['overall_match_score'], 1) if ranked_candidates else 0,
                
                # Skill coverage analysis
                'skill_coverage_percentage': round((len(covered_skills) / len(all_required_skills) * 100), 1) if all_required_skills else 100,
                'uncovered_skills': list(uncovered_skills),
                'most_common_skills': self._get_most_common_skills(ranked_candidates),
                
                # Top candidate info
                'top_candidate': {
                    'name': top_candidate['candidate_name'],
                    'score': top_candidate['overall_match_score'],
                    'tier': top_candidate['ranking_tier'],
                    'email': top_candidate['candidate_email']
                } if top_candidate else None,
                
                # Tier distribution
                'tier_distribution': {tier: len(candidates) for tier, candidates in candidate_tiers.items()},
                
                'job_title': state['job_title'],
                'job_requirements': {
                    'required_skills': state['required_skills'],
                    'preferred_skills': state['preferred_skills'],
                    'min_experience': state['min_experience']
                },
                
                # Recommendations for recruiter
                'recruiter_recommendations': self._generate_recruiter_recommendations(
                    qualified_candidates, candidate_tiers, list(uncovered_skills)
                )
            }
            
            state['processing_status'] = "results_aggregated"
            state['messages'].append({
                'type': 'system',
                'content': f"Results ranked and aggregated: {qualified_count}/{successful_processing} candidates qualified. Top score: {state['processing_summary']['highest_score']}"
            })
            
            logger.info(f"Enhanced aggregation completed: {qualified_count} qualified, top score: {state['processing_summary']['highest_score']}")
            return state
            
        except Exception as e:
            logger.error(f"Enhanced result aggregation failed: {e}")
            state['error_message'] = str(e)
            state['processing_status'] = "failed"
            return state
    
    def send_bulk_notifications_node(self, state: BulkCVProcessingState) -> BulkCVProcessingState:
        """Node: Send bulk email notifications to qualified candidates"""
        try:
            logger.info("Sending bulk email notifications")
            state['workflow_stage'] = "sending_notifications"
            
            qualified_candidates = state.get('qualified_candidates', [])
            if not qualified_candidates:
                logger.info("No qualified candidates to notify")
                state['processing_status'] = "notifications_skipped"
                return state
            
            # Prepare job info for email
            job_info = {
                'job_title': state['job_title'],
                'job_description': state['job_description']
            }
            
            # Send bulk email invitations
            email_results = self.email_sender.send_bulk_test_invitations(qualified_candidates, job_info)
            
            # Store assessment links in database and count successful emails
            emails_sent_count = 0
            for result in email_results:
                if result.get('email_sent') and result.get('assessment_uuid'):
                    candidate_id = None
                    # Find candidate ID from qualified candidates
                    for candidate in qualified_candidates:
                        if candidate.get('candidate_email') == result.get('candidate_email'):
                            # Get candidate_id from the candidate_states if available
                            for cs in state.get('candidate_states', []):
                                if (cs.get('structured_data', {}).get('email') == result.get('candidate_email') and
                                    cs.get('candidate_id')):
                                    candidate_id = cs['candidate_id']
                                    break
                            break
                    
                    if candidate_id:
                        DatabaseManager.store_assessment_link(
                            state['db_connection'],
                            result['assessment_uuid'],
                            str(candidate_id),
                            result['job_title'],
                            result['candidate_email']
                        )
                        emails_sent_count += 1
            
            state['emails_sent'] = emails_sent_count
            state['email_results'] = email_results
            state['processing_status'] = "notifications_sent"
            
            # Update batch progress
            DatabaseManager.update_batch_progress(
                state['batch_id'],
                state['processed_resumes'],
                len(state['qualified_candidates']),
                emails_sent_count,
                "notifications_sent",
                state['db_connection']
            )
            
            state['messages'].append({
                'type': 'system',
                'content': f"Sent {emails_sent_count} email notifications out of {len(state['qualified_candidates'])} qualified candidates."
            })
            
            logger.info(f"Email notifications completed. Sent {emails_sent_count} emails.")
            return state
            
        except Exception as e:
            logger.error(f"Bulk email notification failed: {e}")
            state['error_message'] = str(e)
            state['processing_status'] = "failed"
            return state
    
    def finalize_bulk_processing_node(self, state: BulkCVProcessingState) -> BulkCVProcessingState:
        """Node: Finalize bulk processing and prepare results"""
        try:
            logger.info("Finalizing bulk CV processing")
            state['workflow_stage'] = "finalizing"
            
            # Calculate processing time
            if state['start_time']:
                processing_time = datetime.now().timestamp() - state['start_time']
                state['processing_time'] = processing_time
            
            # Update final batch status
            DatabaseManager.update_batch_progress(
                state['batch_id'],
                state['processed_resumes'],
                len(state['qualified_candidates']),
                state['emails_sent'],
                "completed",
                state['db_connection']
            )
            
            state['processing_status'] = "completed"
            
            state['messages'].append({
                'type': 'system',
                'content': "Bulk CV processing workflow completed successfully."
            })
            
            logger.info("Bulk CV processing workflow completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Finalization failed: {e}")
            state['error_message'] = str(e)
            state['processing_status'] = "failed"
            return state
    
    def error_handler_node(self, state: BulkCVProcessingState) -> BulkCVProcessingState:
        """Node: Handle errors and cleanup"""
        logger.error(f"Bulk processing failed at stage '{state.get('workflow_stage', 'unknown')}': {state.get('error_message', 'Unknown error')}")
        state['processing_status'] = "failed"
        state['workflow_stage'] = "error_handling"
        
        # Update batch status to failed
        if state.get('batch_id'):
            try:
                DatabaseManager.update_batch_progress(
                    state['batch_id'],
                    state.get('processed_resumes', 0),
                    len(state.get('qualified_candidates', [])),
                    state.get('emails_sent', 0),
                    "failed",
                    state['db_connection']
                )
            except Exception as e:
                logger.error(f"Failed to update batch status: {e}")
        
        return state
    
    def _create_extraction_prompt(self, cv_text: str) -> str:
        """Create the prompt for structured data extraction"""
        # Truncate CV text if too long for Groq context
        max_length = 6000  # Conservative limit for Groq
        if len(cv_text) > max_length:
            cv_text = cv_text[:max_length] + "\n[... content truncated ...]"
        
        return f"""Extract information from this resume and return it in this EXACT JSON format. Return ONLY the JSON, no additional text:

{{
    "name": "Full Name",
    "role": "Job Title/Role",
    "email": "email@domain.com",
    "phone": "phone number",
    "location": "city, country",
    "github_url": "github link or null",
    "linkedin_url": "linkedin link or null",
    "portfolio_url": "portfolio link or null",
    "summary": "Brief professional summary",
    "total_experience": 5.5,
    "education_gap": false,
    "work_gap": false,
    "education": [
        {{
            "institute": "University Name",
            "degree": "Degree Name",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01"
        }}
    ],
    "experience": [
        {{
            "title": "Job Title",
            "company": "Company Name",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "description": "Job description"
        }}
    ],
    "skills": ["skill1", "skill2", "skill3"],
    "soft_skills": [
        {{
            "skill": "Communication",
            "strength_level": "High"
        }}
    ],
    "projects": [
        {{
            "title": "Project Name",
            "description": "Project description"
        }}
    ],
    "scoring": {{
        "tech_score": 8.5,
        "communication_score": 7.0,
        "ai_fit_score": 8.0,
        "overall_score": 7.8
    }}
}}

Resume Content:
{cv_text}"""

# =================================================================
# ROUTING LOGIC FOR BULK PROCESSING
# =================================================================

def bulk_workflow_router(state: BulkCVProcessingState) -> str:
    """Determine next step based on current state"""
    status = state.get('processing_status', '')
    
    if status == "failed":
        return "error_handler"
    elif status == "batch_initialized":
        return "process_resumes"
    elif status == "resumes_processed":
        return "aggregate_results"
    elif status == "results_aggregated":
        return "send_notifications"
    elif status in ["notifications_sent", "notifications_skipped"]:
        return "finalize"
    elif status == "completed":
        return END
    else:
        return "initialize_batch"

# =================================================================
# MAIN ENHANCED LANGGRAPH BULK PROCESSING AGENT
# =================================================================

class EnhancedLangGraphBulkCVAgent:
    """Enhanced LangGraph-based Bulk CV Processing Agent with Comprehensive Ranking"""
    
    def __init__(self):
        self.nodes = EnhancedBulkCVWorkflowNodes()
        self.graph = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the enhanced LangGraph workflow for bulk processing"""
        workflow = StateGraph(BulkCVProcessingState)
        
        # Add all workflow nodes
        workflow.add_node("initialize_batch", self.nodes.initialize_batch_node)
        workflow.add_node("process_resumes", self.nodes.process_resumes_batch_node)
        workflow.add_node("aggregate_results", self.nodes.aggregate_results_node)
        workflow.add_node("send_notifications", self.nodes.send_bulk_notifications_node)
        workflow.add_node("finalize", self.nodes.finalize_bulk_processing_node)
        workflow.add_node("error_handler", self.nodes.error_handler_node)
        
        # Set entry point
        workflow.set_entry_point("initialize_batch")
        
        # Add conditional routing between nodes
        workflow.add_conditional_edges(
            "initialize_batch",
            bulk_workflow_router,
            {
                "process_resumes": "process_resumes",
                "error_handler": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "process_resumes",
            bulk_workflow_router,
            {
                "aggregate_results": "aggregate_results",
                "error_handler": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "aggregate_results",
            bulk_workflow_router,
            {
                "send_notifications": "send_notifications",
                "error_handler": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "send_notifications",
            bulk_workflow_router,
            {
                "finalize": "finalize",
                "error_handler": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "finalize",
            bulk_workflow_router,
            {
                END: END,
                "error_handler": "error_handler"
            }
        )
        
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    def process_bulk_resumes(self, resume_files: List[str], job_posting: JobPostingModel, db_connection=None) -> Dict[str, Any]:
        """Process multiple resumes through the complete enhanced LangGraph workflow"""
        
        # Initialize database connection if not provided
        if db_connection is None:
            db_connection = DatabaseManager.get_connection()
        
        # Initialize state
        initial_state: BulkCVProcessingState = {
            "job_title": job_posting.job_title,
            "job_description": job_posting.job_description,
            "required_skills": job_posting.required_skills,
            "preferred_skills": job_posting.preferred_skills,
            "min_experience": job_posting.min_experience,
            "department": job_posting.department,
            "min_match_threshold": job_posting.min_match_threshold,
            "resume_files": resume_files,
            "total_resumes": len(resume_files),
            "processed_resumes": 0,
            "candidate_states": [],
            "qualified_candidates": [],
            "all_candidates": [],
            "candidate_tiers": {},
            "processing_summary": {},
            "emails_sent": 0,
            "email_results": [],
            "error_message": None,
            "processing_status": "initialized",
            "db_connection": db_connection,
            "messages": [],
            "processing_time": None,
            "workflow_stage": "initializing",
            "start_time": datetime.now().timestamp(),
            "job_id": None,
            "batch_id": None
        }
        
        try:
            logger.info(f"Starting enhanced bulk CV processing with ranking for {len(resume_files)} resumes against job: {job_posting.job_title}")
            
            # Run the complete workflow
            final_state = self.graph.invoke(initial_state)
            
            # Prepare comprehensive result with enhanced ranking
            result = {
                "success": final_state["processing_status"] == "completed",
                "job_title": final_state["job_title"],
                "job_id": final_state.get("job_id"),
                "batch_id": final_state.get("batch_id"),
                "processing_summary": final_state.get("processing_summary", {}),
                
                # Enhanced ranking results
                "qualified_candidates": final_state.get("qualified_candidates", []),
                "all_candidates": final_state.get("all_candidates", []),  # Already ranked
                "candidate_tiers": final_state.get("candidate_tiers", {}),
                "ranking_metadata": {
                    "total_ranked": len(final_state.get("all_candidates", [])),
                    "ranking_criteria": ["Required Skills Match", "Preferred Skills Match", "Experience Level"],
                    "tier_system": ["Excellent Match", "Very Good Match", "Good Match", "Fair Match", "Minimal Match", "Not Qualified"]
                },
                
                "candidate_states": final_state.get("candidate_states", []),
                "emails_sent": final_state.get("emails_sent", 0),
                "email_results": final_state.get("email_results", []),
                "error_message": final_state.get("error_message"),
                "processing_status": final_state["processing_status"],
                "workflow_stage": final_state.get("workflow_stage"),
                "messages": final_state["messages"],
                "processing_time": final_state.get("processing_time"),
                "agent_type": "enhanced_langgraph_bulk_workflow_groq_with_ranking"
            }
            
            if result["success"]:
                    summary = result["processing_summary"]
                    # Safely access top_candidate details
                    top_candidate_info = summary.get('top_candidate') or {}
                    logger.info(f"Enhanced bulk CV processing completed successfully. Job: {result['job_title']}, "
                                f"Top candidate: {top_candidate_info.get('name', 'N/A')} "
                                f"(Score: {summary.get('highest_score', 0)})")
            else:
                logger.error(f"Enhanced bulk CV processing failed: {result['error_message']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in enhanced bulk CV processing: {e}")
            return {
                "success": False,
                "job_title": job_posting.job_title,
                "job_id": None,
                "batch_id": None,
                "processing_summary": {},
                "qualified_candidates": [],
                "all_candidates": [],
                "candidate_tiers": {},
                "ranking_metadata": {},
                "candidate_states": [],
                "emails_sent": 0,
                "email_results": [],
                "error_message": str(e),
                "processing_status": "failed",
                "workflow_stage": "error",
                "messages": [],
                "processing_time": None,
                "agent_type": "enhanced_langgraph_bulk_workflow_groq_with_ranking"
            }
        finally:
            # Close database connection if we created it
            if db_connection:
                try:
                    db_connection.close()
                except:
                    pass

# =================================================================
# ENHANCED FASTAPI APPLICATION
# =================================================================

app = FastAPI(
    title="Enhanced TalentFlow Pro - CV Processing System with Advanced Ranking",
    description="Advanced bulk CV processing with comprehensive ranking, tier classification, and detailed candidate evaluation",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent variable
enhanced_bulk_cv_agent = None

def get_enhanced_bulk_cv_agent():
    """Get or create enhanced bulk CV agent instance"""
    global enhanced_bulk_cv_agent
    if enhanced_bulk_cv_agent is None:
        enhanced_bulk_cv_agent = EnhancedLangGraphBulkCVAgent()
    return enhanced_bulk_cv_agent

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced TalentFlow Pro - CV Processing System with Advanced Ranking",
        "version": "4.0.0",
        "agent_type": "enhanced_langgraph_bulk_workflow_groq_with_ranking",
        "ai_provider": "Groq",
        "features": [
            "Bulk document parsing (PDF/DOCX/ZIP)",
            "AI-powered data extraction with Groq (ultra-fast inference)",
            "Advanced ranking system with tier classification",
            "Comprehensive candidate evaluation and scoring",
            "Database storage with batch tracking",
            "Job-specific candidate matching and scoring",
            "Recruiter recommendations and insights",
            "Bulk email notifications for qualified candidates",
            "Workflow orchestration with LangGraph",
            "Parallel processing with rate limiting",
            "Skill gap analysis and coverage metrics",
            "Percentile ranking and position tracking"
        ],
        "supported_models": [
            "llama3-70b-8192",
            "llama3-8b-8192", 
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ],
        "ranking_tiers": [
            "Excellent Match (90%+)",
            "Very Good Match (80-89%)",
            "Good Match (70-79%)",
            "Fair Match (60-69%)",
            "Minimal Match (50-59%)",
            "Not Qualified (<50%)"
        ],
        "endpoints": {
            "/process-resumes": "POST - Upload resumes and job details for enhanced bulk processing with ranking",
            "/health": "GET - Health check with system status",
            "/": "GET - API information and documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_connection = DatabaseManager.get_connection()
        db_connection.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Test Groq API
    try:
        groq_config = GroqConfig.from_env()
        ai_status = f"configured (model: {groq_config.model_name})"
    except Exception as e:
        ai_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "agent_type": "enhanced_langgraph_bulk_workflow_groq_with_ranking",
        "ai_provider": "Groq",
        "database": db_status,
        "ai_model": ai_status,
        "ranking_system": "enabled",
        "tier_classification": "enabled",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/process-resumes")
async def process_resumes_enhanced(
    job_details: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Enhanced endpoint: Upload and process multiple resumes with comprehensive ranking
    
    Features:
    - Advanced ranking system with tier classification
    - Comprehensive candidate evaluation
    - Skill gap analysis and coverage metrics  
    - Recruiter recommendations and insights
    - Percentile ranking and position tracking
    """
    temp_dir = None
    
    try:
        # Parse job details from form data
        try:
            job_data = json.loads(job_details)
            job_posting = JobPostingModel(**job_data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid job details format: {e}")
        
        if not files:
            raise HTTPException(status_code=422, detail="No resume files provided")
        
        temp_dir = tempfile.mkdtemp()
        resume_files = []
        
        for file in files:
            file_extension = Path(file.filename).suffix.lower()
            
            if file.size and file.size > 20 * 1024 * 1024:  # 20MB limit
                raise HTTPException(status_code=422, detail=f"File {file.filename} is too large. Maximum size is 20MB")
            
            temp_file_path = os.path.join(temp_dir, file.filename)
            with open(temp_file_path, "wb") as f:
                f.write(await file.read())
            
            if file_extension == '.zip':
                # Extract files from ZIP archive
                extracted = DocumentProcessor.extract_files_from_zip(temp_file_path, temp_dir)
                resume_files.extend(extracted)
                os.remove(temp_file_path)  # remove zip file after extraction
            elif file_extension in ['.pdf', '.docx']:
                resume_files.append(temp_file_path)
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported file type: {file.filename}. Only PDF, DOCX, and ZIP files are supported"
                )
        
        if not resume_files:
            raise HTTPException(status_code=422, detail="No valid resume files (PDF/DOCX) found in the upload")
        
        logger.info(f"Processing {len(resume_files)} resumes for job: {job_posting.job_title}")
        
        # Get enhanced agent and process resumes
        agent = get_enhanced_bulk_cv_agent()
        result = agent.process_bulk_resumes(resume_files, job_posting)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": result["success"],
                "message": "Enhanced bulk resume processing with comprehensive ranking completed using Groq AI",
                "data": result
            }
        )
        
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error", 
                "message": str(e),
                "agent_type": "enhanced_langgraph_bulk_workflow_groq_with_ranking"
            }
        )
    finally:
        # Cleanup temporary directory and files
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup temp directory: {cleanup_error}")

# =================================================================
# CLI INTERFACE
# =================================================================

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced TalentFlow Pro - CV Processing System with Advanced Ranking")
    parser.add_argument("--server", "-s", action="store_true", help="Start FastAPI server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    if args.server:
        import uvicorn
        logger.info(f"Starting Enhanced TalentFlow Pro server with comprehensive ranking on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()