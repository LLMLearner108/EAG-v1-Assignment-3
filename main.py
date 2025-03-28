from fastapi import FastAPI, HTTPException
import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from urllib.parse import urljoin
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# Configure detailed logging
detailed_logger = logging.getLogger('HinataDetailed')
detailed_logger.setLevel(logging.INFO)
detailed_handler = logging.FileHandler('hinata_detailed.log')
detailed_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
detailed_logger.addHandler(detailed_handler)

# Configure endpoint logging
endpoint_logger = logging.getLogger('HinataEndpoints')
endpoint_logger.setLevel(logging.INFO)
endpoint_handler = logging.FileHandler('hinata_endpoints.log')
endpoint_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
endpoint_logger.addHandler(endpoint_handler)

# Load environment variables
load_dotenv()

# Load email configuration
with open('smtp.config', 'r') as f:
    email_config = json.load(f)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(os.getenv("MODEL_NAME"))

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load anime map
with open('anime_map.json', 'r') as f:
    anime_map = json.load(f)

# Pydantic models for request/response validation
class Anime(BaseModel):
    title: str
    score: float

class AnimeList(BaseModel):
    anime_list: List[Anime]

class FilterRequest(BaseModel):
    anime_data: AnimeList
    min_score: Optional[float] = None

class EmailRequest(BaseModel):
    recipient_email: Optional[str] = None
    subject: Optional[str] = "Your Filtered Anime List"

class AgentRequest(BaseModel):
    query: str
    task_type: str  # "top_k", "score_filter", or "email"
    top_k: Optional[int] = None
    min_score: Optional[float] = None
    recipient_email: Optional[str] = None

def log_endpoint(endpoint: str, method: str):
    """Log endpoint access"""
    endpoint_logger.info(f"{method} {endpoint}")

def log_detailed(action: str, details: str):
    """Log detailed actions"""
    detailed_logger.info(f"Hinata: {action} - {details}")

def send_email(recipient_email: str, subject: str, anime_list: List[Dict]) -> bool:
    """Send email using SMTP."""
    try:
        log_detailed("Sending email", f"To: {recipient_email}")
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Create email body
        body = f"""
        Here's your filtered anime list with {len(anime_list)} entries:

        {chr(10).join(f"{anime.title} - Score: {anime.score}" for anime in anime_list)}

        Best regards,
        Your Anime List Service
        """

        msg.attach(MIMEText(body, 'plain'))

        # Create SMTP session
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['sender_email'], email_config['sender_password'])

        # Send email
        server.send_message(msg)
        server.quit()

        log_detailed("Email sent successfully", f"To: {recipient_email}")
        return True
    except Exception as e:
        log_detailed("Email sending failed", str(e))
        print(f"Error sending email: {str(e)}")
        return False

def get_all_pages_content(base_url: str) -> str:
    """Get content from all pages of the anime list."""
    all_content = []
    current_url = base_url
    total_anime = 0
    
    while current_url and total_anime < 250:
        try:
            log_detailed("Fetching page", current_url)
            response = requests.get(current_url)
            response.raise_for_status()
            all_content.append(response.text)
            
            # Parse the HTML to find the next page link
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'top-ranking-table'})
            if table:
                rows = table.find_all('tr')[1:]  # Skip header row
                total_anime += len(rows)
            
            next_link = soup.find('link', {'rel': 'next'})
            
            if next_link and 'href' in next_link.attrs and total_anime < 250:
                current_url = urljoin(base_url, next_link['href'])
            else:
                current_url = None
                
        except requests.RequestException as e:
            log_detailed("Error fetching page", f"{current_url}: {str(e)}")
            break
            
    log_detailed("Pagination complete", f"Total anime fetched: {total_anime}")
    return "\n".join(all_content)

@app.get("/scrape/{key}")
async def scrape_anime_page(key: str):
    log_endpoint(f"/scrape/{key}", "GET")
    log_detailed("Scraping anime page", f"Key: {key}")
    
    if key not in anime_map:
        raise HTTPException(status_code=404, detail="Invalid key. Available keys: upcoming, favorite, airing")
    
    url = anime_map[key]
    try:
        content = get_all_pages_content(url)
        return {"content": content}
    except Exception as e:
        log_detailed("Scraping failed", str(e))
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

@app.post("/extract")
async def extract_anime_info(content: dict):
    log_endpoint("/extract", "POST")
    log_detailed("Extracting anime info", "Starting extraction")
    
    if "content" not in content:
        raise HTTPException(status_code=400, detail="Content not provided")
    
    html_content = content["content"]
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract all table contents
    tables = soup.find_all('table', {'class': 'top-ranking-table'})
    if not tables:
        raise HTTPException(status_code=404, detail="No anime tables found in the content")
    
    # Combine all table texts
    all_table_text = "\n".join(table.get_text() for table in tables)
    
    # Create prompt for Gemini with more specific instructions
    prompt = f"""Extract anime titles and their scores from the following table content.
    Return a JSON array of objects with exactly this structure:
    [
        {{"title": "Anime Title", "score": 8.5}},
        {{"title": "Another Anime", "score": 7.8}}
    ]
    
    Rules:
    1. Each object must have exactly "title" and "score" keys
    2. Title must be a string
    3. Score must be a number (float)
    4. Return ONLY the JSON array, no other text
    5. Do not include any markdown formatting
    6. Do not include any explanations or additional text
    
    Table content:
    {all_table_text}
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean the response text
        response_text = response.text.strip()
        
        # Remove any markdown code block markers
        response_text = re.sub(r'```json\s*|\s*```', '', response_text)
        
        # Remove any leading/trailing whitespace and newlines
        response_text = response_text.strip()
        
        # Try to find JSON array in the response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        # Parse the cleaned response as JSON
        anime_data = json.loads(response_text)
        
        # Validate the structure
        for anime in anime_data:
            if not isinstance(anime.get("title"), str) or not isinstance(anime.get("score"), (int, float)):
                raise ValueError("Invalid data structure in response")
        
        log_detailed("Extraction complete", f"Found {len(anime_data)} anime entries")
        return {"anime_list": anime_data}
    except Exception as e:
        log_detailed("Extraction failed", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing with Gemini: {str(e)}")

@app.post("/filter")
async def filter_anime_by_score(request: FilterRequest):
    log_detailed("Filtering anime", f"Min score: {request.min_score}")
    try:
        if request.min_score is None:
            return {
                "filtered_anime_list": request.anime_data.anime_list,
                "min_score": None,
                "total_anime": len(request.anime_data.anime_list),
                "filtered_count": len(request.anime_data.anime_list)
            }
            
        # Filter anime with scores above the threshold
        filtered_anime = [
            anime for anime in request.anime_data.anime_list
            if anime.score >= request.min_score
        ]
        
        log_detailed("Filtering complete", f"Found {len(filtered_anime)} anime above score {request.min_score}")
        return {
            "filtered_anime_list": filtered_anime,
            "min_score": request.min_score,
            "total_anime": len(request.anime_data.anime_list),
            "filtered_count": len(filtered_anime)
        }
    except ValueError as e:
        log_detailed("Filtering failed", str(e))
        raise HTTPException(status_code=400, detail="Invalid score format in anime data")
    except Exception as e:
        log_detailed("Filtering failed", str(e))
        raise HTTPException(status_code=500, detail=f"Error filtering anime: {str(e)}")

@app.post("/send-email")
async def send_anime_email(filtered_data: dict, email_request: EmailRequest):
    log_detailed("Sending email", f"To: {email_request.recipient_email}")
    try:
        # Get the filtered anime list
        anime_list = filtered_data.get("filtered_anime_list", [])
        if not anime_list:
            raise HTTPException(status_code=400, detail="No anime data to send")

        # Get recipient email
        recipient_email = email_request.recipient_email or email_config["default_recipient"]

        # Send email
        success = send_email(
            recipient_email=recipient_email,
            subject=email_request.subject,
            anime_list=anime_list
        )

        if success:
            log_detailed("Email sent successfully", f"To: {recipient_email}")
            return {
                "message": "Email sent successfully",
                "recipient": recipient_email,
                "anime_count": len(anime_list)
            }
        else:
            log_detailed("Email sending failed", "Check server logs for details")
            raise HTTPException(
                status_code=500,
                detail="Failed to send email. Please check the server logs for details."
            )

    except Exception as e:
        log_detailed("Email sending failed", str(e))
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")

@app.post("/agent")
async def hinata_agent(request: AgentRequest):
    log_detailed("Agent started", f"Task type: {request.task_type}, Query: {request.query}")
    
    try:
        # Step 1: Scrape the content
        scrape_response = await scrape_anime_page(request.query)
        log_detailed("Scraping complete", "Content fetched successfully")
        
        # Step 2: Extract anime info
        extract_response = await extract_anime_info(scrape_response)
        anime_list = extract_response["anime_list"]
        log_detailed("Extraction complete", f"Found {len(anime_list)} anime entries")
        
        # Step 3: Process based on task type
        if request.task_type == "top_k":
            if not request.top_k:
                raise HTTPException(status_code=400, detail="top_k parameter required for this task")
            # Sort by score and get top k
            sorted_anime = sorted(anime_list, key=lambda x: x["score"], reverse=True)
            result = sorted_anime[:request.top_k]
            log_detailed("Top K processing complete", f"Returned top {request.top_k} anime")
            
        elif request.task_type == "score_filter":
            if not request.min_score:
                raise HTTPException(status_code=400, detail="min_score parameter required for this task")
            # Filter by score
            filter_request = FilterRequest(
                anime_data=AnimeList(anime_list=anime_list),
                min_score=request.min_score
            )
            filter_response = await filter_anime_by_score(filter_request)
            result = filter_response["filtered_anime_list"]
            log_detailed("Score filtering complete", f"Found {len(result)} anime above score {request.min_score}")
            
        elif request.task_type == "email":
            if not request.min_score or not request.recipient_email:
                raise HTTPException(status_code=400, detail="min_score and recipient_email required for this task")
            # Filter by score
            filter_request = FilterRequest(
                anime_data=AnimeList(anime_list=anime_list),
                min_score=request.min_score
            )
            filter_response = await filter_anime_by_score(filter_request)
            filtered_anime = filter_response["filtered_anime_list"]
            
            # Send email
            email_request = EmailRequest(
                recipient_email=request.recipient_email,
                subject=f"Anime List (Score >= {request.min_score})"
            )
            await send_anime_email({"filtered_anime_list": filtered_anime}, email_request)
            result = filtered_anime
            log_detailed("Email task complete", f"Sent {len(result)} anime to {request.recipient_email}")
            
        else:
            log_detailed("Invalid task type", request.task_type)
            raise HTTPException(status_code=400, detail="Invalid task type. Must be one of: top_k, score_filter, email")
        
        log_detailed("Agent task complete", f"Successfully processed {request.task_type} task")
        return {
            "status": "success",
            "task_type": request.task_type,
            "result": result
        }
        
    except Exception as e:
        log_detailed("Agent task failed", str(e))
        raise HTTPException(status_code=500, detail=f"Agent task failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 