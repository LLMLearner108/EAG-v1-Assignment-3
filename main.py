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

def send_email(recipient_email: str, subject: str, anime_list: List[Dict]) -> bool:
    """Send email using SMTP."""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Create email body
        body = f"""
        Here's your filtered anime list with {len(anime_list)} entries:

        {chr(10).join(f"{anime['title']} - Score: {anime['score']}" for anime in anime_list)}

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

        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def get_all_pages_content(base_url: str) -> str:
    """Get content from all pages of the anime list."""
    all_content = []
    current_url = base_url
    
    while current_url:
        try:
            response = requests.get(current_url)
            response.raise_for_status()
            all_content.append(response.text)
            
            # Parse the HTML to find the next page link
            soup = BeautifulSoup(response.text, 'html.parser')
            next_link = soup.find('link', {'rel': 'next'})
            
            if next_link and 'href' in next_link.attrs:
                current_url = urljoin(base_url, next_link['href'])
            else:
                current_url = None
                
        except requests.RequestException as e:
            print(f"Error fetching page {current_url}: {str(e)}")
            break
            
    return "\n".join(all_content)

@app.get("/scrape/{key}")
async def scrape_anime_page(key: str):
    if key not in anime_map:
        raise HTTPException(status_code=404, detail="Invalid key. Available keys: upcoming, favorite, airing")
    
    url = anime_map[key]
    try:
        # Get content from all pages
        content = get_all_pages_content(url)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

@app.post("/extract")
async def extract_anime_info(content: dict):
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
    
    # Create prompt for Gemini
    prompt = f"""Extract anime titles and their scores from the following table content. 
    Format the response as a JSON array of objects with 'title' and 'score' keys.
    Include all entries from all tables.
    Return ONLY the JSON array, no markdown formatting or additional text.
    The score should be a number, not a string.
    
    Table content:
    {all_table_text}
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean the response text
        response_text = response.text.strip()
        # Remove markdown code block markers if present
        response_text = re.sub(r'```json\s*|\s*```', '', response_text)
        # Remove any leading/trailing whitespace and newlines
        response_text = response_text.strip()
        
        # Parse the cleaned response as JSON
        anime_data = json.loads(response_text)
        return {"anime_list": anime_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with Gemini: {str(e)}")

@app.post("/filter")
async def filter_anime_by_score(request: FilterRequest):
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
        
        return {
            "filtered_anime_list": filtered_anime,
            "min_score": request.min_score,
            "total_anime": len(request.anime_data.anime_list),
            "filtered_count": len(filtered_anime)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid score format in anime data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error filtering anime: {str(e)}")

@app.post("/send-email")
async def send_anime_email(filtered_data: dict, email_request: EmailRequest):
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
            anime_list=anime_list  # Pass the list directly without JSON conversion
        )

        if success:
            return {
                "message": "Email sent successfully",
                "recipient": recipient_email,
                "anime_count": len(anime_list)
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to send email. Please check the server logs for details."
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 