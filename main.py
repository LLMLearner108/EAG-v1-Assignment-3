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

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

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

@app.get("/scrape/{key}")
async def scrape_anime_page(key: str):
    if key not in anime_map:
        raise HTTPException(status_code=404, detail="Invalid key. Available keys: upcoming, favorite, airing")
    
    url = anime_map[key]
    try:
        response = requests.get(url)
        response.raise_for_status()
        return {"content": response.text}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

@app.post("/extract")
async def extract_anime_info(content: dict):
    if "content" not in content:
        raise HTTPException(status_code=400, detail="Content not provided")
    
    html_content = content["content"]
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract the table content
    table = soup.find('table', {'class': 'top-ranking-table'})
    if not table:
        raise HTTPException(status_code=404, detail="No anime table found in the content")
    
    # Convert table to text for Gemini
    table_text = table.get_text()
    
    # Create prompt for Gemini
    prompt = f"""Extract anime titles and their scores from the following table content. 
    Format the response as a JSON array of objects with 'title' and 'score' keys.
    Only include the top 10 entries.
    Return ONLY the JSON array, no markdown formatting or additional text.
    The score should be a number, not a string.
    
    Table content:
    {table_text}
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 