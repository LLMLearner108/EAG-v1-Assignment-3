from fastapi import FastAPI, HTTPException
import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

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
    
    Table content:
    {table_text}
    """
    
    try:
        response = model.generate_content(prompt)
        # Parse the response as JSON
        anime_data = json.loads(response.text)
        return {"anime_list": anime_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with Gemini: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 