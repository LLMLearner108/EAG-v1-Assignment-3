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

# Add these functions at the top level, after the imports
async def function_caller(func_name: str, params: dict, last_response: dict = None) -> dict:
    """Execute API endpoint based on function name and parameters"""
    function_map = {
        "scrape_anime_page": scrape_anime_page,
        "extract_anime_info": extract_anime_info,
        "filter_anime_by_score": filter_anime_by_score,
        "send_anime_email": send_anime_email
    }
    
    if func_name in function_map:
        try:
            # Special handling for extract_anime_info
            if func_name == "extract_anime_info" and last_response and "content" in last_response:
                params = {"content": last_response}
            
            # Convert params to appropriate Pydantic models if needed
            if func_name == "filter_anime_by_score":
                # Ensure anime_list exists and has the correct structure
                if "request" not in params or "anime_data" not in params["request"]:
                    raise ValueError("Missing required fields in filter_anime_by_score parameters")
                
                anime_data = params["request"]["anime_data"]
                if "anime_list" not in anime_data:
                    raise ValueError("Missing anime_list in anime_data")
                
                # Convert each anime entry to Anime model
                anime_list = [Anime(**anime) for anime in anime_data["anime_list"]]
                
                params = {
                    "request": FilterRequest(
                        anime_data=AnimeList(anime_list=anime_list),
                        min_score=params["request"].get("min_score")
                    )
                }
            elif func_name == "send_anime_email":
                if "filtered_data" not in params or "email_request" not in params:
                    raise ValueError("Missing required fields in send_anime_email parameters")
                
                # Ensure filtered_data has the correct structure
                if "filtered_anime_list" not in params["filtered_data"]:
                    raise ValueError("Missing filtered_anime_list in filtered_data")
                
                # Convert each anime entry to Anime model
                anime_list = [Anime(**anime) for anime in params["filtered_data"]["filtered_anime_list"]]
                
                params = {
                    "filtered_data": {"filtered_anime_list": anime_list},
                    "email_request": EmailRequest(**params["email_request"])
                }
            
            return await function_map[func_name](**params)
        except Exception as e:
            raise ValueError(f"Error preparing parameters for {func_name}: {str(e)}")
    else:
        raise ValueError(f"Function {func_name} not found")

@app.post("/agent")
async def hinata_agent(request: AgentRequest):
    log_detailed("Agent started", f"Task type: {request.task_type}, Query: {request.query}")
    
    # System prompt that defines the available functions and their purposes with exact schema requirements
    system_prompt = """You are an agent that can perform tasks by calling available API endpoints.
    Respond with EXACTLY ONE of these formats:
    1. FUNCTION_CALL: function_name|params
    2. FINAL_ANSWER: result

    IMPORTANT JSON FORMATTING RULES:
    1. Use double quotes for all strings, not single quotes
    2. Use commas to separate array elements and object properties
    3. Do not include trailing commas
    4. Use proper JSON number format (no leading zeros)
    5. Escape special characters in strings
    6. Do not include any comments or explanations in the JSON

    Available functions with their exact parameter schemas:

    1. scrape_anime_page(key: str)
       Example: FUNCTION_CALL: scrape_anime_page|{"key": "upcoming"}

    2. extract_anime_info(content: dict)
       IMPORTANT: For this function, you can simply return FUNCTION_CALL: extract_anime_info|{}
       The system will automatically use the content from the previous scrape_anime_page call.
       No need to provide any parameters.

    3. filter_anime_by_score(request: dict)
       Schema:
       {
         "request": {
           "anime_data": {
             "anime_list": [
               {"title": "string", "score": number}
             ]
           },
           "min_score": number
         }
       }
       Example: FUNCTION_CALL: filter_anime_by_score|{"request": {"anime_data": {"anime_list": [{"title": "Anime 1", "score": 8.5}]}, "min_score": 8.0}}

    4. send_anime_email(filtered_data: dict, email_request: dict)
       Schema:
       {
         "filtered_data": {
           "filtered_anime_list": [
             {"title": "string", "score": number}
           ]
         },
         "email_request": {
           "recipient_email": "string",
           "subject": "string"
         }
       }
       Example: FUNCTION_CALL: send_anime_email|{"filtered_data": {"filtered_anime_list": [{"title": "Anime 1", "score": 8.5}]}, "email_request": {"recipient_email": "user@example.com", "subject": "Your Anime List"}}

    ANIME DATA EXTRACTION RULES:
    1. Each anime entry must have exactly two fields: "title" and "score"
    2. Title must be a string with double quotes
    3. Score must be a number (float) without quotes
    4. Example of valid anime entry: {"title": "Attack on Titan", "score": 8.5}
    5. Example of invalid anime entry: {'title': 'Attack on Titan', 'score': '8.5'}

    Rules:
    1. Each function call must follow the exact schema shown above
    2. All required fields must be present
    3. Data types must match exactly (string, number)
    4. Return ONLY the function call or final answer, no other text
    5. Do not include any markdown formatting
    6. Do not include any explanations or additional text
    7. Ensure all JSON is properly formatted and parsable
    8. For extract_anime_info, just return FUNCTION_CALL: extract_anime_info|{}
    
    After each function call, you'll receive the result and can decide the next step."""

    max_iterations = 5
    iteration = 0
    last_response = None
    iteration_response = []

    try:
        while iteration < max_iterations:
            log_detailed(f"Agent iteration {iteration + 1}", "Processing")
            
            # Construct the current query
            if iteration == 0:
                current_query = f"Task: {request.task_type}\nQuery: {request.query}\nParameters: top_k={request.top_k}, min_score={request.min_score}, recipient_email={request.recipient_email}"
            else:
                # Add more context about the last response
                last_step = iteration_response[-1] if iteration_response else ""
                current_query = f"""Task: {request.task_type}
Query: {request.query}
Parameters: top_k={request.top_k}, min_score={request.min_score}, recipient_email={request.recipient_email}

Previous steps:
{chr(10).join(iteration_response)}

Based on the last response, you should:
1. For top_k task: Call filter_anime_by_score with the anime_list from the last response
2. For score_filter task: Call filter_anime_by_score with the anime_list from the last response
3. For email task: Call filter_anime_by_score first, then send_anime_email with the filtered results

What should be the next step?"""

            # Get model's response
            prompt = f"{system_prompt}\n\nQuery: {current_query}"
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            log_detailed("LLM Response", response_text)

            # Check if it's a function call
            if response_text.startswith("FUNCTION_CALL:"):
                try:
                    _, function_info = response_text.split(":", 1)
                    func_name, params = [x.strip() for x in function_info.split("|", 1)]
                    
                    # Clean the params string before parsing
                    params = params.strip()
                    if params.startswith("'") and params.endswith("'"):
                        params = params[1:-1]  # Remove single quotes if present
                    
                    # Special handling for extract_anime_info
                    if func_name == "extract_anime_info":
                        params = "{}"  # Force empty object for extract_anime_info
                    else:
                        params = json.loads(params)
                    
                    # Execute the function
                    result = await function_caller(func_name, params, last_response)
                    
                    # Record the iteration
                    iteration_response.append(f"In iteration {iteration + 1}, called {func_name} with params {params} and got result {result}")
                    
                    # Check if we've completed the task
                    if request.task_type == "top_k" and func_name == "filter_anime_by_score":
                        if len(result["filtered_anime_list"]) <= (request.top_k or 0):
                            return {
                                "status": "success",
                                "task_type": request.task_type,
                                "result": result["filtered_anime_list"]
                            }
                    elif request.task_type == "score_filter" and func_name == "filter_anime_by_score":
                        return {
                            "status": "success",
                            "task_type": request.task_type,
                            "result": result["filtered_anime_list"]
                        }
                    elif request.task_type == "email" and func_name == "send_anime_email":
                        return {
                            "status": "success",
                            "task_type": request.task_type,
                            "result": result
                        }
                    
                    last_response = result
                    iteration += 1
                    
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in function call: {str(e)}\nResponse text: {response_text}")
                except ValueError as e:
                    raise ValueError(f"Invalid parameters: {str(e)}")
                
            # Check if it's the final answer
            elif response_text.startswith("FINAL_ANSWER:"):
                try:
                    _, result = response_text.split(":", 1)
                    return {
                        "status": "success",
                        "task_type": request.task_type,
                        "result": json.loads(result)
                    }
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in final answer: {str(e)}")
            
            else:
                # If the response is empty or just whitespace, and we're calling extract_anime_info
                if not response_text and last_response and "content" in last_response:
                    func_name = "extract_anime_info"
                    params = "{}"
                    result = await function_caller(func_name, params, last_response)
                    iteration_response.append(f"In iteration {iteration + 1}, called {func_name} with empty params and got result {result}")
                    last_response = result
                    iteration += 1
                    continue
                
                # If we have anime_list in the last response, suggest the next step
                if last_response and "anime_list" in last_response:
                    if request.task_type == "top_k":
                        return {
                            "status": "success",
                            "task_type": request.task_type,
                            "result": last_response["anime_list"][:request.top_k]
                        }
                    elif request.task_type == "score_filter":
                        return {
                            "status": "success",
                            "task_type": request.task_type,
                            "result": last_response["anime_list"]
                        }
                
                raise ValueError("Invalid response format from LLM")

        # If we've reached max iterations without completing the task
        raise HTTPException(status_code=500, detail="Task could not be completed within maximum iterations")

    except Exception as e:
        log_detailed("Agent task failed", str(e))
        raise HTTPException(status_code=500, detail=f"Agent task failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 