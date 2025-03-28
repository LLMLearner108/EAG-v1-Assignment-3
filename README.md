# Anime Data Scraper and Agent

A FastAPI-based service that scrapes anime data from MyAnimeList, processes it using Google's Gemini AI, and provides various functionalities through a REST API and Chrome extension.

## Features

- **Data Scraping**: Fetches anime data from MyAnimeList (upcoming, favorite, and airing categories)
- **AI-Powered Extraction**: Uses Google's Gemini AI to extract structured anime information
- **Filtering Capabilities**: Filter anime by score and get top-k entries
- **Email Integration**: Send filtered anime lists via email
- **Agent System**: An intelligent agent named "Hinata" that orchestrates complex tasks
- **Chrome Extension**: User-friendly interface for interacting with the service
- **Comprehensive Logging**: Detailed logging of all operations and endpoint access

## Prerequisites

- Python 3.8+
- Google Gemini API key
- SMTP server credentials
- Chrome browser (for the extension)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following:
```
GEMINI_API_KEY=your_gemini_api_key
MODEL_NAME=gemini-2.0-flash
```

4. Configure SMTP:
Create a `smtp.config` file with your email settings:
```json
{
    "smtp_server": "your_smtp_server",
    "smtp_port": 587,
    "sender_email": "your_email@example.com",
    "sender_password": "your_password",
    "default_recipient": "default@example.com"
}
```

## Running the Server

Start the FastAPI server:
```bash
python main.py
```

The server will run on `http://localhost:8000`

## API Endpoints

### 1. Scrape Anime Data
```
GET /scrape/{key}
```
- `key`: One of "upcoming", "favorite", or "airing"
- Returns raw HTML content from MyAnimeList

### 2. Extract Anime Information
```
POST /extract
```
- Processes HTML content using Gemini AI
- Returns structured anime data with titles and scores

### 3. Filter Anime
```
POST /filter
```
- Filters anime by minimum score
- Returns filtered list with statistics

### 4. Send Email
```
POST /send-email
```
- Sends filtered anime list via email
- Supports custom recipient and subject

### 5. Agent Endpoint
```
POST /agent
```
- Orchestrates complex tasks
- Supports three task types:
  - `top_k`: Get top K anime by score
  - `score_filter`: Filter anime by minimum score
  - `email`: Send filtered list via email

## Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the extension directory
4. The extension icon will appear in your toolbar

### Extension Features
- Select anime category (upcoming/favorite/airing)
- Choose task type (top-k/score filter/email)
- Set parameters (top-k value, minimum score)
- Enter email address for results
- View results directly in the popup

## Logging

The service maintains two log files:
- `hinata_detailed.log`: Detailed action logs
- `hinata_endpoints.log`: Endpoint access logs

## Error Handling

- Comprehensive error handling for all endpoints
- Detailed error messages in logs
- User-friendly error responses
- Rate limiting and pagination controls

## Security

- CORS middleware for cross-origin requests
- Environment variable protection for sensitive data
- Input validation using Pydantic models
- Secure email sending with TLS

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 