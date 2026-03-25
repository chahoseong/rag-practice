import os
import uvicorn
from dotenv import load_dotenv

# Load .env into os.environ at startup
load_dotenv()

def main():
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"

    # Note: EXPOSE_RETRIEVE_ENDPOINT=true will be read by app/app.py
    # to include the /api/retrieve router.

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    main()
