from dotenv import load_dotenv
load_dotenv()

from ui import build_ui

if __name__ == "__main__":
    build_ui().launch(share=True)
