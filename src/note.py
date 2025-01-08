from langchain_core.tools import tool
from pydantic import BaseModel

# Định nghĩa schema cho tham số đầu vào
class NoteInput(BaseModel):
    note: str  # Định nghĩa kiểu dữ liệu rõ ràng cho tham số

@tool
def note_tool(note: str) -> str:
    """
    Saves a note to a local file

    Args:
        note: the text note to
    """
    
    with open("notes.txt", "a") as file:
        file.write(note+"\n")

    return "Note saved: notes.txt\n"

    