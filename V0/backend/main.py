from src.graph import Bot
import uuid

from src.utils import _print_event

if __name__ == "__main__":
    bot = Bot()
    graph = bot.kickstart()

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "user_name": "Islam",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    """TODO: Improve the printing of result"""

    while True:
        _printed = set()
        user = input("User (q/Q to quit): ")
        if user in {"q", "Q"}:
            print("Samantha: Byebye")
            break
        for output in graph.stream({"messages": ("user", user)}, config, stream_mode="values"):
            _print_event(output, _printed)