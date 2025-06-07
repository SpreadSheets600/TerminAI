import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime

try:
    import google.generativeai as genai
    import tiktoken
    from rich.align import Align
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install google-generativeai rich tiktoken")
    sys.exit(1)


@dataclass
class ChatMessage:
    role: str  # 'user' or 'model'
    content: str
    timestamp: datetime
    tokens: int | None = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tokens=data.get("tokens"),
        )


class GeminiTerminalChat:
    """Main chat application class"""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.console = Console()
        self.api_key = api_key
        self.model_name = model_name
        self.chat_history: list[ChatMessage] = []
        self.session_file = "chat_session.json"
        self.max_context_tokens = 30000  # Adjust based on model limits

        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.chat = None

        # Initialize tokenizer for context management
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split()) * 1.3  # Rough estimate

    def manage_context(self):
        """Manage chat context to stay within token limits"""
        if not self.chat_history:
            return

        total_tokens = sum(self.count_tokens(msg.content) for msg in self.chat_history)

        # If we're over the limit, remove oldest messages (keep system context)
        while total_tokens > self.max_context_tokens and len(self.chat_history) > 2:
            removed_msg = self.chat_history.pop(0)
            total_tokens -= self.count_tokens(removed_msg.content)
            self.console.print(
                f"[dim yellow]Context trimmed: Removed old message ({removed_msg.timestamp.strftime('%H:%M:%S')})[/]",
            )

    def save_session(self):
        """Save current chat session to file"""
        try:
            session_data = {
                "messages": [msg.to_dict() for msg in self.chat_history],
                "model": self.model_name,
                "saved_at": datetime.now().isoformat(),
            }
            with open(self.session_file, "w") as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            self.console.print(f"[red]Error saving session: {e}[/]")

    def load_session(self) -> bool:
        """Load previous chat session"""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file) as f:
                    session_data = json.load(f)

                self.chat_history = [
                    ChatMessage.from_dict(msg_data)
                    for msg_data in session_data["messages"]
                ]

                self.console.print(
                    f"[green]Loaded session with {len(self.chat_history)} messages[/]",
                )
                return True
        except Exception as e:
            self.console.print(f"[red]Error loading session: {e}[/]")
        return False

    def clear_session(self):
        """Clear current chat session"""
        self.chat_history.clear()
        self.chat = None
        if os.path.exists(self.session_file):
            os.remove(self.session_file)
        self.console.print("[green]Session cleared! Starting fresh.[/]")

    def display_welcome(self):
        """Display welcome message and instructions"""
        welcome_text = """
# 🤖 Gemini Terminal Chat

Welcome to your intelligent terminal companion!

## Commands:
- `/clear` - Clear chat history and start fresh
- `/save` - Save current session
- `/load` - Load previous session
- `/stats` - Show session statistics
- `/help` - Show this help message
- `/quit` or `Ctrl+C` - Exit application

## Features:
✨ **Context Aware**: Full conversation history maintained
🎨 **Beautiful UI**: Rich terminal interface with markdown support
💾 **Persistent**: Auto-saves your conversations
🧠 **Smart**: Automatic context management
🚀 **Fast**: Optimized for terminal use

Start chatting by typing your message!
        """

        panel = Panel(
            Markdown(welcome_text),
            title="🌟 Gemini Terminal Chat",
            border_style="bright_blue",
            padding=(1, 2),
        )
        self.console.print(panel)
        self.console.print()

    def display_message(self, message: ChatMessage):
        """Display a chat message with proper formatting"""
        timestamp = message.timestamp.strftime("%H:%M:%S")

        if message.role == "user":
            # User message
            panel = Panel(
                Text(message.content, style="white"),
                title=f"👤 You ({timestamp})",
                border_style="bright_cyan",
                width=None,
            )
        else:
            # AI message with markdown support
            panel = Panel(
                Markdown(message.content),
                title=f"🤖 Gemini ({timestamp})",
                border_style="bright_green",
                width=None,
            )

        self.console.print(panel)
        self.console.print()

    def display_stats(self):
        """Display session statistics"""
        if not self.chat_history:
            self.console.print("[yellow]No messages in current session[/]")
            return

        user_msgs = len([m for m in self.chat_history if m.role == "user"])
        ai_msgs = len([m for m in self.chat_history if m.role == "model"])
        total_tokens = sum(self.count_tokens(m.content) for m in self.chat_history)

        stats_table = Table(title="📊 Session Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")

        stats_table.add_row("User Messages", str(user_msgs))
        stats_table.add_row("AI Messages", str(ai_msgs))
        stats_table.add_row("Total Messages", str(len(self.chat_history)))
        stats_table.add_row("Estimated Tokens", f"{total_tokens:,}")
        stats_table.add_row("Model", self.model_name)

        if self.chat_history:
            first_msg = self.chat_history[0].timestamp.strftime("%Y-%m-%d %H:%M:%S")
            last_msg = self.chat_history[-1].timestamp.strftime("%Y-%m-%d %H:%M:%S")
            stats_table.add_row("Session Started", first_msg)
            stats_table.add_row("Last Message", last_msg)

        self.console.print(stats_table)
        self.console.print()

    async def send_message(self, user_input: str) -> str:
        """Send message to Gemini and get response"""
        try:
            # Create chat if it doesn't exist
            if self.chat is None:
                # Convert history to Gemini format
                history = []
                for msg in self.chat_history:
                    if msg.role in ["user", "model"]:
                        history.append(
                            {
                                "role": msg.role,
                                "parts": [msg.content],
                            },
                        )

                self.chat = self.model.start_chat(history=history)

            # Show thinking indicator
            with Live(
                Align.center(Spinner("dots", text="🤖 Gemini is thinking...")),
                refresh_per_second=4,
            ):
                response = await asyncio.to_thread(
                    self.chat.send_message,
                    user_input,
                )

            return response.text

        except Exception as e:
            return f"Error: {e!s}"

    def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        command = command.lower().strip()

        if command == "/clear":
            self.clear_session()
            return True

        if command == "/save":
            self.save_session()
            self.console.print("[green]Session saved successfully![/]")
            return True

        if command == "/load":
            if self.load_session():
                self.chat = None  # Reset chat to reload history
                self.display_chat_history()
            return True

        if command == "/stats":
            self.display_stats()
            return True

        if command == "/help":
            self.display_welcome()
            return True

        if command in ["/quit", "/exit"]:
            self.save_session()
            self.console.print("[green]Session saved. Goodbye! 👋[/]")
            sys.exit(0)

        return False

    def display_chat_history(self):
        """Display the current chat history"""
        if not self.chat_history:
            return

        self.console.print(Rule("[dim]Chat History[/]"))
        for message in self.chat_history:
            self.display_message(message)

    async def run(self):
        """Main application loop"""
        self.console.clear()
        self.display_welcome()

        # Ask if user wants to load previous session
        if os.path.exists(self.session_file):
            load_session = Prompt.ask(
                "Previous session found. Load it?",
                choices=["y", "n"],
                default="y",
            )
            if load_session.lower() == "y":
                if self.load_session():
                    self.display_chat_history()

        self.console.print(Rule())
        self.console.print()

        try:
            while True:
                # Get user input
                user_input = Prompt.ask(
                    "[bold blue]You[/bold blue]",
                    default="",
                ).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if self.handle_command(user_input):
                        continue
                    self.console.print(f"[red]Unknown command: {user_input}[/]")
                    continue

                # Add user message to history
                user_msg = ChatMessage(
                    role="user",
                    content=user_input,
                    timestamp=datetime.now(),
                    tokens=self.count_tokens(user_input),
                )
                self.chat_history.append(user_msg)
                self.display_message(user_msg)

                # Get AI response
                ai_response = await self.send_message(user_input)

                # Add AI message to history
                ai_msg = ChatMessage(
                    role="model",
                    content=ai_response,
                    timestamp=datetime.now(),
                    tokens=self.count_tokens(ai_response),
                )
                self.chat_history.append(ai_msg)
                self.display_message(ai_msg)

                # Manage context and auto-save
                self.manage_context()
                self.save_session()

        except KeyboardInterrupt:
            self.console.print("\n[green]Session saved. Goodbye! 👋[/]")
            self.save_session()
        except Exception as e:
            self.console.print(f"\n[red]Unexpected error: {e}[/]")
            self.save_session()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Gemini Terminal Chat")
    parser.add_argument(
        "--api-key",
        help="Gemini API key (or set GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--model",
        default="gemini-1.5-flash",
        help="Gemini model to use (default: gemini-1.5-flash)",
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        console = Console()
        console.print("[red]Error: No API key provided![/]")
        console.print("Set GEMINI_API_KEY environment variable or use --api-key")
        console.print(
            "\nGet your API key from: https://makersuite.google.com/app/apikey",
        )
        sys.exit(1)

    # Create and run chat application
    chat_app = GeminiTerminalChat(api_key=api_key, model_name=args.model)
    asyncio.run(chat_app.run())


if __name__ == "__main__":
    main()
