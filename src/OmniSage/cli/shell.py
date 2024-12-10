from typing import Optional
import os
import requests
from datetime import datetime
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import WordCompleter
from src.omnisage.core.client import LlamaChatClient, Message

class CommandCompleter(WordCompleter):
    """Custom command completer that only activates for commands."""
    def get_completions(self, document, complete_event):
        if document.text.startswith('/'):
            text_before_cursor = document.text_before_cursor
            yield from super().get_completions(document, complete_event)

class OmniShell:
    def __init__(self, base_url: str = "http://localhost:8000", debug: bool = False):
        self.base_url = base_url
        self.debug = debug
        self.current_chat_id: Optional[int] = None
        self.current_chat_title: Optional[str] = None
        self.conversation_history: list[Message] = []
        
        # Setup prompt styling
        self.style = Style.from_dict({
            'prompt': '#00aa00 bold',
            'command': '#ffa500',      # Orange for commands
            'chat-id': '#666666',      # Grey
            'error': '#ff0000 bold',   # Bold red for errors
            'warning': '#ffaa00',      # Yellow for warnings
            'info': '#0000ff',         # Blue for info
            'title': '#00aa00 bold',   # Bold green for titles
        })
        
        # Setup command completion
        self.completer = CommandCompleter(
            ['/new', '/load', '/delete', '/clear', '/status', '/help', '/quit']
        )
        
        # Initialize prompt session
        self.session = PromptSession(
            completer=self.completer,
            style=self.style,
            complete_while_typing=True
        )
        
        # Setup key bindings
        self.kb = KeyBindings()
        self._setup_key_bindings()
        
        # Initialize client with connection check
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the API client and verify connection."""
        self.client = LlamaChatClient(base_url=self.base_url)
        self._check_server_connection()

    def _check_server_connection(self) -> bool:
        """Check if the API server is running."""
        try:
            response = requests.get(f"{self.base_url}/docs")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _print_styled(self, message: str, style: str = 'default'):
        """Print styled message."""
        if style == 'error':
            print_formatted_text(FormattedText([('#ff0000 bold', message)]))
        elif style == 'warning':
            print_formatted_text(FormattedText([('#ffaa00', message)]))
        elif style == 'info':
            print_formatted_text(FormattedText([('#0000ff', message)]))
        elif style == 'command':
            print_formatted_text(FormattedText([('#ffa500', message)]))
        else:
            print(message)

    def _setup_key_bindings(self):
        """Setup custom key bindings."""
        @self.kb.add('c-c')
        def _(event):
            """Handle Ctrl+C."""
            event.app.exit()

    def _get_prompt_message(self) -> HTML:
        """Get formatted prompt message based on current state."""
        if self.current_chat_id:
            return HTML(
                '<prompt>omni</prompt>'
                f'<chat-id>({self.current_chat_id})</chat-id>> '
            )
        return HTML('<prompt>omni</prompt>> ')

    def _format_message(self, role: str, content: str) -> FormattedText:
        """Format a message with proper styling."""
        if role == "user":
            # Grey for user
            return FormattedText([
                ('#808080', 'user'),
                ('', '> '),
                ('', content)
            ])
        else:
            # Green for omni
            return FormattedText([
                ('#00aa00', 'omni'),
                ('', '> '),
                ('', content)
            ])
    
    def _print_formatted(self, formatted_text: FormattedText):
        """Print formatted text with proper styling."""
        print_formatted_text(formatted_text)

    def _format_chat_list(self, chats: list[dict]) -> str:
        """Format chat list for display."""
        if not chats:
            return "No saved chats found."
            
        result = "\nSaved Chats:\n"
        for chat in chats:
            latest = chat["latest_message"] or ""
            if latest and len(latest) > 50:
                latest = latest[:47] + "..."
                
            updated = datetime.fromisoformat(chat["updated_at"])
            result += (
                f"{chat['id']}: {chat['title']} "
                f"(Last updated: {updated.strftime('%Y-%m-%d %H:%M')})\n"
                f"    Latest: {latest}\n"
            )
        
        return result

    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_help(self):
        """Display help information."""
        help_text = """
        Available Commands:
        /new <title>  - Create new chat
        /load         - List and load saved chats
        /delete       - Delete a chat
        /clear        - Clear current conversation
        /status       - Check server connection status
        /help         - Show this help message
        /quit         - Exit OmniShell
        """
        print(help_text)

    def _get_prompt_message(self) -> FormattedText:
        """Get formatted prompt message."""
        return FormattedText([
            ('#808080', 'user'),  # Grey for user
            ('', '> ')
        ])

    def _display_chat_header(self):
        """Display chat title and separator."""
        if self.current_chat_title:
            print("\n" + "=" * 50)
            self._print_formatted(FormattedText([
                ('#00aa00 bold', f"Active Chat: {self.current_chat_title}")
            ]))
            print("=" * 50 + "\n")

    def _display_chat_history(self):
        """Display the entire chat history with proper formatting."""
        if not self.conversation_history:
            return

        self._clear_screen()
        self._display_chat_header()
        
        for msg in self.conversation_history:
            formatted_msg = self._format_message(msg.role, msg.content)
            self._print_formatted(formatted_msg)
            print()  # Add spacing between messages

    async def _handle_command(self, command: str) -> bool:
        """Handle command input. Returns False if should exit."""
        cmd_parts = command[1:].split(maxsplit=1)
        cmd = cmd_parts[0].lower()
        
        if cmd in ["quit", "exit"]:
            print("\nGoodbye!")
            return False
            
        elif cmd == "status":
            if self._check_server_connection():
                self._print_styled("Server is running and accessible", 'info')
            else:
                self._print_styled(
                    f"Cannot connect to server at {self.base_url}\n"
                    "Make sure the API server is running with:\n"
                    "python -m src.OmniSage.api.app", 
                    'error'
                )
            
        elif cmd == "new":
            if not self._check_server_connection():
                self._print_styled("Cannot create chat: Server is not accessible", 'error')
                return True
                
            if len(cmd_parts) < 2:
                self._print_styled("Please provide a title for the new chat", 'warning')
                return True
                
            title = cmd_parts[1]
            try:
                chat = self.client.create_chat(title)
                self.current_chat_id = chat["id"]
                self.current_chat_title = title
                self.conversation_history = []
                self._clear_screen()
                self._display_chat_header()
            except Exception as e:
                self._print_styled(f"Failed to create chat: {str(e)}", 'error')
            
        elif cmd == "load":
            if not self._check_server_connection():
                self._print_styled("Cannot load chats: Server is not accessible", 'error')
                return True
                
            try:
                chats = self.client.list_chats()
                print(self._format_chat_list(chats))
                
                chat_choice = input("Enter chat ID to load (or press Enter to cancel): ").strip()
                
                if chat_choice:
                    try:
                        chat_id = int(chat_choice)
                        messages = self.client.get_chat_messages(chat_id)
                        self.conversation_history = [
                            Message(role=msg["role"], content=msg["content"])
                            for msg in messages
                        ]
                        self.current_chat_id = chat_id
                        # Find chat title from the list
                        chat_info = next((c for c in chats if c["id"] == chat_id), None)
                        if chat_info:
                            self.current_chat_title = chat_info["title"]
                        self._display_chat_history()
                    except ValueError:
                        self._print_styled("Invalid chat ID", 'error')
            except Exception as e:
                self._print_styled(f"Failed to load chats: {str(e)}", 'error')
                    
        elif cmd == "delete":
            if not self._check_server_connection():
                self._print_styled("Cannot delete chat: Server is not accessible", 'error')
                return True
                
            try:
                chats = self.client.list_chats()
                print(self._format_chat_list(chats))
                
                # Use regular input instead of prompt_toolkit for simple inputs
                chat_choice = input("Enter chat ID to delete (or press Enter to cancel): ").strip()
                
                if chat_choice:
                    try:
                        chat_id = int(chat_choice)
                        self.client.delete_chat(chat_id)
                        if chat_id == self.current_chat_id:
                            self.current_chat_id = None
                            self.conversation_history = []
                        self._print_styled(f"\nDeleted chat {chat_id}", 'info')
                    except ValueError:
                        self._print_styled("Invalid chat ID", 'error')
            except Exception as e:
                self._print_styled(f"Failed to delete chat: {str(e)}", 'error')
                    
        elif cmd == "clear":
            self.current_chat_id = None
            self.conversation_history = []
            self._clear_screen()
            self._print_styled("Conversation history cleared.", 'info')
            
        elif cmd == "help":
            self._print_help()
            
        else:
            self._print_styled(f"Unknown command: {cmd}", 'error')
            
        return True

    async def _process_chat(self, user_input: str):
        """Process chat input and display response."""
        if not self._check_server_connection():
            self._print_styled("Cannot send message: Server is not accessible", 'error')
            return
            
        try:
            self.conversation_history.append(
                Message(role="user", content=user_input)
            )
            
            # Print omni prefix with proper formatting
            print_formatted_text(
                FormattedText([
                    ('#00aa00', 'omni'),  # Green for omni
                    ('', '> ')
                ]),
                end='',
                flush=True
            )
            
            full_response = []
            
            for chunk in self.client.chat_stream(
                self.conversation_history,
                chat_id=self.current_chat_id,
                max_tokens=None,  # Let server use model's max_output_length
                temperature=0.7
            ):
                if isinstance(chunk, dict):
                    if self.debug:
                        self._print_styled(
                            f"\nUsing model group: {chunk['model_group']}", 
                            'info'
                        )
                else:
                    print(chunk, end="", flush=True)
                    full_response.append(chunk)
            
            # Add assistant's response to history
            self.conversation_history.append(
                Message(
                    role="assistant",
                    content="".join(full_response).strip()
                )
            )
            print()
            
        except Exception as e:
            self._print_styled(f"\nError: {str(e)}", 'error')

    async def start(self):
        """Start the OmniShell terminal interface."""
        self._clear_screen()
        print("Welcome to OmniShell!\n")
        self._print_help()
        print()
        
        while True:
            try:
                # Get user input with prompt
                user_input = await self.session.prompt_async(
                    self._get_prompt_message(),
                    key_bindings=self.kb
                )
                
                if not user_input.strip():
                    continue
                    
                # Handle commands
                if user_input.startswith('/'):
                    should_continue = await self._handle_command(user_input)
                    if not should_continue:
                        break
                        
                # Ensure we have an active chat
                elif not self.current_chat_id:
                    self._print_styled(
                        "\nNo active chat. Please create a new chat first with /new <title>",
                        'warning'
                    )
                    
                # Process chat input
                else:
                    await self._process_chat(user_input)
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                self._print_styled(f"\nError: {str(e)}", 'error')
                if self.debug:
                    import traceback
                    traceback.print_exc()

def main():
    """Entry point for the OmniShell."""
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Start the OmniShell terminal interface")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:8000",
        help="API server host URL"
    )
    
    args = parser.parse_args()
    
    try:
        shell = OmniShell(base_url=args.host, debug=args.debug)
        asyncio.run(shell.start())
    except Exception as e:
        print(f"\033[91mError: {str(e)}\033[0m")

if __name__ == "__main__":
    main()