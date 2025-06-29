import socket
import threading
import time
import json
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

class MothCommand(Enum):
    GENERATE_MOTH = "generatemoth"
    BIRTH_MOTH = "birthmoth"
    MATING_MOTH = "matingmoth"
    KILL_MOTH = "killmoth"

@dataclass
class UDPConfig:
    host: str = "127.0.0.1"
    send_port: int = 8889
    receive_port: int = 8888
    buffer_size: int = 1024
    timeout: float = 5.0

class UDPBridgeClient:
    def __init__(self, config: UDPConfig = None):
        self.config = config or UDPConfig()
        self.socket = None
        self.is_running = False
        self.last_received_message = ""
        self.last_sent_message = ""
        self.on_message_received: Optional[Callable[[str], None]] = None
        self.receive_thread = None
        
    def start(self):
        """Initialize and start the UDP client"""
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.config.timeout)
            
            # Bind to receive port for listening
            self.socket.bind(('', self.config.receive_port))
            
            self.is_running = True
            print(f"UDP Bridge Client started on {self.config.host}:{self.config.receive_port}")
            
            # Start receiving thread
            self.receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
            self.receive_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting UDP client: {e}")
            return False
    
    def stop(self):
        """Stop the UDP client"""
        self.is_running = False
        if self.socket:
            self.socket.close()
        print("UDP Bridge Client stopped")
    
    def _receive_messages(self):
        """Background thread for receiving messages"""
        while self.is_running:
            try:
                data, addr = self.socket.recvfrom(self.config.buffer_size)
                message = data.decode('utf-8')
                self.last_received_message = message
                print(f"Received from {addr}: {message}")
                
                if self.on_message_received:
                    self.on_message_received(message)
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"Error receiving message: {e}")
    
    def send_message(self, message: str) -> bool:
        """Send a message via UDP"""
        try:
            if not self.socket:
                print("Socket not initialized")
                return False
                
            encoded_message = message.encode('utf-8')
            self.socket.sendto(encoded_message, (self.config.host, self.config.send_port))
            self.last_sent_message = message
            print(f"Sent: {message}")
            return True
            
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    def send_moth_command(self, command: MothCommand, *args) -> bool:
        """Send a moth-related command"""
        message_parts = [command.value] + list(args)
        message = "|".join(message_parts)
        return self.send_message(message)
    
    def generate_new_moth(self, moth_id: str, texture_path: str, prompt: str = "") -> bool:
        """Generate a new moth"""
        return self.send_moth_command(MothCommand.GENERATE_MOTH, moth_id, texture_path, prompt)
    
    def birth_moth(self, moth_id: str, texture_path: str, prompt: str = "") -> bool:
        """Birth a moth"""
        return self.send_moth_command(MothCommand.BIRTH_MOTH, moth_id, texture_path, prompt)
    
    def mating_moth(self, moth_id1: str, moth_id2: str) -> bool:
        """Mate two moths"""
        return self.send_moth_command(MothCommand.MATING_MOTH, moth_id1, moth_id2)
    
    def kill_moth(self, moth_id: str) -> bool:
        """Kill a moth"""
        return self.send_moth_command(MothCommand.KILL_MOTH, moth_id)
    
    def send_json_message(self, data: dict) -> bool:
        """Send a JSON formatted message"""
        try:
            message = json.dumps(data)
            return self.send_message(message)
        except Exception as e:
            print(f"Error sending JSON message: {e}")
            return False

def message_handler(message: str):
    """Example message handler function"""
    print(f"Processing received message: {message}")
    
    # Parse response messages
    if message.startswith("Generated moth:"):
        print("✓ Moth generation confirmed")
    elif message.startswith("Birth moth:"):
        print("✓ Moth birth confirmed") 
    elif message.startswith("Mating moths:"):
        print("✓ Moth mating confirmed")
    elif message.startswith("Killed moth:"):
        print("✓ Moth kill confirmed")
    else:
        print(f"Unknown response: {message}")

def main():
    """Example usage"""
    # Configuration
    config = UDPConfig(
        host="127.0.0.1",
        send_port=12345,
        receive_port=12346
    )
    
    # Create client
    client = UDPBridgeClient(config)
    client.on_message_received = message_handler
    
    # Start client
    if not client.start():
        print("Failed to start UDP client")
        return
    
    try:
        # Example commands
        print("\n=== Testing Moth Commands ===")
        
        # Generate new moth
        client.generate_new_moth("TestMoth001", "texture_001.png")
        time.sleep(0.5)
        
        # Birth moth
        client.birth_moth("BirthMoth001", "birth_texture.png")
        time.sleep(0.5)
        
        # Mating moths
        client.mating_moth("TestMoth001", "BirthMoth001")
        time.sleep(0.5)
        
        # Kill moth
        client.kill_moth("TestMoth001")
        time.sleep(0.5)
        
        # Send custom JSON message
        custom_data = {
            "command": "custom_command",
            "parameters": {
                "moth_id": "CustomMoth001",
                "properties": {"color": "red", "size": "large"}
            }
        }
        client.send_json_message(custom_data)
        
        # Keep running to receive responses
        print("\nListening for responses... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client.stop()

if __name__ == "__main__":
    main()