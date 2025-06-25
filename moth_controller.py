import os
import time
import json
import uuid
import logging
import threading
import requests
from pathlib import Path
from typing import Dict, Set
from dataclasses import dataclass
from udp_bridge_client import UDPBridgeClient, UDPConfig
from mask import GeneticMaskBlender
import cv2

@dataclass
class MothInfo:
    moth_id: str
    texture_path: str
    birth_time: float
    directory: str
    prompt: str = ""  # Prompt text from prompt.txt
    is_alive: bool = True
    has_mated: bool = False
    is_dead: bool = False
    mate_id: str = None
    mating_time: float = None
    current_lifespan: float = None  # Dynamic lifespan for this moth
    lifespan_extensions: int = 0    # How many times lifespan was extended

class MothController:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize UDP client
        udp_config = UDPConfig(
            host=self.config["udp_client"]["host"],
            send_port=self.config["udp_client"]["send_port"],
            receive_port=self.config["udp_client"]["receive_port"],
            buffer_size=self.config["udp_client"]["buffer_size"],
            timeout=self.config["udp_client"]["timeout"]
        )
        self.udp_client = UDPBridgeClient(udp_config)
        self.udp_client.on_message_received = self._handle_udp_message
        
        # Moth tracking
        self.moths: Dict[str, MothInfo] = {}
        self.monitored_directories: Set[str] = set()
        self.is_running = False
        self.monitor_thread = None
        self.lifecycle_thread = None
        
        # Create directories
        self.breeding_output_dir = self.config["moth_settings"]["breeding_output_dir"]
        os.makedirs(self.breeding_output_dir, exist_ok=True)
        
        # Population control settings
        self.min_moths = self.config["moth_settings"]["min_moths"]
        self.max_moths = self.config["moth_settings"]["max_moths"]
        self.lifecycle_check_interval = self.config["moth_settings"]["lifecycle_check_interval"]
        
        # Dynamic lifespan settings
        self.base_lifespan = self.config["moth_settings"]["base_lifespan_seconds"]
        self.min_lifespan = self.config["moth_settings"]["min_lifespan_seconds"]
        self.max_lifespan = self.config["moth_settings"]["max_lifespan_seconds"]
        self.lifespan_adjustment_factor = self.config["moth_settings"]["lifespan_adjustment_factor"]
        
        # Ollama settings
        self.ollama_enabled = self.config["ollama"]["enabled"]
        self.ollama_host = self.config["ollama"]["host"]
        self.ollama_model = self.config["ollama"]["model"]
        self.ollama_timeout = self.config["ollama"]["timeout"]
        self.ollama_prompt_template = self.config["ollama"]["prompt_template"]
        
        self.logger.info("Moth Controller initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config["logging"]["level"].upper())
        log_file = self.config["logging"]["log_file"]
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _handle_udp_message(self, message: str):
        """Handle incoming UDP messages"""
        self.logger.info(f"Received UDP message: {message}")
        
        if message.startswith("Generated moth:"):
            moth_id = message.split(":")[1].strip()
            self.logger.info(f"✓ Moth generation confirmed: {moth_id}")
        elif message.startswith("Killed moth:"):
            moth_id = message.split(":")[1].strip()
            if moth_id in self.moths:
                self.moths[moth_id].is_alive = False
            self.logger.info(f"✓ Moth kill confirmed: {moth_id}")
    
    def start(self):
        """Start the moth controller"""
        if not self.udp_client.start():
            self.logger.error("Failed to start UDP client")
            return False
        
        self.is_running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_directories, daemon=True)
        self.monitor_thread.start()
        
        # Start lifecycle management thread
        self.lifecycle_thread = threading.Thread(target=self._manage_lifecycle, daemon=True)
        self.lifecycle_thread.start()
        
        self.logger.info("Moth Controller started")
        return True
    
    def stop(self):
        """Stop the moth controller"""
        self.is_running = False
        self.udp_client.stop()
        self.logger.info("Moth Controller stopped")
    
    def _monitor_directories(self):
        """Monitor directories for new subdirectories"""
        watch_dir = Path(self.config["monitoring"]["watch_directory"])
        scan_interval = self.config["monitoring"]["scan_interval"]
        
        self.logger.info(f"Starting directory monitoring: {watch_dir}")
        
        while self.is_running:
            try:
                if watch_dir.exists():
                    current_dirs = {str(d) for d in watch_dir.iterdir() if d.is_dir()}
                    new_dirs = current_dirs - self.monitored_directories
                    
                    for new_dir in new_dirs:
                        self._process_new_directory(new_dir)
                        self.monitored_directories.add(new_dir)
                
                time.sleep(scan_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring directories: {e}")
                time.sleep(scan_interval)
    
    def _process_new_directory(self, directory_path: str):
        """Process a newly detected directory"""
        self.logger.info(f"New directory detected: {directory_path}")
        
        # Construct texture path
        texture_pattern = self.config["monitoring"]["texture_path_pattern"]
        texture_path = os.path.join(directory_path, texture_pattern)
        
        # Construct prompt path
        prompt_path = os.path.join(directory_path, "prompt.txt")
        
        # Check if texture file exists
        if os.path.exists(texture_path):
            # Read prompt text
            prompt_text = self._read_prompt_file(prompt_path)
            self._create_moth_from_texture(directory_path, texture_path, prompt_text)
        else:
            self.logger.warning(f"Texture file not found: {texture_path}")
    
    def _read_prompt_file(self, prompt_path: str) -> str:
        """Read prompt text from prompt.txt file"""
        try:
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()
                self.logger.info(f"Read prompt from {prompt_path}: {prompt_text[:50]}...")
                return prompt_text
            else:
                self.logger.warning(f"Prompt file not found: {prompt_path}")
                return ""
        except Exception as e:
            self.logger.error(f"Error reading prompt file {prompt_path}: {e}")
            return ""
    
    def _merge_prompts_with_ollama(self, prompt1: str, prompt2: str) -> str:
        """Use Ollama to merge two prompts into one coherent prompt"""
        if not self.ollama_enabled:
            # Fallback to simple concatenation if Ollama is disabled
            return f"Offspring of: [{prompt1}] + [{prompt2}]"
        
        try:
            # Format the prompt using the template
            merged_prompt_request = self.ollama_prompt_template.format(
                prompt1=prompt1, 
                prompt2=prompt2
            )
            
            # Prepare Ollama API request
            payload = {
                "model": self.ollama_model,
                "prompt": merged_prompt_request,
                "stream": False
            }
            
            # Send request to Ollama
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=self.ollama_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                merged_prompt = result.get("response", "").strip()
                
                if merged_prompt:
                    self.logger.info(f"Ollama merged prompts: '{prompt1}' + '{prompt2}' -> '{merged_prompt}'")
                    return merged_prompt
                else:
                    self.logger.warning("Ollama returned empty response")
                    
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Ollama request timed out after {self.ollama_timeout}s")
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Could not connect to Ollama at {self.ollama_host}")
        except Exception as e:
            self.logger.error(f"Error calling Ollama API: {e}")
        
        # Fallback to simple concatenation if Ollama fails
        fallback_prompt = f"Offspring of: [{prompt1}] + [{prompt2}]"
        self.logger.info(f"Using fallback prompt: {fallback_prompt}")
        return fallback_prompt
    
    def _create_moth_from_texture(self, directory: str, texture_path: str, prompt: str = ""):
        """Create a new moth from texture file"""
        if len(self.moths) >= self.max_moths:
            self.logger.warning("Maximum moth limit reached, skipping creation")
            return
        
        # Generate unique moth ID
        moth_id = str(uuid.uuid4())
        
        # Create moth info
        moth_info = MothInfo(
            moth_id=moth_id,
            texture_path=texture_path,
            birth_time=time.time(),
            directory=directory,
            prompt=prompt,
            current_lifespan=self.base_lifespan
        )
        
        # Send generation command with prompt
        success = self.udp_client.generate_new_moth(moth_id, texture_path, prompt)
        
        if success:
            self.moths[moth_id] = moth_info
            self.logger.info(f"Created new moth: {moth_id} with texture: {texture_path}")
        else:
            self.logger.error(f"Failed to create moth: {moth_id}")
    
    def _manage_lifecycle(self):
        """Manage moth lifecycle with dynamic lifespan adjustment"""
        while self.is_running:
            try:
                current_time = time.time()
                alive_moths = [moth for moth in self.moths.values() if moth.is_alive]
                alive_count = len(alive_moths)
                
                self.logger.debug(f"Lifecycle check: {alive_count} alive moths (min: {self.min_moths}, max: {self.max_moths})")
                
                # Adjust lifespans based on population needs
                self._adjust_lifespans_based_on_population(alive_moths, alive_count)
                
                # Check for moths that should die based on their individual lifespans
                moths_to_kill = []
                for moth_id, moth_info in self.moths.items():
                    if moth_info.is_alive:
                        age = current_time - moth_info.birth_time
                        if age > moth_info.current_lifespan:
                            moths_to_kill.append(moth_id)
                
                # Kill old moths, but respect minimum population
                for moth_id in moths_to_kill:
                    # Only kill if we won't go below minimum after this death
                    if alive_count > self.min_moths:
                        self._kill_moth(moth_id, "age")
                        alive_count -= 1
                    else:
                        # Extend lifespan instead of killing to maintain minimum population
                        self._extend_moth_lifespan(moth_id, "population_maintenance")
                
                time.sleep(self.lifecycle_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error managing lifecycle: {e}")
                time.sleep(self.lifecycle_check_interval)
    
    def _kill_moth(self, moth_id: str, reason: str = "manual"):
        """Kill a moth"""
        if moth_id not in self.moths:
            self.logger.warning(f"Moth not found: {moth_id}")
            return False
        
        success = self.udp_client.kill_moth(moth_id)
        
        if success:
            self.moths[moth_id].is_alive = False
            self.moths[moth_id].is_dead = True
            self.logger.info(f"Killed moth {moth_id} (reason: {reason})")
        else:
            self.logger.error(f"Failed to kill moth: {moth_id}")
        
        return success
    
    def _adjust_lifespans_based_on_population(self, alive_moths: list, alive_count: int):
        """Adjust moth lifespans based on current population needs"""
        if alive_count < self.min_moths:
            # Population too low - extend lifespans to keep moths alive longer
            self._extend_lifespans_for_all(alive_moths, "low_population")
        elif alive_count > self.max_moths:
            # Population too high - reduce lifespans to allow natural deaths
            self._reduce_lifespans_for_oldest(alive_moths, alive_count - self.max_moths)
    
    def _extend_lifespans_for_all(self, alive_moths: list, reason: str):
        """Extend lifespans for all living moths when population is low"""
        for moth in alive_moths:
            current_age = time.time() - moth.birth_time
            # Only extend if moth is getting close to death
            if current_age > (moth.current_lifespan * 0.8):
                self._extend_moth_lifespan(moth.moth_id, reason)
    
    def _reduce_lifespans_for_oldest(self, alive_moths: list, reduce_count: int):
        """Reduce lifespans for oldest moths when population is high"""
        # Sort by age (oldest first)
        oldest_moths = sorted(alive_moths, key=lambda m: m.birth_time)[:reduce_count]
        
        for moth in oldest_moths:
            new_lifespan = max(
                moth.current_lifespan / self.lifespan_adjustment_factor,
                self.min_lifespan
            )
            
            if new_lifespan != moth.current_lifespan:
                moth.current_lifespan = new_lifespan
                self.logger.info(f"Reduced lifespan for moth {moth.moth_id[:8]} to {new_lifespan:.1f}s (overpopulation)")
    
    def _extend_moth_lifespan(self, moth_id: str, reason: str):
        """Extend a specific moth's lifespan"""
        if moth_id not in self.moths:
            return
        
        moth = self.moths[moth_id]
        old_lifespan = moth.current_lifespan
        new_lifespan = min(
            moth.current_lifespan * self.lifespan_adjustment_factor,
            self.max_lifespan
        )
        
        if new_lifespan != old_lifespan:
            moth.current_lifespan = new_lifespan
            moth.lifespan_extensions += 1
            self.logger.info(f"Extended lifespan for moth {moth_id[:8]} from {old_lifespan:.1f}s to {new_lifespan:.1f}s ({reason})")
    
    def mate_moths(self, moth_id1: str, moth_id2: str) -> bool:
        """Mate two moths"""
        if moth_id1 not in self.moths or moth_id2 not in self.moths:
            self.logger.warning(f"One or both moths not found: {moth_id1}, {moth_id2}")
            return False
        
        moth1 = self.moths[moth_id1]
        moth2 = self.moths[moth_id2]
        
        if not moth1.is_alive or not moth2.is_alive:
            self.logger.warning(f"One or both moths are dead: {moth_id1}, {moth_id2}")
            return False
        
        if moth1.has_mated or moth2.has_mated:
            self.logger.warning(f"One or both moths have already mated: {moth_id1}, {moth_id2}")
            return False
        
        success = self.udp_client.mating_moth(moth_id1, moth_id2)
        
        if success:
            current_time = time.time()
            moth1.has_mated = True
            moth1.mate_id = moth_id2
            moth1.mating_time = current_time
            moth2.has_mated = True
            moth2.mate_id = moth_id1
            moth2.mating_time = current_time
            self.logger.info(f"Moths mated successfully: {moth_id1} + {moth_id2}")
            
            # Generate offspring texture using genetic algorithm
            offspring_texture = self._generate_offspring_texture(moth1.texture_path, moth2.texture_path)
            
            if offspring_texture:
                # Create offspring moth
                offspring_id = self._create_offspring_moth(moth_id1, moth_id2, offspring_texture)
                if offspring_id:
                    self.logger.info(f"Breeding complete! New offspring: {offspring_id}")
                else:
                    self.logger.error("Failed to create offspring moth")
            else:
                self.logger.error("Failed to generate offspring texture")
        else:
            self.logger.error(f"Failed to mate moths: {moth_id1}, {moth_id2}")
        
        return success
    
    def _generate_offspring_texture(self, parent1_texture: str, parent2_texture: str) -> str:
        """Generate new texture by blending two parent textures using genetic algorithm"""
        try:
            # Create unique offspring texture name
            offspring_id = str(uuid.uuid4())[:8]
            offspring_filename = f"offspring_{offspring_id}.png"
            offspring_path = os.path.join(self.breeding_output_dir, offspring_filename)
            
            # Initialize genetic mask blender
            blender = GeneticMaskBlender(
                parent1_texture, 
                parent2_texture,
                mask_resolution=tuple(self.config["moth_settings"]["mask_resolution"]),
                population_size=self.config["moth_settings"]["population_size"],
                generations=self.config["moth_settings"]["generations"],
                mutation_rate=self.config["moth_settings"]["mutation_rate"]
            )
            
            # Run genetic algorithm to find optimal blend
            self.logger.info(f"Starting texture blending: {parent1_texture} + {parent2_texture}")
            best_mask, best_blend = blender.run()
            
            # Resize and save the offspring texture
            offspring_texture = cv2.resize(best_blend, (512, 512))
            cv2.imwrite(offspring_path, offspring_texture)
            
            self.logger.info(f"Generated offspring texture: {offspring_path}")
            return offspring_path
            
        except Exception as e:
            self.logger.error(f"Error generating offspring texture: {e}")
            return None
    
    def _create_offspring_moth(self, parent1_id: str, parent2_id: str, offspring_texture: str):
        """Create a new moth from offspring texture"""
        if len(self.moths) >= self.max_moths:
            self.logger.warning("Maximum moth limit reached, cannot create offspring")
            return None
        
        # Generate unique offspring moth ID
        offspring_moth_id = str(uuid.uuid4())
        
        # Generate offspring prompt from parents using Ollama
        parent1 = self.moths[parent1_id]
        parent2 = self.moths[parent2_id]
        offspring_prompt = self._merge_prompts_with_ollama(parent1.prompt, parent2.prompt)
        
        # Create moth info for offspring
        offspring_info = MothInfo(
            moth_id=offspring_moth_id,
            texture_path=offspring_texture,
            birth_time=time.time(),
            directory=f"offspring_from_{parent1_id[:8]}_{parent2_id[:8]}",
            prompt=offspring_prompt,
            current_lifespan=self.base_lifespan
        )
        
        # Send birth command (for offspring moths) with prompt
        success = self.udp_client.birth_moth(offspring_moth_id, offspring_texture, offspring_prompt)
        
        if success:
            self.moths[offspring_moth_id] = offspring_info
            self.logger.info(f"Created offspring moth: {offspring_moth_id} from parents {parent1_id} + {parent2_id}")
            return offspring_moth_id
        else:
            self.logger.error(f"Failed to create offspring moth: {offspring_moth_id}")
            return None
    
    def get_moth_status(self) -> Dict:
        """Get current moth status"""
        alive_moths = sum(1 for moth in self.moths.values() if moth.is_alive)
        dead_moths = sum(1 for moth in self.moths.values() if moth.is_dead)
        mated_moths = sum(1 for moth in self.moths.values() if moth.has_mated)
        
        return {
            "total_moths": len(self.moths),
            "alive_moths": alive_moths,
            "dead_moths": dead_moths,
            "mated_moths": mated_moths,
            "monitored_directories": len(self.monitored_directories),
            "min_moths": self.min_moths,
            "max_moths": self.max_moths,
            "population_status": "normal" if self.min_moths <= alive_moths <= self.max_moths else 
                              ("low" if alive_moths < self.min_moths else "high")
        }
    
    def force_kill_moth(self, moth_id: str) -> bool:
        """Manually kill a specific moth"""
        return self._kill_moth(moth_id, "manual")
    
    def list_moths(self) -> Dict[str, Dict]:
        """List all moths with their info"""
        result = {}
        current_time = time.time()
        for moth_id, moth_info in self.moths.items():
            age = current_time - moth_info.birth_time
            remaining_lifespan = max(0, moth_info.current_lifespan - age) if moth_info.current_lifespan else 0
            
            result[moth_id] = {
                "uuid": moth_info.moth_id,
                "texture_path": moth_info.texture_path,
                "directory": moth_info.directory,
                "prompt": moth_info.prompt,
                "age_seconds": round(age, 2),
                "is_alive": moth_info.is_alive,
                "is_dead": moth_info.is_dead,
                "has_mated": moth_info.has_mated,
                "mate_id": moth_info.mate_id,
                "mating_time": moth_info.mating_time,
                "current_lifespan": moth_info.current_lifespan,
                "remaining_lifespan": round(remaining_lifespan, 2),
                "lifespan_extensions": moth_info.lifespan_extensions
            }
        return result

def main():
    """Main function to run the moth controller"""
    controller = MothController()
    
    try:
        if not controller.start():
            print("Failed to start Moth Controller")
            return
        
        print("Moth Controller is running...")
        print("Commands:")
        print("  status - Show moth status")
        print("  list - List all moths")
        print("  kill <moth_id> - Kill a specific moth")
        print("  mate <moth_id1> <moth_id2> - Mate two moths")
        print("  quit - Exit the controller")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "status":
                    status = controller.get_moth_status()
                    print(f"Status: {json.dumps(status, indent=2)}")
                elif command == "list":
                    moths = controller.list_moths()
                    print(f"Moths: {json.dumps(moths, indent=2)}")
                elif command.startswith("kill "):
                    moth_id = command.split(" ", 1)[1]
                    success = controller.force_kill_moth(moth_id)
                    print(f"Kill command sent: {'Success' if success else 'Failed'}")
                elif command.startswith("mate "):
                    parts = command.split(" ")
                    if len(parts) == 3:
                        moth_id1, moth_id2 = parts[1], parts[2]
                        success = controller.mate_moths(moth_id1, moth_id2)
                        print(f"Mate command sent: {'Success' if success else 'Failed'}")
                    else:
                        print("Usage: mate <moth_id1> <moth_id2>")
                else:
                    print("Unknown command")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        controller.stop()

if __name__ == "__main__":
    main()