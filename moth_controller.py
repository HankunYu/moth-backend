import os
import time
import json
import uuid
import logging
import threading
import requests
import random
import re
import subprocess
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
    mate_history: list = None  # List of mate_ids
    last_mating_time: float = None
    breeding_count: int = 0
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
        self.breeding_thread = None
        
        # Create directories - handle Windows paths in WSL
        breeding_dir = self.config["moth_settings"]["breeding_output_dir"]
        if self._is_windows_path(breeding_dir):
            # Convert Windows path to WSL path
            self.breeding_output_dir = self._windows_to_wsl_path(breeding_dir)
        elif not os.path.isabs(breeding_dir):
            # Convert relative path to absolute path
            self.breeding_output_dir = os.path.abspath(breeding_dir)
        else:
            self.breeding_output_dir = breeding_dir
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
        
        # Auto breeding settings
        self.auto_breeding_enabled = self.config["moth_settings"]["auto_breeding_enabled"]
        self.breeding_check_interval = self.config["moth_settings"]["breeding_check_interval"]
        self.min_breeding_interval = self.config["moth_settings"]["min_breeding_interval"]
        self.max_breeding_interval = self.config["moth_settings"]["max_breeding_interval"]
        self.min_remaining_lifespan_for_breeding = self.config["moth_settings"]["min_remaining_lifespan_for_breeding"]
        self.allow_multiple_breeding = self.config["moth_settings"]["allow_multiple_breeding"]
        
        # Ollama settings
        self.ollama_enabled = self.config["ollama"]["enabled"]
        self.ollama_host = self.config["ollama"]["host"]
        self.ollama_model = self.config["ollama"]["model"]
        self.ollama_timeout = self.config["ollama"]["timeout"]
        self.ollama_prompt_template = self.config["ollama"]["prompt_template"]
        
        # Perform startup self-check
        self._perform_startup_selfcheck()
        
        self.logger.info("Moth Controller initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
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
    
    def _perform_startup_selfcheck(self):
        """Perform startup self-check including Ollama connectivity test"""
        self.logger.info("Performing startup self-check...")
        
        # Test Ollama connectivity if enabled
        if self.ollama_enabled:
            self.logger.info("Testing Ollama connectivity...")
            if not self._test_ollama_connection():
                self.logger.error("Ollama connectivity test failed! Disabling Ollama...")
                self.ollama_enabled = False
            else:
                self.logger.info("Ollama connectivity test passed")
        else:
            self.logger.info("Ollama is disabled, skipping connectivity test")
        
        # Test UDP client can be created
        try:
            test_message = "test"
            self.logger.info("UDP client configuration validated")
        except Exception as e:
            self.logger.error(f"UDP client configuration error: {e}")
            raise
        
        # Check directories
        if not os.path.exists(self.breeding_output_dir):
            os.makedirs(self.breeding_output_dir, exist_ok=True)
            self.logger.info(f"Created breeding output directory: {self.breeding_output_dir}")
        
        self.logger.info("Startup self-check completed")
    
    def _test_ollama_connection(self) -> bool:
        """Test Ollama API connectivity and model availability"""
        try:
            # Test basic API connectivity
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                self.logger.error(f"Ollama API not responding: {response.status_code}")
                return False
            
            # Check if specified model is available
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            if self.ollama_model not in model_names:
                self.logger.error(f"Model '{self.ollama_model}' not found in Ollama. Available models: {model_names}")
                return False
            
            # Test a simple generation request
            test_payload = {
                "model": self.ollama_model,
                "prompt": "Test prompt",
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=test_payload,
                timeout=10
            )
            
            if response.status_code != 200:
                self.logger.error(f"Ollama generation test failed: {response.status_code} - {response.text}")
                return False
            
            result = response.json()
            if not result.get("response"):
                self.logger.error("Ollama returned empty response in test")
                return False
            
            return True
            
        except requests.exceptions.Timeout:
            self.logger.error("Ollama connection timeout")
            return False
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Cannot connect to Ollama at {self.ollama_host}")
            return False
        except Exception as e:
            self.logger.error(f"Ollama connectivity test error: {e}")
            return False
    
    def _clean_prompt_content(self, prompt: str) -> str:
        """Remove <think></think> tags and their content from prompt"""
        # Remove <think></think> blocks (case insensitive, multiline)
        cleaned = re.sub(r'<think>.*?</think>', '', prompt, flags=re.IGNORECASE | re.DOTALL)
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    def _is_wsl_environment(self) -> bool:
        """Check if running in WSL environment"""
        try:
            # Check if /proc/version contains 'microsoft' (WSL signature)
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                return 'microsoft' in version_info or 'wsl' in version_info
        except:
            return False
    
    def _is_windows_path(self, path: str) -> bool:
        """Check if path is in Windows format (e.g., C:\path or D:\path)"""
        return bool(re.match(r'^[A-Za-z]:[/\\]', path))
    
    def _windows_to_wsl_path(self, windows_path: str) -> str:
        """Convert Windows path to WSL path"""
        if not self._is_wsl_environment():
            return windows_path
        
        try:
            # Use wslpath command to convert Windows path to WSL path
            result = subprocess.run(['wslpath', windows_path], 
                                  capture_output=True, text=True, check=True)
            wsl_path = result.stdout.strip()
            self.logger.info(f"Converted Windows path to WSL: {windows_path} -> {wsl_path}")
            return wsl_path
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Failed to convert Windows path {windows_path}: {e}")
            return windows_path
        except FileNotFoundError:
            self.logger.warning("wslpath command not found, using original path")
            return windows_path
    
    def _wsl_to_windows_path(self, wsl_path: str) -> str:
        """Convert WSL path to Windows path for UDP commands"""
        if not self._is_wsl_environment():
            return wsl_path
        
        try:
            # Use wslpath with -w flag to convert WSL path to Windows path
            result = subprocess.run(['wslpath', '-w', wsl_path], 
                                  capture_output=True, text=True, check=True)
            windows_path = result.stdout.strip()
            self.logger.info(f"Converted WSL path to Windows: {wsl_path} -> {windows_path}")
            return windows_path
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Failed to convert WSL path {wsl_path}: {e}")
            return wsl_path
        except FileNotFoundError:
            self.logger.warning("wslpath command not found, using original path")
            return wsl_path
    
    def _ensure_absolute_path(self, path: str) -> str:
        """Convert relative path to absolute path if needed, with WSL path conversion"""
        # First convert to absolute path if needed
        if not os.path.isabs(path):
            absolute_path = os.path.abspath(path)
            self.logger.info(f"Converted relative path to absolute: {path} -> {absolute_path}")
            path = absolute_path
        
        # If in WSL and path looks like a Windows path, convert it
        if self._is_wsl_environment() and (path.startswith('C:') or path.startswith('D:') or 
                                          path.startswith('E:') or '\\' in path):
            path = self._windows_to_wsl_path(path)
        
        return path
    
    def _handle_udp_message(self, message: str):
        """Handle incoming UDP messages"""
        self.logger.info(f"Received UDP message: {message}")
        
        if message.startswith("Generated moth:"):
            moth_id = message.split(":")[1].strip()
            self.logger.info(f"Moth generation confirmed: {moth_id}")
        elif message.startswith("Killed moth:"):
            moth_id = message.split(":")[1].strip()
            if moth_id in self.moths:
                self.moths[moth_id].is_alive = False
            self.logger.info(f"Moth kill confirmed: {moth_id}")
    
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
        
        # Start auto breeding thread if enabled
        if self.auto_breeding_enabled:
            self.breeding_thread = threading.Thread(target=self._manage_auto_breeding, daemon=True)
            self.breeding_thread.start()
        
        self.logger.info("Moth Controller started")
        return True
    
    def stop(self):
        """Stop the moth controller"""
        self.is_running = False
        self.udp_client.stop()
        self.logger.info("Moth Controller stopped")
    
    def _monitor_directories(self):
        """Monitor directories for new subdirectories"""
        # Get the watch directory path from config
        watch_dir_str = self.config["monitoring"]["watch_directory"]
        
        # Check if it's a Windows path format (e.g., C:\path or D:\path)
        if self._is_windows_path(watch_dir_str):
            # Convert Windows path to WSL path
            watch_dir_str = self._windows_to_wsl_path(watch_dir_str)
        elif not os.path.isabs(watch_dir_str):
            # Convert relative path to absolute path
            watch_dir_str = os.path.abspath(watch_dir_str)
        
        watch_dir = Path(watch_dir_str)
        scan_interval = self.config["monitoring"]["scan_interval"]
        
        self.logger.info(f"Starting directory monitoring: {watch_dir}")
        
        while self.is_running:
            try:
                if watch_dir.exists():
                    current_dirs = {str(d) for d in watch_dir.iterdir() if d.is_dir()}
                    new_dirs = current_dirs - self.monitored_directories
                    
                    for i, new_dir in enumerate(new_dirs):
                        if i > 0:  # Add 1 second cooldown between generations (except for first one)
                            time.sleep(1)
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
                # Try different encodings to handle various file formats
                encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
                prompt_text = ""
                
                for encoding in encodings:
                    try:
                        with open(prompt_path, 'r', encoding=encoding) as f:
                            prompt_text = f.read().strip()
                        self.logger.debug(f"Successfully read prompt file with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if not prompt_text:
                    self.logger.warning(f"Could not decode prompt file with any supported encoding: {prompt_path}")
                    return ""
                
                # Clean prompt content before returning
                cleaned_prompt = self._clean_prompt_content(prompt_text)
                self.logger.info(f"Read prompt from {prompt_path}: {cleaned_prompt[:50]}...")
                return cleaned_prompt
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
            fallback_prompt = f"Offspring of: [{prompt1}] + [{prompt2}]"
            return self._clean_prompt_content(fallback_prompt)
        
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
                    # Remove <think></think> content from the merged prompt
                    cleaned_prompt = self._clean_prompt_content(merged_prompt)
                    self.logger.info(f"Ollama merged prompts: '{prompt1}' + '{prompt2}' -> '{cleaned_prompt}'")
                    return cleaned_prompt
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
        # Clean the fallback prompt as well
        cleaned_fallback = self._clean_prompt_content(fallback_prompt)
        self.logger.info(f"Using fallback prompt: {cleaned_fallback}")
        return cleaned_fallback
    
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
        
        # Ensure texture path is absolute
        abs_texture_path = self._ensure_absolute_path(texture_path)
        
        # Convert path to Windows format for UDP command if in WSL
        udp_texture_path = self._wsl_to_windows_path(abs_texture_path)
        
        # Send generation command with prompt
        print(f"Creating new moth: {moth_id} with texture: {abs_texture_path}")
        success = self.udp_client.generate_new_moth(moth_id, udp_texture_path, prompt)
        
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
    
    def _manage_auto_breeding(self):
        """Manage automatic breeding between suitable moths"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Get eligible moths for breeding
                eligible_moths = self._get_eligible_moths_for_breeding(current_time)
                
                if len(eligible_moths) >= 2:
                    # Randomly select two moths for breeding
                    moth1, moth2 = random.sample(eligible_moths, 2)
                    
                    self.logger.info(f"Auto breeding attempt: {moth1.moth_id[:8]} + {moth2.moth_id[:8]}")
                    success = self.mate_moths(moth1.moth_id, moth2.moth_id)
                    
                    if success:
                        self.logger.info(f"Auto breeding successful: {moth1.moth_id[:8]} + {moth2.moth_id[:8]}")
                    else:
                        self.logger.warning(f"Auto breeding failed: {moth1.moth_id[:8]} + {moth2.moth_id[:8]}")
                
                # Wait for next breeding check with some randomness
                sleep_time = random.uniform(
                    self.breeding_check_interval * 0.8,
                    self.breeding_check_interval * 1.2
                )
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error managing auto breeding: {e}")
                time.sleep(self.breeding_check_interval)
    
    def _get_eligible_moths_for_breeding(self, current_time: float) -> list:
        """Get list of moths eligible for breeding"""
        eligible_moths = []
        
        for moth in self.moths.values():
            if not moth.is_alive:
                continue
                
            # Check if moth has enough remaining lifespan
            age = current_time - moth.birth_time
            remaining_lifespan = moth.current_lifespan - age
            
            if remaining_lifespan < self.min_remaining_lifespan_for_breeding:
                continue
            
            # Check breeding cooldown
            if moth.last_mating_time is not None:
                time_since_last_mating = current_time - moth.last_mating_time
                breeding_interval = random.uniform(
                    self.min_breeding_interval,
                    self.max_breeding_interval
                )
                
                if time_since_last_mating < breeding_interval:
                    continue
            
            eligible_moths.append(moth)
        
        return eligible_moths
    
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
        
        if not self.allow_multiple_breeding and (moth1.has_mated or moth2.has_mated):
            self.logger.warning(f"One or both moths have already mated: {moth_id1}, {moth_id2}")
            return False
        
        success = self.udp_client.mating_moth(moth_id1, moth_id2)
        
        if success:
            current_time = time.time()
            moth1.has_mated = True
            moth1.last_mating_time = current_time
            moth1.breeding_count += 1
            if moth1.mate_history is None:
                moth1.mate_history = []
            moth1.mate_history.append(moth_id2)
            
            moth2.has_mated = True
            moth2.last_mating_time = current_time
            moth2.breeding_count += 1
            if moth2.mate_history is None:
                moth2.mate_history = []
            moth2.mate_history.append(moth_id1)
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
        
        # Ensure offspring texture path is absolute
        abs_offspring_texture = self._ensure_absolute_path(offspring_texture)
        
        # Convert path to Windows format for UDP command if in WSL
        udp_offspring_texture = self._wsl_to_windows_path(abs_offspring_texture)
        
        # Send birth command (for offspring moths) with prompt
        success = self.udp_client.birth_moth(offspring_moth_id, udp_offspring_texture, offspring_prompt)
        
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
                "mate_history": moth_info.mate_history or [],
                "last_mating_time": moth_info.last_mating_time,
                "breeding_count": moth_info.breeding_count,
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