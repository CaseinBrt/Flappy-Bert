import cv2
import mediapipe as mp
import pygame
import sounddevice as sd
import numpy as np
import math
import sys
import json
import os
from datetime import datetime
import random  # Add import for random
import glob  # For loading gif frames
try:
    from PIL import Image  # For GIF handling
    PIL_AVAILABLE = True
    print("PIL successfully imported. GIF loading enabled.")
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. GIF loading will be limited.")

# Initialize Pygame and window
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Flappy Bert")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 177, 76)
GRAY = (128, 128, 128)
SKY_BLUE = (135, 206, 235)
NIGHT_SKY = (25, 25, 112)  # Midnight blue for night mode
DARK_GREEN = (0, 100, 0)  # Dark green for the start of the gradient

# Leaderboard functionality
LEADERBOARD_FILE = "leaderboard.json"

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_leaderboard(leaderboard):
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(leaderboard, f)

def add_score_to_leaderboard(name, score):
    leaderboard = load_leaderboard()
    leaderboard.append({"name": name, "score": score, "date": datetime.now().strftime("%Y-%m-%d %H:%M")})
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    leaderboard = leaderboard[:10]  # Keep only top 10 scores
    save_leaderboard(leaderboard)

def reset_leaderboard():
    # Create an empty leaderboard and save it
    save_leaderboard([])
    print("Leaderboard has been reset")

# Text input functionality
class TextInput:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = ""
        self.active = True
        self.font = pygame.font.SysFont(None, 36)
        
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if len(self.text) < 15:  # Limit name length
                    self.text += event.unicode
        return False
    
    def draw(self):
        # Draw the input box
        pygame.draw.rect(screen, WHITE if self.active else GRAY, self.rect, 2)
        # Render the text with outline
        draw_text_with_outline(self.text, self.font, WHITE, BLACK, 
                             self.rect.centerx - self.font.size(self.text)[0] // 2,
                             self.rect.centery - self.font.size(self.text)[1] // 2)

def draw_text_with_outline(text, font, text_color, outline_color, x, y):
    # Render the outline by offsetting the text in multiple directions
    outline_surfaces = []
    for offset_x, offset_y in [(-1,-1), (-1,1), (1,-1), (1,1), (-1,0), (1,0), (0,-1), (0,1)]:
        outline_surface = font.render(text, True, outline_color)
        screen.blit(outline_surface, (x + offset_x, y + offset_y))
    
    # Render the main text on top
    text_surface = font.render(text, True, text_color)
    screen.blit(text_surface, (x, y))

# Game states
STATE_NAME_INPUT = "name_input"
STATE_PLAYING = "playing"
STATE_GAME_OVER = "game_over"
STATE_LEADERBOARD = "leaderboard"
STATE_SETTINGS = "settings"

# Adjust sensitivity level range
sensitivity_level = 50  # Default sensitivity level
max_sensitivity_level = 100  # Maximum sensitivity level

# Load the wing image
wing_image = pygame.image.load('wing.png')  # Ensure 'wing.png' is in the same directory
wing_image = pygame.transform.scale(wing_image, (25, 35))  # Scale to fit the wing size

# Load the bat wing image for night mode
bat_wing_image = pygame.image.load('bat_wing.png')  # Ensure 'bat_wing.png' is in the same directory
bat_wing_image = pygame.transform.scale(bat_wing_image, (25, 35))  # Scale to fit the wing size

# Face Bird class
class FaceBird:
    def __init__(self):
        self.x = 100
        self.y = height // 2
        self.width = 50
        self.height = 50
        self.velocity = 0
        self.gravity = 0.5
        self.lift = -4
        self.face_image = None
        self.wing_flap = 0  # Variable to animate wing flapping
        self.wing_color = WHITE
        self.wing_style = 'regular'
        
    def update(self, voice_intensity):
        # Convert voice intensity to a boolean jump flag
        should_jump = voice_intensity > 0.001
        
        # Apply gravity always
        self.velocity += self.gravity
        
        # Apply lift based on voice intensity (if above threshold)
        if should_jump:
            self.velocity = self.lift
            
        # Update position
        self.y += self.velocity
        
        # Screen boundaries
        if self.y > height - self.height:
            self.y = height - self.height
            self.velocity = 0
        if self.y < 0:
            self.y = 0
            self.velocity = 0

        # Update wing flap for animation
        self.wing_flap = (self.wing_flap + 1) % 20  # Simple flap animation

    def draw(self):
        # Draw wings using the wing icon
        wing_offset_y = 5 * math.sin(self.wing_flap * math.pi / 10)  # Add flapping animation
        
        # Choose which wing image to use based on the wing style
        if self.wing_style == 'bat':
            # Use the bat wing image for night mode
            current_wing_image = bat_wing_image
        else:
            # Use the regular wing image
            current_wing_image = wing_image
            
        # Left wing
        left_wing = current_wing_image.copy()
        if self.wing_style != 'bat' and self.wing_color != WHITE:
            # For colored wings, tint the wing image to the specified color
            color_surface = pygame.Surface(left_wing.get_size()).convert_alpha()
            color_surface.fill(self.wing_color)
            left_wing.blit(color_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            
        # Flip the left wing horizontally
        left_wing = pygame.transform.flip(left_wing, True, False)
        screen.blit(left_wing, (self.x - current_wing_image.get_width(), self.y + self.height // 4 + wing_offset_y))
        
        # Right wing (no need to flip)
        right_wing = current_wing_image.copy()
        if self.wing_style != 'bat' and self.wing_color != WHITE:
            # For colored wings, tint the wing image to the specified color
            color_surface = pygame.Surface(right_wing.get_size()).convert_alpha()
            color_surface.fill(self.wing_color)
            right_wing.blit(color_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            
        screen.blit(right_wing, (self.x + self.width, self.y + self.height // 4 + wing_offset_y))

        # Draw face
        if self.face_image is not None:
            # Resize face image to fit bird dimensions
            resized_face = cv2.resize(self.face_image, (self.width, self.height))
            # Create a circular mask
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.circle(mask, (self.width // 2, self.height // 2), self.width // 2, 255, -1)
            # Apply the mask to the face image
            masked_face = cv2.bitwise_and(resized_face, resized_face, mask=mask)
            # Convert masked face to RGB
            masked_face_rgb = cv2.cvtColor(masked_face, cv2.COLOR_BGR2RGB)
            # Create a surface from the masked face
            face_surface = pygame.surfarray.make_surface(masked_face_rgb)
            face_surface.set_colorkey((0, 0, 0))  # Set black as transparent
            # Draw the face
            screen.blit(face_surface, (self.x, self.y))
        else:
            # Fallback if no face detected yet
            placeholder = pygame.Surface((self.width, self.height))
            placeholder.fill((200, 200, 200))
            screen.blit(placeholder, (self.x, self.y))

# Game constants
PIPE_SPEED = 4  # Increased speed for more difficulty
PIPE_GAP_MIN = 200  # Minimum gap size
PIPE_GAP_MAX = 300  # Maximum gap size
PIPE_SPAWN_TIME = 2000  # Time in milliseconds between pipe spawns

# Function to draw a gradient rectangle
def draw_gradient_rect(surface, color1, color2, rect):
    """Draw a vertical gradient rectangle."""
    x, y, width, height = rect
    for i in range(height):
        ratio = i / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pygame.draw.line(surface, (r, g, b), (x, y + i), (x + width, y + i))

# Pipe class
class Pipe:
    def __init__(self):
        self.x = width
        self.gap = np.random.randint(PIPE_GAP_MIN, PIPE_GAP_MAX)  # Randomize the gap size
        self.height = np.random.randint(80, height - self.gap - 80)  # Adjusted range for better balance with new gap
        self.passed = False
        self.width = 60
        
    def draw(self):
        # Draw top pipe with gradient and black outline
        pipe_rect_top = pygame.Rect(self.x, 0, self.width, self.height)
        draw_gradient_rect(screen, DARK_GREEN, GREEN, pipe_rect_top)
        pygame.draw.rect(screen, BLACK, pipe_rect_top, 2)  # Black outline

        # Draw top pipe cap
        cap_height = 20
        pipe_cap_top = pygame.Rect(self.x - 5, self.height - cap_height, self.width + 10, cap_height)
        pygame.draw.rect(screen, GREEN, pipe_cap_top)
        pygame.draw.rect(screen, BLACK, pipe_cap_top, 2)  # Black outline

        # Draw bottom pipe with gradient and black outline
        pipe_rect_bottom = pygame.Rect(self.x, self.height + self.gap, self.width, height - self.height - self.gap)
        draw_gradient_rect(screen, DARK_GREEN, GREEN, pipe_rect_bottom)
        pygame.draw.rect(screen, BLACK, pipe_rect_bottom, 2)  # Black outline

        # Draw bottom pipe cap
        pipe_cap_bottom = pygame.Rect(self.x - 5, self.height + self.gap, self.width + 10, cap_height)
        pygame.draw.rect(screen, GREEN, pipe_cap_bottom)
        pygame.draw.rect(screen, BLACK, pipe_cap_bottom, 2)  # Black outline
        
    def update(self):
        self.x -= PIPE_SPEED
        
    def collide(self, bird):
        bird_rect = pygame.Rect(bird.x, bird.y, bird.width, bird.height)
        top_pipe = pygame.Rect(self.x, 0, self.width, self.height)
        bottom_pipe = pygame.Rect(self.x, self.height + self.gap, self.width, height - self.height - self.gap)
        
        return bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe)

    def is_offscreen(self):
        return self.x + self.width < 0

# Collectable class for power-ups
class Collectable:
    def __init__(self):
        self.width = 30
        self.height = 30
        self.x = width + self.width
        self.y = np.random.randint(50, height - 50)
        self.collected = False
        self.color = (255, 223, 0)  # Gold color for collectable

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self):
        pygame.draw.ellipse(screen, self.color, (self.x, self.y, self.width, self.height))

    def collide(self, bird):
        bird_rect = pygame.Rect(bird.x, bird.y, bird.width, bird.height)
        collectable_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        return bird_rect.colliderect(collectable_rect)

    def is_offscreen(self):
        return self.x + self.width < 0

# Initialize Face Detection with landmark detection for better face extraction
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize audio parameters
sample_rate = 44100
block_size = 512  # Reduced block size for faster response
audio_threshold = 0.005  # Lower threshold for better sensitivity
last_jump_time = 0  # Reset to 0 to avoid initial delay
jump_cooldown = 300  # Reduced cooldown for more responsive jumps

# Global variable to store current voice intensity
current_voice_intensity = 0.0

# Audio callback function - runs in a separate thread
def audio_callback(indata, frames, time, status):
    global current_voice_intensity
    if status:
        print(status)
    # Calculate volume
    volume_norm = np.linalg.norm(indata) / len(indata)
    current_voice_intensity = volume_norm
    print(f"Current voice intensity: {current_voice_intensity}")  # Debug print to check intensity

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    pygame.quit()
    sys.exit()

# Display initial loading message
font = pygame.font.SysFont(None, 36)
screen.fill((0, 0, 0))
loading_text = font.render("Loading camera and audio...", True, (255, 255, 255))
screen.blit(loading_text, (width//2 - loading_text.get_width()//2, height//2))
pygame.display.flip()

# Start audio stream
try:
    stream = sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=sample_rate,
        blocksize=block_size
    )
    stream.start()
except Exception as e:
    print(f"Error starting audio stream: {e}")
    pygame.quit()
    sys.exit()

# Main game loop
bird = FaceBird()
running = True
clock = pygame.time.Clock()
start_time = pygame.time.get_ticks()
score = 0

# Game state
game_state = STATE_NAME_INPUT
game_active = False
game_over = False

# Reset confirmation variables
show_reset_confirmation = False
reset_confirmation_time = 0

# Initialize name input
name_input = TextInput(width//2 - 150, height//2 - 25, 300, 50)
player_name = ""

# Initialize pipes
pipes = []
last_pipe_spawn = 0

# Cloud class for background animation
class Cloud:
    def __init__(self):
        self.width = np.random.randint(100, 200)
        self.height = np.random.randint(40, 80)
        self.x = width + self.width
        self.y = np.random.randint(0, height//2)
        self.speed = np.random.uniform(0.5, 1.5)
        
    def update(self):
        self.x -= self.speed
        
    def draw(self):
        # Draw main cloud body
        pygame.draw.ellipse(screen, WHITE, (self.x, self.y, self.width, self.height))
        # Draw additional cloud puffs
        puff_size = self.height * 0.8
        pygame.draw.ellipse(screen, WHITE, (self.x + self.width * 0.2, self.y - self.height * 0.1, 
                                          puff_size, puff_size))
        pygame.draw.ellipse(screen, WHITE, (self.x + self.width * 0.4, self.y - self.height * 0.15, 
                                          puff_size, puff_size))
    
    def is_offscreen(self):
        return self.x + self.width < 0

# Initialize clouds
clouds = [Cloud() for _ in range(3)]  # Start with 3 clouds
CLOUD_SPAWN_TIME = 4000  # Time between cloud spawns in milliseconds
last_cloud_spawn = pygame.time.get_ticks()

# Display info
font = pygame.font.SysFont(None, 24)
title_font = pygame.font.SysFont(None, 48)
debug_font = pygame.font.SysFont(None, 18)

# Debug variables
show_debug = True

# Load sound effects
bell_sound = pygame.mixer.Sound('bell.wav')  # Ensure 'bell.wav' is in the same directory
fly_sound = pygame.mixer.Sound('fly.wav')  # Ensure 'fly.wav' is in the same directory
hit_sound = pygame.mixer.Sound('hit.wav')  # Ensure 'hit.wav' is in the same directory
explosion_sound = pygame.mixer.Sound('explosion.wav')  # Ensure 'explosion.wav' is in the same directory
bell_sound.set_volume(0.3)  # Set bell sound volume to 30%
fly_sound.set_volume(0.5)  # Set fly sound volume to 50%
hit_sound.set_volume(0.5)  # Set hit sound volume to 50%
explosion_sound.set_volume(0.5)  # Set explosion sound volume to 50%

# Initialize collectables
collectables = []
COLLECTABLE_SPAWN_TIME = 3000  # Time between collectable spawns in milliseconds
last_collectable_spawn = pygame.time.get_ticks()

# Add a variable to track the last score milestone
last_collectable_score = -1

# Load collectable sound effect
collectable_sound = pygame.mixer.Sound('collectable.wav')  # Ensure 'collectable.wav' is in the same directory
collectable_sound.set_volume(0.5)  # Set collectable sound volume to 50%

# Load game over sound effect
try:
    game_over_sound = pygame.mixer.Sound('game_over.wav')  # Ensure 'game_over.wav' is in the same directory
    game_over_sound.set_volume(0.5)  # Set game over sound volume to 50%
except pygame.error as e:
    print(f"Error loading game over sound: {e}")  # Log error if sound fails to load

# Function to draw the settings menu
def draw_settings_menu():
    screen.fill(BLACK)
    title_x = width//2 - title_font.size("Settings")[0]//2
    draw_text_with_outline("Settings", title_font, WHITE, BLACK, title_x, height//4)
    
    sensitivity_text = f"Audio Sensitivity: {sensitivity_level}/{max_sensitivity_level}"
    sensitivity_x = width//2 - font.size(sensitivity_text)[0]//2
    draw_text_with_outline(sensitivity_text, font, WHITE, BLACK, sensitivity_x, height//2)
    
    instruction_text = "Use UP/DOWN to adjust, ENTER to save"
    instruction_x = width//2 - font.size(instruction_text)[0]//2
    draw_text_with_outline(instruction_text, font, WHITE, BLACK, instruction_x, height//2 + 50)

# Define a button class for the menu
class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.SysFont(None, 36)

    def draw(self):
        # Draw filled rectangle with a bright, contrasting color
        pygame.draw.rect(screen, GREEN, self.rect)  # Use green for visibility
        pygame.draw.rect(screen, WHITE, self.rect, 2)  # Outline
        # Render text with outline
        draw_text_with_outline(self.text, self.font, BLACK, WHITE,
                             self.rect.centerx - self.font.size(self.text)[0] // 2,
                             self.rect.centery - self.font.size(self.text)[1] // 2)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Initialize the menu button
menu_button = Button(width - 110, 10, 100, 40, "Menu")

# Load the pterodactyl image
pterodactyl_image = pygame.image.load('pterodactyl.png')  # Ensure 'pterodactyl.png' is in the same directory
pterodactyl_image = pygame.transform.scale(pterodactyl_image, (50, 30))  # Scale to fit the flying bird size

# Flying Bird class for background animation
class FlyingBird:
    def __init__(self):
        self.width = 50
        self.height = 30
        self.x = width + self.width
        self.y = np.random.randint(50, height // 2)
        self.speed = np.random.uniform(1.0, 2.0)

    def update(self):
        self.x -= self.speed

    def draw(self):
        # Draw the pterodactyl image
        screen.blit(pterodactyl_image, (self.x, self.y))

    def is_offscreen(self):
        return self.x + self.width < 0

# Initialize flying birds
flying_birds = [FlyingBird() for _ in range(2)]  # Start with 2 flying birds
FLYING_BIRD_SPAWN_TIME = 5000  # Time between flying bird spawns in milliseconds
last_flying_bird_spawn = pygame.time.get_ticks()

# RainDrop class for rain effect
class RainDrop:
    def __init__(self):
        self.x = random.randint(0, width)
        self.y = random.randint(-height, 0)
        self.length = random.randint(5, 15)
        self.speed = random.uniform(2.0, 5.0)

    def update(self):
        self.y += self.speed
        if self.y > height:
            self.y = random.randint(-height, 0)
            self.x = random.randint(0, width)

    def draw(self):
        pygame.draw.line(screen, WHITE, (self.x, self.y), (self.x, self.y + self.length), 1)

# Initialize rain effect
rain_drops = [RainDrop() for _ in range(100)]  # Create 100 raindrops
is_raining = random.choice([True, False])  # Randomly decide if it will rain
rain_toggle_time = pygame.time.get_ticks() + 40000  # Toggle rain every 40 seconds
rain_intensity = random.randint(50, 150)  # Random number of raindrops

# Load the mountain image
mountain_image = pygame.image.load('mountain.png')  # Ensure 'mountain.png' is in the same directory
mountain_image = pygame.transform.scale(mountain_image, (200, 150))  # Scale to fit the desired size

# Load the building image for night mode
building_image = pygame.image.load('building.png')  # Ensure 'building.png' is in the same directory
building_image = pygame.transform.scale(building_image, (200, 180))  # Scale to fit the desired size

# Load the coconut tree image for day mode
coconut_tree_image = pygame.image.load('coconut_tree.png')  # Ensure 'coconut_tree.png' is in the same directory
coconut_tree_image = pygame.transform.scale(coconut_tree_image, (80, 120))  # Scale to fit the desired size

# Load the cactus image for sunset mode
cactus_image = pygame.image.load('cactus.png')  # Ensure 'cactus.png' is in the same directory
cactus_image = pygame.transform.scale(cactus_image, (60, 100))  # Scale to fit the desired size

# Load the tumbleweed image for sunset mode
tumbleweed_image = pygame.image.load('tumbleweed.png')  # Ensure 'tumbleweed.png' is in the same directory
tumbleweed_image = pygame.transform.scale(tumbleweed_image, (40, 40))  # Scale to fit the desired size

# Load the star image for night mode
star_image = pygame.image.load('star.png')  # Ensure 'star.png' is in the same directory
star_image = pygame.transform.scale(star_image, (20, 20))  # Scale to fit the desired size

# Load the moon image for night mode
moon_image = pygame.image.load('moon.png')  # Ensure 'moon.png' is in the same directory
moon_image = pygame.transform.scale(moon_image, (80, 80))  # Scale to fit the desired size

# Load Lapras gif frames for morning mode
lapras_frames = []
try:
    # First try to load a GIF file directly using PIL if available
    if PIL_AVAILABLE:
        try:
            gif_path = 'lapras.gif'
            if os.path.exists(gif_path):
                print(f"Loading GIF file: {gif_path}")
                gif = Image.open(gif_path)
                frame_count = 0
                
                # Extract all frames from the GIF
                while True:
                    try:
                        gif.seek(frame_count)
                        frame_img = gif.convert('RGBA')
                        frame_data = np.array(frame_img)
                        # Convert PIL image to pygame surface
                        frame_surface = pygame.image.frombuffer(
                            frame_data.tobytes(), frame_img.size, 'RGBA')
                        # Scale the frame
                        frame_surface = pygame.transform.scale(frame_surface, (100, 80))
                        lapras_frames.append(frame_surface)
                        frame_count += 1
                    except EOFError:
                        # End of frames
                        break
                
                print(f"Loaded {len(lapras_frames)} frames from Lapras GIF")
        except Exception as e:
            print(f"Error loading Lapras GIF: {e}")
            # Fall back to other methods if GIF loading fails
            lapras_frames = []
    
    # If GIF loading failed or PIL is not available, try loading individual frames
    if not lapras_frames:
        # Try to load individual frames
        lapras_files = sorted(glob.glob('lapras_*.png'))
        if lapras_files:
            for file in lapras_files:
                frame = pygame.image.load(file)
                frame = pygame.transform.scale(frame, (100, 80))
                lapras_frames.append(frame)
            print(f"Loaded {len(lapras_frames)} individual Lapras frame files")
        else:
            # If no individual frames, try to load a single image
            lapras_img = pygame.image.load('lapras.png')
            lapras_img = pygame.transform.scale(lapras_img, (100, 80))
            lapras_frames = [lapras_img]
            print("Loaded single Lapras image")
    
    if not lapras_frames:
        raise Exception("No Lapras images found")
        
except Exception as e:
    print(f"Error loading Lapras images: {e}")
    # Create a placeholder if loading fails
    lapras_frames = [pygame.Surface((100, 80))]
    lapras_frames[0].fill((0, 105, 255))  # Blue color as placeholder
    print("Using placeholder for Lapras")

# Load Charizard gif frames for sunset mode
charizard_frames = []
try:
    # First try to load a GIF file directly using PIL if available
    if PIL_AVAILABLE:
        try:
            gif_path = 'charizard.gif'
            if os.path.exists(gif_path):
                print(f"Loading GIF file: {gif_path}")
                gif = Image.open(gif_path)
                frame_count = 0
                
                # Extract all frames from the GIF
                while True:
                    try:
                        gif.seek(frame_count)
                        frame_img = gif.convert('RGBA')
                        frame_data = np.array(frame_img)
                        # Convert PIL image to pygame surface
                        frame_surface = pygame.image.frombuffer(
                            frame_data.tobytes(), frame_img.size, 'RGBA')
                        # Scale the frame
                        frame_surface = pygame.transform.scale(frame_surface, (120, 100))
                        charizard_frames.append(frame_surface)
                        frame_count += 1
                    except EOFError:
                        # End of frames
                        break
                
                print(f"Loaded {len(charizard_frames)} frames from GIF")
        except Exception as e:
            print(f"Error loading GIF: {e}")
            # Fall back to other methods if GIF loading fails
            charizard_frames = []
    
    # If GIF loading failed or PIL is not available, try loading individual frames
    if not charizard_frames:
        # Try to load individual frames
        charizard_files = sorted(glob.glob('charizard_*.png'))
        if charizard_files:
            for file in charizard_files:
                frame = pygame.image.load(file)
                frame = pygame.transform.scale(frame, (120, 100))
                charizard_frames.append(frame)
            print(f"Loaded {len(charizard_frames)} individual frame files")
        else:
            # If no individual frames, try to load a single image
            charizard_img = pygame.image.load('charizard.png')
            charizard_img = pygame.transform.scale(charizard_img, (120, 100))
            charizard_frames = [charizard_img]
            print("Loaded single charizard image")
    
    if not charizard_frames:
        raise Exception("No Charizard images found")
        
except Exception as e:
    print(f"Error loading Charizard images: {e}")
    # Create a placeholder if loading fails
    charizard_frames = [pygame.Surface((120, 100))]
    charizard_frames[0].fill((255, 165, 0))  # Orange color as placeholder
    print("Using placeholder for Charizard")

# Star class for night mode
class Star:
    def __init__(self):
        # Randomize star size for more variety
        size = random.randint(10, 30)
        self.image = pygame.transform.scale(star_image, (size, size))
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.x = random.randint(0, width)
        self.y = random.randint(0, height // 2)  # Stars only in the upper half of the screen
        self.twinkle_speed = random.uniform(0.01, 0.05)
        self.twinkle_value = random.random()
        self.twinkle_direction = 1  # 1 for increasing brightness, -1 for decreasing
        self.visible = True
        self.visibility_change_time = pygame.time.get_ticks() + random.randint(5000, 15000)  # Time until visibility might change

    def update(self):
        current_time = pygame.time.get_ticks()
        
        # Check if it's time to change visibility
        if current_time > self.visibility_change_time:
            # 20% chance to toggle visibility
            if random.random() < 0.2:
                self.visible = not self.visible
                
                # If becoming visible, reposition the star
                if self.visible:
                    self.x = random.randint(0, width)
                    self.y = random.randint(0, height // 2)
                    # Randomize star size again
                    size = random.randint(10, 30)
                    self.image = pygame.transform.scale(star_image, (size, size))
                    self.width = self.image.get_width()
                    self.height = self.image.get_height()
            
            # Set next visibility change time
            self.visibility_change_time = current_time + random.randint(5000, 15000)
        
        # Only update twinkling if visible
        if self.visible:
            # Make the star twinkle by changing its alpha value
            self.twinkle_value += self.twinkle_speed * self.twinkle_direction
            if self.twinkle_value > 1.0:
                self.twinkle_value = 1.0
                self.twinkle_direction = -1
            elif self.twinkle_value < 0.3:  # Don't go completely dark
                self.twinkle_value = 0.3
                self.twinkle_direction = 1

    def draw(self):
        # Only draw if visible
        if self.visible:
            # Create a copy of the star image with adjusted alpha for twinkling effect
            twinkle_star = self.image.copy()
            twinkle_star.set_alpha(int(self.twinkle_value * 255))
            screen.blit(twinkle_star, (self.x, self.y))

# Initialize stars for night mode
stars = [Star() for _ in range(60)]  # Create more stars to ensure enough are visible at any time

# Coconut Tree class for random placement in day mode
class CoconutTree:
    def __init__(self):
        self.width = coconut_tree_image.get_width()
        self.height = coconut_tree_image.get_height()
        self.x = width + self.width
        # Place trees at ground level with some random variation
        self.y = height - self.height + random.randint(-10, 10)
        self.speed = 2  # Move slower than pipes for parallax effect

    def update(self):
        self.x -= self.speed

    def draw(self):
        screen.blit(coconut_tree_image, (self.x, self.y))

    def is_offscreen(self):
        return self.x + self.width < 0

# Cactus class for random placement in sunset mode
class Cactus:
    def __init__(self):
        self.width = cactus_image.get_width()
        self.height = cactus_image.get_height()
        self.x = width + self.width
        # Place cacti at ground level with some random variation
        self.y = height - self.height + random.randint(-5, 5)
        self.speed = 1.5  # Move slower than pipes for parallax effect

    def update(self):
        self.x -= self.speed

    def draw(self):
        screen.blit(cactus_image, (self.x, self.y))

    def is_offscreen(self):
        return self.x + self.width < 0

# Initialize coconut trees and cacti
coconut_trees = []
cacti = []
COCONUT_TREE_SPAWN_TIME = 6000  # Time between coconut tree spawns in milliseconds
CACTUS_SPAWN_TIME = 7000  # Time between cactus spawns in milliseconds
last_coconut_tree_spawn = pygame.time.get_ticks()
last_cactus_spawn = pygame.time.get_ticks()

# Tumbleweed class for rolling animation in sunset mode
class Tumbleweed:
    def __init__(self):
        self.original_image = tumbleweed_image
        self.image = self.original_image.copy()
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.x = width + self.width
        # Place tumbleweed at ground level
        self.y = height - self.height - random.randint(0, 20)
        self.speed = random.uniform(3.0, 6.0)  # Faster than cacti for realistic movement
        self.rotation = 0
        self.rotation_speed = random.uniform(5, 15)  # Degrees per frame
        # Add some vertical bouncing
        self.bounce_height = random.randint(5, 15)
        self.bounce_speed = random.uniform(0.1, 0.2)
        self.bounce_offset = random.uniform(0, 2 * math.pi)  # Random starting point in bounce cycle
        self.bounce_time = 0

    def update(self):
        # Move left
        self.x -= self.speed
        
        # Rotate the tumbleweed
        self.rotation = (self.rotation + self.rotation_speed) % 360
        self.image = pygame.transform.rotate(self.original_image, self.rotation)
        
        # Update bounce
        self.bounce_time += self.bounce_speed
        bounce_y = math.sin(self.bounce_time + self.bounce_offset) * self.bounce_height
        
        # Adjust y position with bounce
        self.y = height - self.height - random.randint(0, 10) + bounce_y

    def draw(self):
        # Get the rect of the rotated image to ensure it's centered properly
        rect = self.image.get_rect(center=(self.x + self.width//2, self.y + self.height//2))
        screen.blit(self.image, rect.topleft)

    def is_offscreen(self):
        return self.x + self.width < 0

# Initialize tumbleweeds
tumbleweeds = []
TUMBLEWEED_SPAWN_TIME = 4000  # Time between tumbleweed spawns in milliseconds
last_tumbleweed_spawn = pygame.time.get_ticks()

# Charizard class for random appearance in sunset mode
class Charizard:
    def __init__(self):
        self.frames = charizard_frames
        self.current_frame = 0
        self.frame_time = 0
        self.frame_delay = 100  # Milliseconds between frame changes
        self.width = self.frames[0].get_width()
        self.height = self.frames[0].get_height()
        self.x = width + self.width
        # Position Charizard in the sky
        self.y = random.randint(50, height // 3)
        self.speed = random.uniform(1.0, 2.5)  # Slower than tumbleweeds for majestic effect

    def update(self, current_time):
        # Move left
        self.x -= self.speed
        
        # Animate the frames
        if current_time > self.frame_time:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.frame_time = current_time + self.frame_delay

    def draw(self):
        screen.blit(self.frames[self.current_frame], (self.x, self.y))

    def is_offscreen(self):
        return self.x + self.width < 0

# Initialize charizards
charizards = []
CHARIZARD_SPAWN_TIME = 10000  # Time between charizard spawns in milliseconds (less frequent than other elements)
last_charizard_spawn = pygame.time.get_ticks()

# Lapras class for swimming in morning mode
class Lapras:
    def __init__(self):
        self.frames = lapras_frames
        self.current_frame = 0
        self.frame_time = 0
        self.frame_delay = 120  # Milliseconds between frame changes
        self.width = self.frames[0].get_width()
        self.height = self.frames[0].get_height()
        self.x = width + self.width
        # Position Lapras in the water (bottom of screen)
        self.y = height - self.height - random.randint(10, 30)
        self.speed = random.uniform(1.0, 2.0)  # Gentle swimming speed
        # Add some vertical bobbing for swimming effect
        self.bob_height = random.randint(3, 8)
        self.bob_speed = random.uniform(0.05, 0.1)
        self.bob_offset = random.uniform(0, 2 * math.pi)
        self.bob_time = 0

    def collides_with_tree(self, tree):
        # Simple rectangle collision detection
        lapras_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        tree_rect = pygame.Rect(tree.x, tree.y, tree.width, tree.height)
        return lapras_rect.colliderect(tree_rect)

    def update(self, current_time):
        # Move left
        self.x -= self.speed
        
        # Animate the frames
        if current_time > self.frame_time:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.frame_time = current_time + self.frame_delay
            
        # Update bobbing motion
        self.bob_time += self.bob_speed
        bob_y = math.sin(self.bob_time + self.bob_offset) * self.bob_height
        
        # Adjust y position with bobbing
        self.y = height - self.height - random.randint(10, 20) + bob_y

    def draw(self):
        screen.blit(self.frames[self.current_frame], (self.x, self.y))

    def is_offscreen(self):
        return self.x + self.width < 0

# Initialize lapras
lapras_pokemon = []
LAPRAS_SPAWN_TIME = 8000  # Time between lapras spawns in milliseconds
last_lapras_spawn = pygame.time.get_ticks()

# Load the pyramid image for sunset mode
pyramid_image = pygame.image.load('pyramid.png')  # Ensure 'pyramid.png' is in the same directory
pyramid_image = pygame.transform.scale(pyramid_image, (200, 150))  # Scale to fit the desired size

# Replace mountain with pyramid in sunset mode
def draw_pyramids():
    current_time = pygame.time.get_ticks()
    pyramid_width = pyramid_image.get_width()
    num_pyramids = width // pyramid_width + 2
    for i in range(num_pyramids):
        screen.blit(pyramid_image, (i * pyramid_width - (current_time // 50) % pyramid_width, height - pyramid_image.get_height()))

# Function to draw mountains
def draw_mountains():
    # Get the current time for scrolling effect
    current_time = pygame.time.get_ticks()
    
    # Calculate the width of the mountain image
    mountain_width = mountain_image.get_width()
    
    # Calculate how many mountains we need to cover the screen width
    num_mountains = width // mountain_width + 2  # Add extra to ensure full coverage
    
    # Draw mountains side by side with no gaps
    for i in range(num_mountains):
        screen.blit(mountain_image, (i * mountain_width - (current_time // 50) % mountain_width, height - mountain_image.get_height()))

# Function to draw buildings for night mode
def draw_buildings():
    # Get the current time for scrolling effect
    current_time = pygame.time.get_ticks()
    
    # Calculate the width of the building image
    building_width = building_image.get_width()
    
    # Calculate how many buildings we need to cover the screen width
    num_buildings = width // building_width + 2  # Add extra to ensure full coverage
    
    # Draw buildings side by side with no gaps
    for i in range(num_buildings):
        screen.blit(building_image, (i * building_width - (current_time // 50) % building_width, height - building_image.get_height()))

# Function to add a new star randomly
def add_random_star():
    if len(stars) < 100:  # Limit the maximum number of stars
        stars.append(Star())

# Load the star image for night mode
star_image = pygame.image.load('star.png')  # Ensure 'star.png' is in the same directory
star_image = pygame.transform.scale(star_image, (20, 20))  # Scale to fit the desired size

# Load Darkrai gif frames for night mode
darkrai_frames = []
try:
    # First try to load a GIF file directly using PIL if available
    if PIL_AVAILABLE:
        try:
            gif_path = 'darkrai.gif'
            if os.path.exists(gif_path):
                print(f"Loading GIF file: {gif_path}")
                gif = Image.open(gif_path)
                frame_count = 0
                
                # Extract all frames from the GIF
                while True:
                    try:
                        gif.seek(frame_count)
                        frame_img = gif.convert('RGBA')
                        frame_data = np.array(frame_img)
                        # Convert PIL image to pygame surface
                        frame_surface = pygame.image.frombuffer(
                            frame_data.tobytes(), frame_img.size, 'RGBA')
                        # Scale the frame
                        frame_surface = pygame.transform.scale(frame_surface, (120, 120))
                        darkrai_frames.append(frame_surface)
                        frame_count += 1
                    except EOFError:
                        # End of frames
                        break
                
                print(f"Loaded {len(darkrai_frames)} frames from Darkrai GIF")
        except Exception as e:
            print(f"Error loading Darkrai GIF: {e}")
            # Fall back to other methods if GIF loading fails
            darkrai_frames = []
    
    # If GIF loading failed or PIL is not available, try loading individual frames
    if not darkrai_frames:
        # Try to load individual frames
        darkrai_files = sorted(glob.glob('darkrai_*.png'))
        if darkrai_files:
            for file in darkrai_files:
                frame = pygame.image.load(file)
                frame = pygame.transform.scale(frame, (120, 120))
                darkrai_frames.append(frame)
            print(f"Loaded {len(darkrai_frames)} individual Darkrai frame files")
        else:
            # If no individual frames, try to load a single image
            darkrai_img = pygame.image.load('darkrai.png')
            darkrai_img = pygame.transform.scale(darkrai_img, (120, 120))
            darkrai_frames = [darkrai_img]
            print("Loaded single Darkrai image")
    
    if not darkrai_frames:
        raise Exception("No Darkrai images found")
        
except Exception as e:
    print(f"Error loading Darkrai images: {e}")
    # Create a placeholder if loading fails
    darkrai_frames = [pygame.Surface((120, 120))]
    darkrai_frames[0].fill((75, 0, 130))  # Dark purple color as placeholder
    print("Using placeholder for Darkrai")

# Darkrai class for night mode
class Darkrai:
    def __init__(self):
        self.frames = darkrai_frames
        self.current_frame = 0
        self.frame_time = 0
        self.frame_delay = 100  # Milliseconds between frame changes
        self.width = self.frames[0].get_width()
        self.height = self.frames[0].get_height()
        self.x = width + self.width
        # Position Darkrai in the night sky
        self.y = random.randint(50, height // 2)
        self.speed = random.uniform(1.0, 2.0)  # Slower movement for ghostly effect
        # Add floating motion
        self.float_offset = random.uniform(0, 2 * math.pi)
        self.float_speed = random.uniform(0.02, 0.04)
        self.float_height = random.randint(10, 20)
        self.float_time = 0

    def update(self, current_time):
        # Move left
        self.x -= self.speed
        
        # Animate the frames
        if current_time > self.frame_time:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.frame_time = current_time + self.frame_delay
            
        # Update floating motion
        self.float_time += self.float_speed
        float_y = math.sin(self.float_time + self.float_offset) * self.float_height
        self.y = self.y + float_y

    def draw(self):
        screen.blit(self.frames[self.current_frame], (self.x, self.y))

    def is_offscreen(self):
        return self.x + self.width < 0

# Initialize darkrai
darkrais = []
DARKRAI_SPAWN_TIME = 12000  # Time between Darkrai spawns in milliseconds (less frequent)
last_darkrai_spawn = pygame.time.get_ticks()

while running:
    # Calculate elapsed time
    elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
    
    # Get current time for various time-based events
    current_time = pygame.time.get_ticks()
    
    # Check if it's time to toggle rain
    if current_time > rain_toggle_time:
        is_raining = not is_raining
        rain_toggle_time = current_time + 40000  # Set next toggle time to 40 seconds later
        if is_raining:
            # Randomize rain intensity when it starts raining
            rain_intensity = random.randint(50, 150)
            rain_drops = [RainDrop() for _ in range(rain_intensity)]
    
    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if game_state == STATE_NAME_INPUT:
                if name_input.handle_event(event):
                    player_name = name_input.text.strip()
                    if player_name:  # Only proceed if name is not empty
                        game_state = STATE_PLAYING
                        is_raining = random.choice([True, False])  # Decide weather at game start
            elif event.key == pygame.K_SPACE:
                if game_state == STATE_GAME_OVER:
                    if show_debug:  # If showing leaderboard
                        show_debug = False
                        game_state = STATE_NAME_INPUT
                        # Stop game over sound
                        game_over_sound.stop()
                        # Reset all game variables
                        bird = FaceBird()
                        score = 0
                        start_time = pygame.time.get_ticks()
                        game_over = False
                        game_active = False
                        pipes.clear()
                        last_pipe_spawn = pygame.time.get_ticks()
                        current_voice_intensity = 0.0
                        name_input = TextInput(width//2 - 150, height//2 - 25, 300, 50)
                        # Reset audio stream
                        try:
                            stream.stop()
                            stream.close()
                            stream = sd.InputStream(
                                callback=audio_callback,
                                channels=1,
                                samplerate=sample_rate,
                                blocksize=block_size
                            )
                            stream.start()
                        except Exception as e:
                            print(f"Error resetting audio stream: {e}")
                    else:  # Show leaderboard first
                        show_debug = True
                elif game_state == STATE_PLAYING:
                    game_active = True
                elif game_state == STATE_SETTINGS:
                    game_state = STATE_NAME_INPUT  # Return to name input after settings
            elif event.key == pygame.K_s:
                # Enter settings menu
                game_state = STATE_SETTINGS
            elif event.key == pygame.K_UP and game_state == STATE_SETTINGS:
                # Increase sensitivity
                sensitivity_level = min(sensitivity_level + 1, max_sensitivity_level)
                audio_threshold = sensitivity_level / 1000  # Adjust threshold
            elif event.key == pygame.K_DOWN and game_state == STATE_SETTINGS:
                # Decrease sensitivity
                sensitivity_level = max(sensitivity_level - 1, 0)
                audio_threshold = sensitivity_level / 1000  # Adjust threshold
            elif event.key == pygame.K_RETURN and game_state == STATE_SETTINGS:
                # Save settings and return to name input
                game_state = STATE_NAME_INPUT
            elif event.key == pygame.K_d:
                # Toggle debug mode
                show_debug = not show_debug
            elif event.key == pygame.K_q:
                # Quit game
                running = False
            elif event.key == pygame.K_DELETE and game_state == STATE_GAME_OVER and show_debug:
                # Reset leaderboard when delete key is pressed
                reset_leaderboard()
                show_reset_confirmation = True
                reset_confirmation_time = current_time + 2000  # Show for 2 seconds
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                if menu_button.is_clicked(event.pos):
                    game_state = STATE_SETTINGS

    # Draw settings menu if in settings state
    if game_state == STATE_SETTINGS:
        draw_settings_menu()
    else:
        # Ensure the button is drawn on top of other elements
        if game_state in [STATE_NAME_INPUT, STATE_PLAYING, STATE_GAME_OVER, STATE_LEADERBOARD]:
            menu_button.draw()

        # Draw the menu button after the background and before other elements
        if game_state != STATE_SETTINGS:
            menu_button.draw()

        # Change background based on score
        if score % 15 >= 10:  # Night mode: scores 10-14, 25-29, 40-44, etc.
            screen.fill(NIGHT_SKY)  # Night sky background
            
            # Randomly add new stars occasionally
            if random.random() < 0.02:  # 2% chance each frame
                add_random_star()
            
            # Update and draw stars
            for star in stars:
                star.update()
                star.draw()
            
            # Draw moon using the moon image instead of a circle
            screen.blit(moon_image, (width - 120, 60))  # Adjusted position for the moon image
            
            # Spawn and update Darkrai in night mode (rare appearance)
            if current_time - last_darkrai_spawn > DARKRAI_SPAWN_TIME and random.random() < 0.15:  # 15% chance to spawn
                darkrais.append(Darkrai())
                last_darkrai_spawn = current_time
            
            # Update and draw Darkrai
            for darkrai in darkrais[:]:
                darkrai.update(current_time)
                darkrai.draw()
                if darkrai.is_offscreen():
                    darkrais.remove(darkrai)
            
            # Draw buildings instead of mountains in night mode
            draw_buildings()
            
            # Change bird wings to bat wings
            bird.wing_style = 'bat'
            # No need to set wing_color for bat wings as we're using a different image

        elif score % 15 >= 5 and score % 15 < 10:  # Sunset mode: scores 5-9, 20-24, 35-39, etc.
            screen.fill((255, 140, 0))  # Orange for sunset
            pygame.draw.circle(screen, (255, 69, 0), (width - 100, 100), 40)
            draw_pyramids()  # Use pyramid instead of mountain
            bird.wing_color = (255, 255, 0)
            bird.wing_style = 'regular'
            
            # Spawn cacti, tumbleweeds, and charizards
            if current_time - last_cactus_spawn > CACTUS_SPAWN_TIME and random.random() < 0.3:
                cacti.append(Cactus())
                last_cactus_spawn = current_time
            if current_time - last_tumbleweed_spawn > TUMBLEWEED_SPAWN_TIME and random.random() < 0.4:
                tumbleweeds.append(Tumbleweed())
                last_tumbleweed_spawn = current_time
            if current_time - last_charizard_spawn > CHARIZARD_SPAWN_TIME and random.random() < 0.2:
                charizards.append(Charizard())
                last_charizard_spawn = current_time

            # Update and draw cacti
            for cactus in cacti[:]:
                cactus.update()
                cactus.draw()
                if cactus.is_offscreen():
                    cacti.remove(cactus)

            # Update and draw tumbleweeds
            for tumbleweed in tumbleweeds[:]:
                tumbleweed.update()
                tumbleweed.draw()
                if tumbleweed.is_offscreen():
                    tumbleweeds.remove(tumbleweed)

            # Update and draw charizards
            for charizard in charizards[:]:
                charizard.update(current_time)
                charizard.draw()
                if charizard.is_offscreen():
                    charizards.remove(charizard)

        else:  # Morning mode: scores 0-4, 15-19, 30-34, etc.
            screen.fill(SKY_BLUE)  # Day sky background
            
            # Draw mountains
            draw_mountains()
            
            # Change bird wings to white
            bird.wing_color = WHITE
            bird.wing_style = 'regular'
            
            # Spawn and update coconut trees in day mode
            if current_time - last_coconut_tree_spawn > COCONUT_TREE_SPAWN_TIME and random.random() < 0.3:  # 30% chance to spawn
                coconut_trees.append(CoconutTree())
                last_coconut_tree_spawn = current_time
                
            # Spawn and update Lapras in morning mode (rare appearance)
            if current_time - last_lapras_spawn > LAPRAS_SPAWN_TIME and random.random() < 0.25:  # 25% chance to spawn
                lapras_pokemon.append(Lapras())
                last_lapras_spawn = current_time

            # Update and draw coconut trees (only in morning mode)
            if score % 15 < 5:  # Only in morning mode
                for tree in coconut_trees[:]:
                    tree.update()
                    tree.draw()
                    if tree.is_offscreen():
                        coconut_trees.remove(tree)
                    
                # Update and draw Lapras (only in morning mode)
                for lapras in lapras_pokemon[:]:
                    lapras.update(current_time)
                    # Check for collisions with coconut trees before drawing
                    collides_with_any_tree = any(lapras.collides_with_tree(tree) for tree in coconut_trees)
                    if not collides_with_any_tree:
                        lapras.draw()
                    if lapras.is_offscreen():
                        lapras_pokemon.remove(lapras)
                    
            # Update and draw cacti (only in sunset mode)
            if score % 15 >= 5 and score % 15 < 10:  # Only in sunset mode
                for cactus in cacti[:]:
                    cactus.update()
                    cactus.draw()
                    if cactus.is_offscreen():
                        cacti.remove(cactus)
            
            # Update and draw tumbleweeds (only in sunset mode)
            for tumbleweed in tumbleweeds[:]:
                tumbleweed.update()
                tumbleweed.draw()
                if tumbleweed.is_offscreen():
                    tumbleweeds.remove(tumbleweed)
                    
            # Update and draw charizards (only in sunset mode)
            for charizard in charizards[:]:
                charizard.update(current_time)
                charizard.draw()
                if charizard.is_offscreen():
                    charizards.remove(charizard)

        # Update and draw clouds
        # current_time is already defined at the beginning of the loop
        
        # Spawn new clouds
        if current_time - last_cloud_spawn > CLOUD_SPAWN_TIME:
            clouds.append(Cloud())
            last_cloud_spawn = current_time
        
        # Update and draw clouds
        for cloud in clouds[:]:
            cloud.update()
            cloud.draw()

        # Update and draw flying birds
        if current_time - last_flying_bird_spawn > FLYING_BIRD_SPAWN_TIME:
            flying_birds.append(FlyingBird())
            last_flying_bird_spawn = current_time

        # Update and draw flying birds
        for flying_bird in flying_birds[:]:
            flying_bird.update()
            flying_bird.draw()
            if flying_bird.is_offscreen():
                flying_birds.remove(flying_bird)

        # Draw rain if it's raining
        if is_raining:
            for drop in rain_drops:
                drop.update()
                drop.draw()

        if game_state == STATE_NAME_INPUT:
            title_text = "Enter Your Name"
            title_x = width//2 - title_font.size(title_text)[0]//2
            draw_text_with_outline(title_text, title_font, WHITE, BLACK, title_x, height//3)
            
            name_input.draw()
            
            instruction_text = "Type your name and press Enter to start"
            instruction_x = width//2 - font.size(instruction_text)[0]//2
            draw_text_with_outline(instruction_text, font, WHITE, BLACK, instruction_x, height//2 + 50)
        
        elif game_state == STATE_PLAYING:
            # Regular game update code
            # Face detection from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Exiting...")
                break
            
            # Rotate the frame if needed (fix for -90 degree rotation)
            # This rotates the frame 90 degrees clockwise to fix the -90 degree rotation
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Flip the frame horizontally for a more intuitive selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = face_mesh.process(frame_rgb)
            
            # Extract face image when a face is detected
            if results.multi_face_landmarks:
                h, w, c = frame.shape
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get bounding box of face
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Add some padding to the face crop
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Extract face image
                if x_min < x_max and y_min < y_max:
                    face_image = frame[y_min:y_max, x_min:x_max].copy()
                    bird.face_image = face_image
            
            # Check for voice jump
            current_time = pygame.time.get_ticks()
            voice_trigger = current_voice_intensity > audio_threshold
            if voice_trigger and not game_over:
                print('Fly sound triggered')  # Debug print
                fly_sound.play()  # Play fly sound
            
            # Update game state
            if game_active and not game_over:
                bird.update(current_voice_intensity)
                
                # Play fly sound with each flap
                if voice_trigger:
                    fly_sound.play()  # Play fly sound

                # Spawn new pipes
                if current_time - last_pipe_spawn > PIPE_SPAWN_TIME:
                    pipes.append(Pipe())
                    last_pipe_spawn = current_time
                
                # Update pipes
                for pipe in pipes[:]:
                    pipe.update()
                    
                    # Check for collision
                    if pipe.collide(bird):
                        print('Collision with pipe detected!')  # Debug print
                        hit_sound.play()  # Play hit sound
                        game_over_sound.play()  # Play game over sound
                        game_over = True
                        game_state = STATE_GAME_OVER
                        add_score_to_leaderboard(player_name, score)
                    
                    # Score points when passing pipes
                    if not pipe.passed and pipe.x + pipe.width < bird.x:
                        score += 1
                        pipe.passed = True
                        bell_sound.play()  # Play bell sound consistently
                    
                    # Remove off-screen pipes
                    if pipe.is_offscreen():
                        pipes.remove(pipe)
                
                # Spawn new collectables
                if score % 10 == 0 and score != 0 and score > last_collectable_score:
                    # Check if a collectable is already present to avoid multiple spawns
                    if not any(c.x > width for c in collectables):
                        collectables.append(Collectable())
                        last_collectable_score = score  # Update the last score milestone

                # Update and draw collectables
                for collectable in collectables[:]:
                    collectable.update()
                    collectable.draw()

                    # Check for collision with bird
                    if collectable.collide(bird):
                        print('Collectable collected!')  # Debug print
                        collectable.collected = True
                        collectable_sound.play()  # Play collectable sound
                        try:
                            # Remove all pipes as a power-up effect
                            if pipes:
                                pipes.clear()  # Remove all pipes
                                explosion_sound.play()  # Play explosion sound
                        except Exception as e:
                            print(f"Error clearing pipes: {e}")  # Log error without closing the game
            
            # Draw game elements
            for pipe in pipes:
                pipe.draw()
            bird.draw()
            
            # Display game info
            info_text = f"Player: {player_name} | Score: {score}"
            draw_text_with_outline(info_text, font, WHITE, BLACK, 10, 10)
            
            # Display audio level indicator
            pygame.draw.rect(screen, GRAY, (width-210, 10, 200, 20), 1)
            intensity_width = int(current_voice_intensity * 1000)
            pygame.draw.rect(screen, GREEN, (width-210, 10, min(intensity_width, 200), 20))
            
            # Draw threshold line
            threshold_x = width-210 + int(audio_threshold * 1000)
            pygame.draw.line(screen, (255, 0, 0), (threshold_x, 10), (threshold_x, 30), 2)
            
            # Display score
            score_text = f"Score: {score}"
            score_x = width//2 - font.size(score_text)[0]//2
            draw_text_with_outline(score_text, font, WHITE, BLACK, score_x, 10)
            
            # Debug info
            if show_debug:
                debug_info = [
                    f"Voice Intensity: {current_voice_intensity:.6f}",
                    f"Threshold: {audio_threshold}",
                    f"Is Jumping: {voice_trigger}",
                    f"Bird Y-Position: {bird.y}",
                    f"Bird Velocity: {bird.velocity}",
                    f"FPS: {int(clock.get_fps())}"
                ]
                
                for i, info in enumerate(debug_info):
                    draw_text_with_outline(info, debug_font, WHITE, BLACK, 10, height - 120 + i*20)
            
            if not game_active:
                start_text = "Press SPACE to Start"
                instruction_text = "Make loud noise to make your face jump!"
                controls_text = "Press D for debug info, Q to quit"
                
                start_x = width//2 - font.size(start_text)[0]//2
                instruction_x = width//2 - font.size(instruction_text)[0]//2
                controls_x = width//2 - font.size(controls_text)[0]//2
                
                draw_text_with_outline(start_text, font, WHITE, BLACK, start_x, height//2 - 30)
                draw_text_with_outline(instruction_text, font, WHITE, BLACK, instruction_x, height//2)
                draw_text_with_outline(controls_text, font, WHITE, BLACK, controls_x, height//2 + 30)
            
            # Display webcam feed in corner (smaller size)
            webcam_width, webcam_height = 160, 120
            webcam_surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (webcam_width, webcam_height)))
            screen.blit(webcam_surface, (width - webcam_width - 10, height - webcam_height - 10))
            
        elif game_state == STATE_GAME_OVER:
            game_over_text = "Game Over!"
            game_over_x = width//2 - title_font.size(game_over_text)[0]//2
            draw_text_with_outline(game_over_text, title_font, (255, 0, 0), BLACK, game_over_x, 50)
            
            score_text = f"Your Score: {score}"
            score_x = width//2 - font.size(score_text)[0]//2
            draw_text_with_outline(score_text, font, WHITE, BLACK, score_x, 120)
            
            if show_debug:
                leaderboard_text = "Leaderboard"
                leaderboard_x = width//2 - title_font.size(leaderboard_text)[0]//2
                draw_text_with_outline(leaderboard_text, title_font, WHITE, BLACK, leaderboard_x, 180)
                
                leaderboard = load_leaderboard()
                y_pos = 250
                for i, entry in enumerate(leaderboard):
                    score_line = f"{i+1}. {entry['name']}: {entry['score']} ({entry['date']}))"
                    score_x = width//2 - font.size(score_line)[0]//2
                    draw_text_with_outline(score_line, font, WHITE, BLACK, score_x, y_pos)
                    y_pos += 30
                
                delete_instruction = "Press DELETE to reset leaderboard"
                delete_x = width//2 - font.size(delete_instruction)[0]//2
                draw_text_with_outline(delete_instruction, font, WHITE, BLACK, delete_x, height - 90)
                
                if show_reset_confirmation and current_time < reset_confirmation_time:
                    confirmation_text = "Leaderboard has been reset!"
                    confirmation_x = width//2 - font.size(confirmation_text)[0]//2
                    confirmation_bg = pygame.Rect(confirmation_x - 10, height//2 - 15, 
                                                font.size(confirmation_text)[0] + 20, 30)
                    pygame.draw.rect(screen, (0, 0, 0, 128), confirmation_bg)
                    draw_text_with_outline(confirmation_text, font, (255, 0, 0), BLACK, confirmation_x, height//2 - 10)
                
                continue_text = "Press SPACE to play again"
                continue_x = width//2 - font.size(continue_text)[0]//2
                draw_text_with_outline(continue_text, font, WHITE, BLACK, continue_x, height - 60)
            else:
                view_text = "Press SPACE to view leaderboard"
                view_x = width//2 - font.size(view_text)[0]//2
                draw_text_with_outline(view_text, font, WHITE, BLACK, view_x, height - 60)
    
    # Update display
    pygame.display.flip()
    clock.tick(60)

# Proper cleanup
try:
    stream.stop()
    stream.close()
except Exception as e:
    print(f"Error closing audio stream: {e}")

cap.release()
pygame.quit()
sys.exit()