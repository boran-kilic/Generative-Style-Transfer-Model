import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Set up the game window
width = 800
height = 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Snake and food properties
snake_block = 20
snake_speed = 15
#sex;
# Initialize clock
clock = pygame.time.Clock()

# Snake class
class Snake:
    def __init__(self):
        self.x = width // 2
        self.y = height // 2
        self.dx = snake_block
        self.dy = 0
        self.body = []
        self.length = 1

    def move(self):
        self.body.append([self.x, self.y])
        if len(self.body) > self.length:
            del self.body[0]
        self.x += self.dx
        self.y += self.dy

    def draw(self):
        for segment in self.body:
            pygame.draw.rect(window, GREEN, [segment[0], segment[1], snake_block, snake_block])

    def check_collision(self):
        # Wall collision
        if self.x >= width or self.x < 0 or self.y >= height or self.y < 0:
            return True
        # Self collision
        for segment in self.body[:-1]:
            if segment == [self.x, self.y]:
                return True
        return False

# Food class
class Food:
    def __init__(self):
        self.x = round(random.randrange(0, width - snake_block) / snake_block) * snake_block
        self.y = round(random.randrange(0, height - snake_block) / snake_block) * snake_block

    def draw(self):
        pygame.draw.rect(window, RED, [self.x, self.y, snake_block, snake_block])

    def respawn(self):
        self.x = round(random.randrange(0, width - snake_block) / snake_block) * snake_block
        self.y = round(random.randrange(0, height - snake_block) / snake_block) * snake_block

def main():
    def reset_game():
        return Snake(), Food(), 0

    running = True
    game_over = False
    snake, food, score = reset_game()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if game_over:
                    if event.key == pygame.K_r:
                        game_over = False
                        snake, food, score = reset_game()
                        continue
                elif event.key == pygame.K_LEFT and snake.dx != snake_block:
                    snake.dx = -snake_block
                    snake.dy = 0
                elif event.key == pygame.K_RIGHT and snake.dx != -snake_block:
                    snake.dx = snake_block
                    snake.dy = 0
                elif event.key == pygame.K_UP and snake.dy != snake_block:
                    snake.dx = 0
                    snake.dy = -snake_block
                elif event.key == pygame.K_DOWN and snake.dy != -snake_block:
                    snake.dx = 0
                    snake.dy = snake_block

        if not game_over:
            snake.move()
            
            if snake.check_collision():
                game_over = True

            if snake.x == food.x and snake.y == food.y:
                food.respawn()
                snake.length += 1
                score += 1

            window.fill(BLACK)
            snake.draw()
            food.draw()
            
            font = pygame.font.SysFont(None, 50)
            score_text = font.render(f"Score: {score}", True, WHITE)
            window.blit(score_text, [10, 10])
        
        if game_over:
            window.fill(BLACK)
            font = pygame.font.SysFont(None, 75)
            game_over_text = font.render("Game Over!", True, WHITE)
            final_score_text = font.render(f"Final Score: {score}", True, WHITE)
            restart_text = font.render("Press R to Restart", True, WHITE)
            window.blit(game_over_text, [width//3, height//3])
            window.blit(final_score_text, [width//3, height//2])
            window.blit(restart_text, [width//3, height//1.5])

        pygame.display.update()
        clock.tick(snake_speed)

    pygame.quit()

if __name__ == "__main__":
    main()
