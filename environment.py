# H-BAV/environment.py

import pygame
import numpy as np
import random
import math

# --- Farb-Design ---
COLOR_BACKGROUND = (28, 33, 40)
COLOR_PLAY_AREA = (43, 48, 56)
COLOR_AGENT = (235, 100, 52)
COLOR_BALL = (52, 180, 235)
COLOR_TEXT = (220, 220, 220)


class HBAVEnv:
    def __init__(self, width=800, height=600, version=1):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("H-BAV Environment")

        self.margin = 50
        self.font = pygame.font.SysFont("Arial", 24)

        self.play_area_rect = pygame.Rect(self.margin, self.margin, self.width - 2 * self.margin,
                                          self.height - 2 * self.margin)

        self.version = version
        self.clock = pygame.time.Clock()

        self.agent_size = 30
        self.agent_speed = 7
        self.ball_radius = 20
        self.ball_speed = 15

        self.reward_survive = 1
        self.reward_terminate = -100
        self.reward_wall_hit = -2

        self.reset()

    def _add_ball(self):
        while True:
            pos = [
                random.randint(self.play_area_rect.left + self.ball_radius,
                               self.play_area_rect.right - self.ball_radius),
                random.randint(self.play_area_rect.top + self.ball_radius,
                               self.play_area_rect.bottom - self.ball_radius)
            ]
            if np.linalg.norm(np.array(pos) - np.array(self.agent_pos.center)) > 50:
                break

        angle = random.uniform(0, 2 * math.pi)
        vel = [math.cos(angle) * self.ball_speed, math.sin(angle) * self.ball_speed]

        self.balls.append({'pos': pos, 'vel': vel})

    def reset(self):
        self.agent_pos = pygame.Rect(
            random.randint(self.play_area_rect.left, self.play_area_rect.right - self.agent_size),
            random.randint(self.play_area_rect.top, self.play_area_rect.bottom - self.agent_size),
            self.agent_size, self.agent_size
        )

        self.balls = []
        self._add_ball()

        self.steps_survived = 0
        return self._get_state()

    def _get_state(self):
        closest_ball = None
        min_dist = float('inf')
        for ball in self.balls:
            dist = np.linalg.norm(np.array(self.agent_pos.center) - np.array(ball['pos']))
            if dist < min_dist:
                min_dist = dist
                closest_ball = ball

        dist_to_wall_left = self.agent_pos.left - self.play_area_rect.left
        dist_to_wall_right = self.play_area_rect.right - self.agent_pos.right
        dist_to_wall_top = self.agent_pos.top - self.play_area_rect.top
        dist_to_wall_bottom = self.play_area_rect.bottom - self.agent_pos.bottom

        agent_x = dist_to_wall_left / self.play_area_rect.width
        agent_y = dist_to_wall_top / self.play_area_rect.height

        ball_x = (closest_ball['pos'][0] - self.play_area_rect.left) / self.play_area_rect.width
        ball_y = (closest_ball['pos'][1] - self.play_area_rect.top) / self.play_area_rect.height

        wall_dists = [
            dist_to_wall_left / self.play_area_rect.width,
            dist_to_wall_right / self.play_area_rect.width,
            dist_to_wall_top / self.play_area_rect.height,
            dist_to_wall_bottom / self.play_area_rect.height
        ]

        if self.version == 1:
            state = [agent_x, agent_y, ball_x, ball_y] + wall_dists
        else:
            max_dist = np.linalg.norm([self.play_area_rect.width, self.play_area_rect.height])
            state = [agent_x, agent_y, min_dist / max_dist] + wall_dists

        return np.array(state, dtype=np.float32)

    def step(self, action):
        old_pos = self.agent_pos.topleft

        is_move_action = action <= 3
        reward = 0
        if is_move_action:
            if action == 0: self.agent_pos.y -= self.agent_speed
            elif action == 1: self.agent_pos.y += self.agent_speed
            elif action == 2: self.agent_pos.x -= self.agent_speed
            elif action == 3: self.agent_pos.x += self.agent_speed
        elif action == 4:
            self._add_ball()
            reward += 10
        elif action == 5:
            if len(self.balls) > 1:
                reward -= 15
                self.balls.pop()

        self.agent_pos.left = max(self.play_area_rect.left, self.agent_pos.left)
        self.agent_pos.right = min(self.play_area_rect.right, self.agent_pos.right)
        self.agent_pos.top = max(self.play_area_rect.top, self.agent_pos.top)
        self.agent_pos.bottom = min(self.play_area_rect.bottom, self.agent_pos.bottom)


        done = False

        has_moved = self.agent_pos.topleft != old_pos

        if is_move_action:
            if has_moved:
                self.steps_survived += len(self.balls)
                reward = self.reward_survive * len(self.balls)
            else:
                reward = self.reward_wall_hit
                # NEU: Die Punktzahl wird um den negativen Wert der Strafe reduziert.
                self.steps_survived += self.reward_wall_hit

        # Ballbewegung
        for ball in self.balls:
            ball['pos'][0] += ball['vel'][0]
            ball['pos'][1] += ball['vel'][1]
            if ball['pos'][0] <= self.play_area_rect.left + self.ball_radius or ball['pos'][0] >= self.play_area_rect.right - self.ball_radius:
                ball['vel'][0] *= -1
            if ball['pos'][1] <= self.play_area_rect.top + self.ball_radius or ball['pos'][1] >= self.play_area_rect.bottom - self.ball_radius:
                ball['vel'][1] *= -1

        # Kollisionsprüfung
        for ball in self.balls:
            ball_rect = pygame.Rect(ball['pos'][0] - self.ball_radius, ball['pos'][1] - self.ball_radius, self.ball_radius * 2, self.ball_radius * 2)
            if self.agent_pos.colliderect(ball_rect):
                done = True
                reward = self.reward_terminate * len(self.balls)
                break

        next_state = self._get_state()
        info = {}
        return next_state, reward, done, info

    def render(self):
        self.screen.fill(COLOR_BACKGROUND)
        pygame.draw.rect(self.screen, COLOR_PLAY_AREA, self.play_area_rect)
        pygame.draw.rect(self.screen, COLOR_AGENT, self.agent_pos)

        for ball in self.balls:
            pygame.draw.circle(self.screen, COLOR_BALL, (int(ball['pos'][0]), int(ball['pos'][1])), self.ball_radius)

        score_text = self.font.render(f"Punkte: {self.steps_survived} | Bälle: {len(self.balls)}", True, COLOR_TEXT)
        self.screen.blit(score_text, (self.margin, self.margin / 4))
        pygame.display.flip()

    def close(self):
        pygame.quit()