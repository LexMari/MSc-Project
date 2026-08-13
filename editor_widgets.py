"""UI Widgets for the simulation - pygame has no native capabilities"""
import pygame

COLOR_WIDGET_BG = (40, 42, 50)
COLOR_WIDGET_BG_HOVER = (52, 55, 65)
COLOR_WIDGET_BORDER = (90, 93, 105)
COLOR_WIDGET_BORDER_FOCUS = (90, 170, 255)
COLOR_WIDGET_TEXT = (225, 227, 232)
COLOR_WIDGET_TEXT_DIM = (140, 143, 152)
COLOR_BUTTON_BG = (60, 110, 160)
COLOR_BUTTON_BG_HOVER = (75, 130, 185)
COLOR_BUTTON_TEXT = (240, 242, 245)
COLOR_BUTTON_DANGER_BG = (140, 55, 55)
COLOR_BUTTON_DANGER_BG_HOVER = (165, 70, 70)


class TextField:

    def __init__(self, rect, value="", numeric=False):
        self.rect = pygame.Rect(rect)
        self.value = str(value)
        self.numeric = numeric   # restricts accepted characters, doesn't force a valid final number
        self.focused = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.focused = self.rect.collidepoint(event.pos)
            return self.focused
        elif event.type == pygame.KEYDOWN and self.focused:
            if event.key == pygame.K_BACKSPACE:
                self.value = self.value[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_TAB, pygame.K_ESCAPE):
                self.focused = False
            elif event.unicode:
                ch = event.unicode
                if self.numeric and not (ch.isdigit() or ch in ".-"):
                    return True
                if ch.isprintable():
                    self.value += ch
            return True
        return False

    def draw(self, screen, font):
        border = COLOR_WIDGET_BORDER_FOCUS if self.focused else COLOR_WIDGET_BORDER
        pygame.draw.rect(screen, COLOR_WIDGET_BG, self.rect)
        pygame.draw.rect(screen, border, self.rect, 1)
        text_surf = font.render(self.value, True, COLOR_WIDGET_TEXT)
        screen.blit(text_surf, (self.rect.x + 6, self.rect.y + (self.rect.height - text_surf.get_height()) // 2))
        if self.focused and (pygame.time.get_ticks() // 500) % 2 == 0:
            cursor_x = self.rect.x + 6 + font.size(self.value)[0] + 1
            pygame.draw.line(screen, COLOR_WIDGET_TEXT, (cursor_x, self.rect.y + 4), (cursor_x, self.rect.bottom - 4), 1)


class Button:

    def __init__(self, rect, label, on_click, danger=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.danger = danger
        self._pressed = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._pressed = self.rect.collidepoint(event.pos)
            return self._pressed
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self._pressed
            if self._pressed and self.rect.collidepoint(event.pos):
                self.on_click()
            self._pressed = False
            return was_pressed
        return False

    def draw(self, screen, font, mouse_pos=None):
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)
        if self.danger:
            bg = COLOR_BUTTON_DANGER_BG_HOVER if hovered else COLOR_BUTTON_DANGER_BG
        else:
            bg = COLOR_BUTTON_BG_HOVER if hovered else COLOR_BUTTON_BG
        pygame.draw.rect(screen, bg, self.rect, border_radius=4)
        text_surf = font.render(self.label, True, COLOR_BUTTON_TEXT)
        screen.blit(text_surf, (self.rect.centerx - text_surf.get_width() // 2, self.rect.centery - text_surf.get_height() // 2))


class Dropdown:

    def __init__(self, rect, options, value=None, max_visible=8, labels=None, allow_empty=False):
        self.rect = pygame.Rect(rect)
        self.options = list(options)
        if value in self.options:
            self.value = value
        elif allow_empty:
            self.value = None
        else:
            self.value = self.options[0] if self.options else None
        self.open = False
        self.max_visible = max_visible
        self.scroll_offset = 0   # index of the first visible option
        self.labels = labels or {}   # option -> display string, defaults to str(option) if absent

    def _label_for(self, option):
        return self.labels.get(option, str(option))

    @staticmethod
    def _truncate(text, max_width, font):
        if font.size(text)[0] <= max_width:
            return text
        truncated = text
        while truncated and font.size(truncated + "...")[0] > max_width:
            truncated = truncated[:-1]
        return truncated + "..." if truncated else "..."

    def _visible_options(self):
        return self.options[self.scroll_offset:self.scroll_offset + self.max_visible]

    def _option_rects(self):
        return [(opt, pygame.Rect(self.rect.x, self.rect.bottom + i * self.rect.height, self.rect.width, self.rect.height))
                for i, opt in enumerate(self._visible_options())]

    def handle_event(self, event):
        if self.open and event.type == pygame.MOUSEWHEEL:
            max_scroll = max(0, len(self.options) - self.max_visible)
            self.scroll_offset = max(0, min(max_scroll, self.scroll_offset - event.y))
            return True
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        if self.rect.collidepoint(event.pos):
            self.open = not self.open
            return True
        if self.open:
            for option, option_rect in self._option_rects():
                if option_rect.collidepoint(event.pos):
                    self.value = option
                    break
            self.open = False
            return True
        return False

    def draw_closed(self, screen, font):
        pygame.draw.rect(screen, COLOR_WIDGET_BG, self.rect)
        pygame.draw.rect(screen, COLOR_WIDGET_BORDER_FOCUS if self.open else COLOR_WIDGET_BORDER, self.rect, 1)
        full_label = self._label_for(self.value) if self.value is not None else "(none selected)"
        max_text_width = self.rect.width - 26
        label = self._truncate(full_label, max_text_width, font)
        text_surf = font.render(label, True, COLOR_WIDGET_TEXT if self.value is not None else COLOR_WIDGET_TEXT_DIM)
        screen.blit(text_surf, (self.rect.x + 6, self.rect.y + (self.rect.height - text_surf.get_height()) // 2))
        arrow = "^" if self.open else "v"
        arrow_surf = font.render(arrow, True, COLOR_WIDGET_TEXT_DIM)
        screen.blit(arrow_surf, (self.rect.right - 16, self.rect.y + (self.rect.height - arrow_surf.get_height()) // 2))

    def draw_open_list(self, screen, font, mouse_pos=None):
        if not self.open:
            return
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        for option, option_rect in self._option_rects():
            hovered = option_rect.collidepoint(mouse_pos)
            pygame.draw.rect(screen, COLOR_WIDGET_BG_HOVER if hovered else COLOR_WIDGET_BG, option_rect)
            pygame.draw.rect(screen, COLOR_WIDGET_BORDER, option_rect, 1)
            opt_label = self._truncate(self._label_for(option), option_rect.width - 12, font)
            opt_surf = font.render(opt_label, True, COLOR_WIDGET_TEXT)
            screen.blit(opt_surf, (option_rect.x + 6, option_rect.y + (option_rect.height - opt_surf.get_height()) // 2))
        if len(self.options) > self.max_visible:
            list_height = len(self._visible_options()) * self.rect.height
            track_x = self.rect.right - 5
            pygame.draw.rect(screen, COLOR_WIDGET_BG, (track_x, self.rect.bottom, 3, list_height))
            max_scroll = len(self.options) - self.max_visible
            thumb_h = max(10, int(list_height * self.max_visible / len(self.options)))
            thumb_y = self.rect.bottom + int((list_height - thumb_h) * (self.scroll_offset / max_scroll)) if max_scroll else self.rect.bottom
            pygame.draw.rect(screen, COLOR_WIDGET_BORDER_FOCUS, (track_x, thumb_y, 3, thumb_h))

    def draw(self, screen, font, mouse_pos=None):

        self.draw_closed(screen, font)
        self.draw_open_list(screen, font, mouse_pos)

    def draw_open_overlay(self, screen, font, mouse_pos=None):

        self.draw_open_list(screen, font, mouse_pos)


class ScrollPanel:

    def __init__(self, rect, content_height):
        self.rect = pygame.Rect(rect)
        self.content_height = max(content_height, self.rect.height)
        self.scroll_y = 0
        self.surface = pygame.Surface((self.rect.width, self.content_height))

    def set_content_height(self, content_height):
        self.content_height = max(content_height, self.rect.height)
        self.surface = pygame.Surface((self.rect.width, self.content_height))
        self.scroll_y = max(0, min(self.scroll_y, self.content_height - self.rect.height))

    def handle_scroll(self, event):
        if event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll_y -= event.y * 30
            self.scroll_y = max(0, min(self.scroll_y, self.content_height - self.rect.height))

    def translate_event(self, event):
        if not hasattr(event, "pos"):
            return event
        if not self.rect.collidepoint(event.pos):
            return None
        translated = pygame.event.Event(event.type, dict(event.__dict__))
        translated.pos = (event.pos[0] - self.rect.x, event.pos[1] - self.rect.y + self.scroll_y)
        return translated

    def blit_to(self, screen):
        screen.set_clip(self.rect)
        screen.blit(self.surface, self.rect.topleft, area=pygame.Rect(0, self.scroll_y, self.rect.width, self.rect.height))
        screen.set_clip(None)
        if self.content_height > self.rect.height:
            track_x = self.rect.right - 6
            pygame.draw.rect(screen, COLOR_WIDGET_BG, (track_x, self.rect.y, 4, self.rect.height))
            thumb_h = max(20, int(self.rect.height * self.rect.height / self.content_height))
            thumb_y = self.rect.y + int((self.rect.height - thumb_h) * (self.scroll_y / (self.content_height - self.rect.height)))
            pygame.draw.rect(screen, COLOR_WIDGET_BORDER_FOCUS, (track_x, thumb_y, 4, thumb_h))


class Slider:

    def __init__(self, rect, min_value, max_value, value=None, label="", show_value=True):
        self.rect = pygame.Rect(rect)
        self.min_value = min_value
        self.max_value = max_value
        self.value = value if value is not None else min_value
        self.label = label
        self.show_value = show_value
        self.dragging = False

    def _value_to_x(self):
        span = self.max_value - self.min_value
        frac = (self.value - self.min_value) / span if span else 0.0
        return self.rect.x + frac * self.rect.width

    def _x_to_value(self, x):
        frac = (x - self.rect.x) / self.rect.width if self.rect.width else 0.0
        frac = max(0.0, min(1.0, frac))
        return self.min_value + frac * (self.max_value - self.min_value)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            handle_x = self._value_to_x()
            handle_rect = pygame.Rect(int(handle_x) - 7, self.rect.centery - 9, 14, 18)
            if handle_rect.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
                self.value = self._x_to_value(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_dragging = self.dragging
            self.dragging = False
            return was_dragging
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.value = self._x_to_value(event.pos[0])
            return True
        return False

    def draw(self, screen, font):
        if self.label:
            label_surf = font.render(self.label, True, COLOR_WIDGET_TEXT_DIM)
            screen.blit(label_surf, (self.rect.x, self.rect.y - 16))
        track_y = self.rect.centery
        pygame.draw.line(screen, COLOR_WIDGET_BORDER, (self.rect.x, track_y), (self.rect.right, track_y), 3)
        handle_x = int(self._value_to_x())
        handle_colour = COLOR_BUTTON_BG_HOVER if self.dragging else COLOR_BUTTON_BG
        pygame.draw.circle(screen, handle_colour, (handle_x, track_y), 8)
        if self.show_value:
            value_surf = font.render(f"{self.value:.1f}m", True, COLOR_WIDGET_TEXT)
            screen.blit(value_surf, (self.rect.right + 8, track_y - value_surf.get_height() // 2))
