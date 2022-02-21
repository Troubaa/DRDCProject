import pyglet
from pyglet import clock

positions1 = [[0, 0], [10, 10], [20, 20], [30, 30], [100, 30]]


class Counter:
    def __init__(self):
        self.counter = 0


counter = Counter()


def move(dt, positions, current, sprites):
    current.counter += 1
    for idx, sprite in enumerate(sprites):
        if current.counter < len(positions[idx]):
            sprite.x = positions[idx][current.counter][0]
            sprite.y = positions[idx][current.counter][1]
            print(f"current: {current.counter}, sprite position x: {sprite.x}, sprite position y: {sprite.y}")

    @window.event
    def on_draw():
        window.clear()
        circle.draw()
        for sprite in sprites:
            sprite.draw()

    @window.event
    def on_key_press(key, modifiers):
        if (key == pyglet.window.key.UP):
            for sprite in sprites:
                sprite.y += 10
        elif (key == pyglet.window.key.DOWN):
            for sprite in sprites:
                sprite.y -= 10
        elif (key == pyglet.window.key.LEFT):
            for sprite in sprites:
                sprite.x -= 10
        elif (key == pyglet.window.key.RIGHT):
            for sprite in sprites:
                sprite.x += 10






window = pyglet.window.Window(width=1000, height=600)
image = pyglet.image.load('../pyglet_sprite_images/missile.png')
image2 = pyglet.image.load('../pyglet_sprite_images/boat.png')
sprite1 = pyglet.sprite.Sprite(image, x=0, y=0)
sprite1.scale = 0.025
circle = pyglet.shapes.Circle(x=100, y=150, radius=100, color=(50, 225, 30))
sprite2 = pyglet.sprite.Sprite(image2, x=200, y=200)
sprite2.scale = 0.08
positions2 = [[200, 300], [300, 300], [400, 400], [500, 500], [400, 400]]
sprites = [sprite1, sprite2]
positions = [positions1, positions2]
clock.schedule_interval(move, 0.1, positions=positions, sprites=sprites, current=counter)
while True:
    clock.tick()

    pyglet.app.run()
    # print(f"FPS is {clock.get_fps()}")





