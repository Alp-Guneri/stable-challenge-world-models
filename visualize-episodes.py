from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import corpus_loader


class FrameViewer:
    def __init__(self, corpus):
        self.corpus = corpus
        self.episode_index = 0
        self.frame_index = 0
        self.frames, self.actions = corpus.episode(0)

        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.fig.subplots_adjust(bottom=0.22, top=0.85)
        self.image = self.ax.imshow(self.frames[0], interpolation='nearest')
        self.ax.axis('off')
        self.fig.text(
            0.5, 0.12,
            'Left / Right: previous / next frame    |    Up / Down: previous / next episode',
            ha='center', fontsize=9,
        )

        controls = [
            ('Previous episode', self.previous_episode),
            ('Previous frame', self.previous_frame),
            ('Next frame', self.next_frame),
            ('Next episode', self.next_episode),
        ]
        self.buttons = []
        for index, (label, callback) in enumerate(controls):
            button_ax = self.fig.add_axes([0.04 + index * 0.24, 0.04, 0.22, 0.05])
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.buttons.append(button)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update()

    def update(self):
        self.image.set_data(self.frames[self.frame_index])
        action = self.actions[self.frame_index]
        self.ax.set_title(
            f'Episode {self.episode_index + 1} / {len(self.corpus)}'
            f'  |  Frame {self.frame_index + 1} / {len(self.frames)}\n'
            f'Action: {action}'
        )
        self.fig.canvas.draw_idle()

    def change_episode(self, index, last_frame=False):
        if 0 <= index < len(self.corpus):
            self.episode_index = index
            self.frames, self.actions = self.corpus.episode(index)
            self.frame_index = len(self.frames) - 1 if last_frame else 0
            self.update()

    def previous_episode(self, event=None):
        self.change_episode(self.episode_index - 1)

    def next_episode(self, event=None):
        self.change_episode(self.episode_index + 1)

    def previous_frame(self, event=None):
        if self.frame_index > 0:
            self.frame_index -= 1
            self.update()
        else:
            self.change_episode(self.episode_index - 1, last_frame=True)

    def next_frame(self, event=None):
        if self.frame_index < len(self.frames) - 1:
            self.frame_index += 1
            self.update()
        else:
            self.change_episode(self.episode_index + 1)

    def on_key(self, event):
        callback = {
            'left': self.previous_frame,
            'right': self.next_frame,
            'up': self.previous_episode,
            'down': self.next_episode,
        }.get(event.key)
        if callback is not None:
            callback()


def main():
    corpus = corpus_loader.load('corpus')
    viewer = FrameViewer(corpus)
    plt.show()


if __name__ == '__main__':
    main()
