import Tkinter as tk
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg, os
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt

# custom toolbar
import tkFileDialog
import cv2
from main_module.main import ImageDB


class CustomToolbar(NavigationToolbar2TkAgg):
    def __init__(self, canvas_, parent_):
        # TODO: make normal buttons later
        self.toolitems = (
            ('Home', '', 'home', 'home'),
            # ('Back', '', 'back', 'back'),
            # ('Forward', '', 'forward', 'forward'),
            # (None, None, None, None),
            # ('Pan', '', 'move', 'pan'),
            # ('Zoom', '', 'zoom_to_rect', 'zoom'),
            # (None, None, None, None),
            # ('Subplots', '', 'subplots', 'configure_subplots'),
            ('Save', '', 'filesave', 'save_figure'),
        )
        NavigationToolbar2TkAgg.__init__(self, canvas_, parent_)


def get_file_path(**kwargs):
    root = tk.Tk()
    root.withdraw()
    file_path = tkFileDialog.askopenfilename(**kwargs)
    root.destroy()
    return file_path


class MyApp(object):
    def __init__(self, root, image_path):
        self.root = root
        self.upper = (0, 0)
        self.lower = (0, 0)
        self.image_path = get_file_path(initialdir=os.path.dirname(image_path))
        self.rect = self._get_selection_rect()
        self._init_app()
        self.db = ImageDB()

    # here we embed the a figure in the Tk GUI
    def _init_app(self):
        self.figure = mpl.figure.Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.toolbar = CustomToolbar(self.canvas, self.root)
        self.toolbar.update()
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.show()
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key)
        self.ax.add_patch(self.rect)

    def _get_selection_rect(self):
        return Rectangle(
            self.upper,  # (x,y)
            self.lower[0] - self.upper[0],
            self.lower[1] - self.upper[1], color='red', fill=None)

    # plot something random
    def plot(self):
        self.ax.imshow(cv2.imread(self.image_path, 1), cmap="hot", aspect="auto")
        self.rect.remove()
        self.rect = self._get_selection_rect()
        self.ax.add_patch(self.rect)

        self.figure.canvas.draw()

    def on_press(self, event):
        print 'press'
        self.upper = (event.xdata, event.ydata)

    def on_release(self, event):
        print 'release'
        self.lower = (event.xdata, event.ydata)
        self.plot()

    def on_key(self, event):
        print('you pressed', event.key, event.xdata, event.ydata)
        if event.key == 'enter':
            self.process_image(self.image_path, self.upper, self.lower)

    def process_image(self, path, upper_corner, lower_corner):
        print path, upper_corner, lower_corner
        data = sorted(enumerate(list(self.db.do_magic(path, upper_corner, lower_corner))), key=lambda x: x[1][2])

        f, axes = plt.subplots(len(data))

        for ax, (i, (img, lbp_diff, hist_diff, img_name)) in zip(axes, data):
            ax.imshow(img)
            ax.set_title("geom:{} lbp:{} hist:{} img:{}".format(i, round(lbp_diff, 2), round(hist_diff, 2), img_name))
            ax.axis('off')
        plt.show()


def main():
    root = tk.Tk()
    app = MyApp(root, '/home/deathnik/src/my/magister/webcrdf-testbed/webcrdf-testbed/data/datadb.segmxr/001.png')
    app.plot()
    root.mainloop()


if __name__ == "__main__":
    main()
