import threading
import tkinter as tk
from mountain_cal.mountainCar_Q import  MountainCar_Q as mc

class QTableUi():
    global tableUpdated

    """
    Create a separate thread to handle Q-table visualization UI
    """

    def __init__(self, s_N, a_N, W_scale=4, H_scale=30):
        # self.Q = Q
        self.s_N = s_N
        self.a_N = a_N
        self.W_scale = W_scale
        self.H_scale = H_scale

    def create_Qtable(self):
        master = tk.Tk()  # Create Tkinter window

        W = self.s_N * self.W_scale  # 'W_scale' horizontal pixels for each row of Q
        H = self.a_N * self.H_scale  # 'H_scale' vertical pixels for each column of Q

        # Create window and canvas
        canvas = tk.Canvas(master, width=W, height=H, bg="#000000")
        canvas.pack()

        # Create image to fill with pixels
        tableImg = self._qTableRedraw(W, H, self.W_scale, self.H_scale, self.s_N, self.a_N)
        tableImgCanvas = canvas.create_image((W / 2, H / 2), image=tableImg, state="normal")

        while True:
            if self.tableUpdated:
                tableImg = self._qTableRedraw(W, H, self.W_scale, self.H_scale, self.s_N, self.a_N)
                canvas.delete(tableImgCanvas)
                tableImgCanvas = canvas.create_image((W / 2, H / 2), image=tableImg, state="normal")
                self.tableUpdated = False

            # Draw lines to break up action segments
            canvas.create_line(0, 1 * self.H_scale, W, 1 * self.H_scale, fill='#b0b0b0', width=2)
            canvas.create_line(0, 2 * self.H_scale, W, 2 * self.H_scale, fill='#b0b0b0', width=2)

            # Update tk window (replaces tk.mainloop())
            master.update_idletasks()
            master.update()

    try:
        QTableThread = threading.Thread(target=create_Qtable())
        QTableThread.start()
    except:
        print("Error: Unable to start UI thread.")

    def _qTableRedraw(self, W, H, W_scale, H_scale, s_N, a_N):
        image = tk.PhotoImage(width=W, height=H)
        # Get current Q-table, transpose it, and use it to color pixels
        Q_t = mc.Q.transpose()

        for s in range(s_N):
            for a in range(a_N):

                for x in range(W_scale):
                    for y in range(H_scale):
                        color = self._mapToColor(Q_t[a][s])  # Generate color for Q-table value

                        image.put(color, (s * W_scale + x, a * H_scale + y))
        return image

    def _mapToColor(self, x):
        # Takes a Q-table value in [-20, 0] and maps it to a BLUE color value in [0, 255].
        # R and G channels are fixed:
        R = '52'
        G = '0C'

        # If the input is 0, maps to X.
        x = int(x)
        if (x == 0):
            RGB = '#b0b0b0'
        else:
            B = round((255 / 20) * x + 255)
            RGB = '#' + R + G + '{:0>2}'.format(hex(B)[2:])

        return RGB
