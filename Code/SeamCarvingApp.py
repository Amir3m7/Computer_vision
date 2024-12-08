import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import imageio.v2 as imageio
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import cv2
import time
import math
from typing import Optional, Tuple


class SeamCarvingApp:
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Seam Carving Application")

        self.frame = tk.Frame(self.root)
        self.frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Load images button
        self.load_button = tk.Button(self.frame, text="Load Images from Folder", command=lambda: threading.Thread(target=self.load_images).start())
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Axis selection radio buttons
        self.axis_var = tk.StringVar(value='c')
        axis_label = tk.Label(self.frame, text="Axis:")
        axis_label.pack(side=tk.LEFT, padx=5)
        self.axis_radiobutton_r = tk.Radiobutton(self.frame, text="Row", variable=self.axis_var, value='r')
        self.axis_radiobutton_r.pack(side=tk.LEFT)
        self.axis_radiobutton_c = tk.Radiobutton(self.frame, text="Column", variable=self.axis_var, value='c')
        self.axis_radiobutton_c.pack(side=tk.LEFT)

        # Scale entry field
        scale_label = tk.Label(self.frame, text="Scale:")
        scale_label.pack(side=tk.LEFT, padx=5)
        self.scale_entry = tk.Entry(self.frame)
        self.scale_entry.pack(side=tk.LEFT, padx=5)

        # Run seam carving button
        self.run_button = tk.Button(self.frame, text="Run Seam Carving", command=self.run_seam_carving)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self.frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        # Progress label
        self.progress_label = tk.Label(self.frame, text="0/0 Removed")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # Time label
        self.time_label = tk.Label(self.frame, text="Elapsed Time: 0.00s")
        self.time_label.pack(side=tk.LEFT, padx=5)

        # Left Frame for Energy Maps
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15)
        

        self.energy_fig = Figure(figsize=(4, 4))
        self.energy_ax = self.energy_fig.add_subplot(111)
        self.energy_canvas = FigureCanvasTkAgg(self.energy_fig, master=self.left_frame)
        self.energy_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Setup for Matplotlib figure and canvas
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add event listeners for zoom and pan
        self.canvas.mpl_connect('scroll_event', self.zoom)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        # Initialize global variables
        self.img: Optional[np.ndarray] = None
        self.depth_map: Optional[np.ndarray] = None
        self.saliency_map: Optional[np.ndarray] = None
        self.img_path: Optional[str] = None
        self.img_name: Optional[str] = None
        self.press_event: Optional[tk.Event] = None

    def claculation_energy_map(self, img: np.ndarray, depth_map: np.ndarray, saliency_map: np.ndarray, alpha: float = 2.0, beta: float = 0.58, gamma: float = 0.7, rho: float = 0.8) -> np.ndarray:
        # Normalize depth map
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        
        # Normalize saliency map
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_map = cv2.magnitude(grad_x, grad_y)
        gradient_map = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
        
        # Calculate energy map
        energy_map = (alpha * depth_map + beta * saliency_map + gamma * gradient_map + rho * laplacian)
        
        return energy_map

    def crop_columns(self, img: np.ndarray, energy_map: np.ndarray, scale_c: float) -> np.ndarray:
        rows, cols, _ = img.shape
        if scale_c < 1:
            total_steps = math.ceil(scale_c * cols)
            self.progress_bar['maximum'] = total_steps
            start_time = time.time()
            for step in range(total_steps):
                img, energy_map = self.remove_column(img, energy_map)
                self.progress_bar['value'] = step + 1
                self.progress_bar.update_idletasks()
                self.progress_label.config(text=f"{step + 1}/{total_steps} Removed")
                self.progress_label.update_idletasks()
                elapsed_time = time.time() - start_time
                self.time_label.config(text=f"Elapsed Time: {elapsed_time:.2f}s")
                self.time_label.update_idletasks()
        return img

    def crop_rows(self, img: np.ndarray, energy_map: np.ndarray, scale_r: float) -> np.ndarray:
        img = np.rot90(img, 1, (0, 1))
        energy_map = np.rot90(energy_map, 1, (0, 1))
        img = self.crop_columns(img, energy_map, scale_r)
        img = np.rot90(img, -1, (0, 1))
        return img

    def remove_column(self, img: np.ndarray, energy_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols, _ = img.shape
        which_axis = self.axis_var.get()
        M, backtrack = self.find_min_seam(energy_map)
        mask = np.ones((rows, cols), dtype=bool)
        j = np.argmin(M[-1])
        seam = []
        for i in reversed(range(rows)):
            if j < cols:
                mask[i, j] = False
                seam.append((i, j))
                j = backtrack[i, j]
        mask = np.stack([mask] * 3, axis=2)
        temp_img = img.copy()
        temp_energy_map = energy_map.copy()
        for (i, j) in seam:
            if j < temp_img.shape[1]:
                temp_img[i, j] = [255, 0, 0]
                temp_energy_map[i, j] = 0  # Highlight the seam in the energy map
        

        energy_map = energy_map[mask[:, :, 0]].reshape((rows, cols - 1))
        img = img[mask].reshape((rows, cols - 1, 3))



        if which_axis == 'r':
            temp_img = np.rot90(temp_img, -1, (0, 1))
            img = np.rot90(img, -1, (0, 1))
            temp_energy_map = np.rot90(temp_energy_map, -1, (0, 1))
            energy_map = np.rot90(energy_map, -1, (0, 1))


        
        self.ax.clear()
        self.ax.imshow(temp_img.astype(np.uint8))
        self.canvas.draw()
        self.canvas.get_tk_widget().after(1)

        self.ax.clear()
        self.ax.imshow(img.astype(np.uint8))
        self.canvas.draw()

        # Update energy map display
        self.energy_ax.clear()
        self.energy_ax.imshow(temp_energy_map, cmap='viridis')
        self.energy_canvas.draw()


        self.energy_ax.clear()
        self.energy_ax.imshow(energy_map, cmap='viridis')
        self.energy_canvas.draw()

        if which_axis == 'r':
            # temp_img = np.rot90(temp_img, 1, (0, 1))
            img = np.rot90(img, 1, (0, 1))
            # temp_energy_map = np.rot90(temp_energy_map, 1, (0, 1))
            energy_map = np.rot90(energy_map, 1, (0, 1))

        return img, energy_map

    def find_min_seam(self, energy_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = energy_map.shape
        M = energy_map.copy()
        backtrack = np.zeros_like(M, dtype=int)
        for i in range(1, rows):
            for j in range(0, cols):
                if j == 0:
                    idx = np.argmin(M[i - 1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]
                M[i, j] += min_energy
        return M, backtrack

    def load_images(self) -> None:
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        if not folder_path:
            return
        img_path = None
        depth_map_path = None
        saliency_map_path = None
        for file in os.listdir(folder_path):
           # if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                if "_DMap" in file:
                    depth_map_path = os.path.join(folder_path, file)
                elif "_SMap" in file:
                    saliency_map_path = os.path.join(folder_path, file)
                else:
                    img_path = os.path.join(folder_path, file)
        if not img_path or not depth_map_path or not saliency_map_path:
            messagebox.showerror("Error", "Folder must contain an image, depth map, and saliency map")
            return
        self.img = imageio.imread(img_path)
        self.depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        self.saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        self.img_name = os.path.splitext(os.path.basename(img_path))[0]
        self.ax.clear()
        self.ax.imshow(self.img)
        self.canvas.draw()

        self.energy_ax.clear()
        self.energy_canvas.draw()

    def save_output_image(self, img: np.ndarray, percent: str) -> None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("BMP files", "*.bmp"), ("GIF files", "*.gif")],
            initialfile=f"Output_{self.img_name}_{percent}.png"
        )
        if file_path:
            imageio.imwrite(file_path, img)
            messagebox.showinfo("Saved", f"Image saved as {file_path}")

    def run_seam_carving(self) -> None:
        try:
            scale = float(self.scale_entry.get())
            if scale <= 0:
                raise ValueError("Scale must be greater than 0")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return
        if self.img is None or self.depth_map is None or self.saliency_map is None:
            messagebox.showerror("Input Error", "No image or maps loaded")
            return

        def process() -> None:
            energy_map = self.claculation_energy_map(self.img, self.depth_map, self.saliency_map)
            which_axis = self.axis_var.get()
            self.progress_label.config(text="0/0 Removed")
            if which_axis == 'r':
                out = self.crop_rows(self.img, energy_map, scale)
            elif which_axis == 'c':
                out = self.crop_columns(self.img, energy_map, scale)
            else:
                messagebox.showerror("Input Error", "Invalid axis selection")
                return
            self.ax.clear()
            self.ax.imshow(out)
            self.canvas.draw()
            percent = str(math.ceil((scale) * 100)) + "%"
            self.save_output_image(out, percent)
            self.set_widgets_state(tk.NORMAL)

        self.set_widgets_state(tk.DISABLED)
        threading.Thread(target=process).start()

    def set_widgets_state(self, state: str) -> None:
        self.load_button.config(state=state)
        self.axis_radiobutton_r.config(state=state)
        self.axis_radiobutton_c.config(state=state)
        self.scale_entry.config(state=state)
        self.run_button.config(state=state)

    def zoom(self, event: tk.Event) -> None:
        base_scale = 1.1
        ax = self.canvas.figure.axes[0]
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.canvas.draw()

    def on_press(self, event: tk.Event) -> None:
        self.press_event = event

    def on_drag(self, event: tk.Event) -> None:
        if self.press_event is None or event.inaxes != self.press_event.inaxes:
            return
        dx = event.xdata - self.press_event.xdata
        dy = event.ydata - self.press_event.ydata
        ax = self.press_event.inaxes
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        ax.set_xlim(cur_xlim - dx)
        ax.set_ylim(cur_ylim - dy)
        self.press_event = event
        self.canvas.draw()

    def on_release(self, event: tk.Event) -> None:
        self.press_event = None


if __name__ == "__main__":
    root = tk.Tk()
    app = SeamCarvingApp(root)
    root.mainloop()