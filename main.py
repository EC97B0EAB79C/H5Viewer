import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import h5py
import numpy as np
import glob
import os

class H5MultiViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HDF5 Dataset Viewer")
        self.panes = []
        self.h5_files = []
        self.current_index = 0

        # ─── Top Controls ─────────────────────────────────────────────────────
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=5, pady=5)
        ttk.Label(ctrl, text="File Pattern:").pack(side="left")
        self.file_entry = ttk.Entry(ctrl, width=40)
        self.file_entry.insert(0,"./data/{}.h5")
        self.file_entry.pack(side="left", padx=5)
        ttk.Label(ctrl, text="Time Step:").pack(side="left")
        self.ts_var = tk.StringVar(value="0000100")
        ttk.Entry(ctrl, textvariable=self.ts_var, width=8).pack(side="left")
        ttk.Button(ctrl, text="Load", command=self.load_h5_file).pack(side="left", padx=5)
        ttk.Button(ctrl, text="Prev", command=self.load_prev_file).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Next", command=self.load_next_file).pack(side="left", padx=2)

        # ─── 2×2 Panes ─────────────────────────────────────────────────────────
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)
        for i in range(2):
            for j in range(2):
                frame = ttk.LabelFrame(main, text=f"Pane {2*i + j + 1}")
                frame.grid(row=i, column=j, sticky="nsew", padx=5, pady=5)
                main.grid_rowconfigure(i, weight=1)
                main.grid_columnconfigure(j, weight=1)

                combo = ttk.Combobox(frame, state="readonly")
                combo.pack(fill="x", padx=5, pady=5)
                combo.bind("<<ComboboxSelected>>", self.on_select)

                fig = Figure(figsize=(4,3))
                ax = fig.add_subplot(111)
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.get_tk_widget().pack(fill="both", expand=True)

                self.panes.append({
                    "combo":   combo,
                    "fig":     fig,
                    "ax":      ax,
                    "canvas":  canvas
                })

    def get_dataset_list(self, filename):
        names = []
        with h5py.File(filename, "r") as f:
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    names.append(name)
            f.visititems(visitor)
        return names

    def get_h5_file_list(self):
        pattern = self.file_entry.get().replace("{}", "*")
        files = sorted(glob.glob(pattern))
        return files

    def get_ts_from_filename(self, filename):
        # Extract timestep from filename using the pattern
        pattern = self.file_entry.get()
        prefix, suffix = pattern.split("{}", 1)
        base = os.path.basename(filename)
        if base.startswith(os.path.basename(prefix)) and base.endswith(suffix):
            return base[len(os.path.basename(prefix)):-len(suffix) if suffix else None]
        return ""

    def load_h5_file(self):
        # Update file list and current index
        self.h5_files = self.get_h5_file_list()
        ts = self.ts_var.get().zfill(7)
        fname = self.file_entry.get().format(ts)
        if fname in self.h5_files:
            self.current_index = self.h5_files.index(fname)
        else:
            self.current_index = 0
            if self.h5_files:
                fname = self.h5_files[0]
                ts = self.get_ts_from_filename(fname)
                self.ts_var.set(ts)
            else:
                fname = self.file_entry.get().format(ts)
        try:
            datasets = self.get_dataset_list(fname)
        except Exception as e:
            return messagebox.showerror("Error", f"Cannot open file:\n{e}")

        for pane in self.panes:
            combo = pane["combo"]
            old = combo.get()
            combo["values"] = datasets

            if not datasets:
                combo.set("")
            elif old in datasets:
                combo.set(old)
            else:
                combo.set(datasets[0])

            sel = combo.get()
            if sel:
                self.redraw_pane(pane, fname, sel)
            else:
                self.clear_pane(pane)

    def load_prev_file(self):
        if not self.h5_files:
            self.h5_files = self.get_h5_file_list()
        if not self.h5_files:
            return
        self.current_index = (self.current_index - 1) % len(self.h5_files)
        fname = self.h5_files[self.current_index]
        ts = self.get_ts_from_filename(fname)
        self.ts_var.set(ts)
        self.load_h5_file()

    def load_next_file(self):
        if not self.h5_files:
            self.h5_files = self.get_h5_file_list()
        if not self.h5_files:
            return
        self.current_index = (self.current_index + 1) % len(self.h5_files)
        fname = self.h5_files[self.current_index]
        ts = self.get_ts_from_filename(fname)
        self.ts_var.set(ts)
        self.load_h5_file()

    def on_select(self, event):
        pane = next(p for p in self.panes if p["combo"] is event.widget)
        ds = pane["combo"].get()
        ts = self.ts_var.get().zfill(7)
        fname = self.file_entry.get().format(ts)
        self.redraw_pane(pane, fname, ds)

    def clear_pane(self, pane):
        fig, ax, canvas = pane["fig"], pane["ax"], pane["canvas"]
        fig.clear()
        pane["ax"] = fig.add_subplot(111)
        canvas.draw()

    def redraw_pane(self, pane, filename, ds_path):
        fig = pane["fig"]
        fig.clear()
        ax = fig.add_subplot(111)
        try:
            with h5py.File(filename, "r") as f:
                data = f[ds_path][()]
        except Exception as e:
            return messagebox.showerror("Error", f"Cannot read dataset:\n{e}")

        if data.ndim == 1:
            ax.plot(data)
        elif data.ndim == 2:
            im = ax.imshow(data, origin="lower", cmap="viridis")
            fig.colorbar(im, ax=ax)
        ax.set_title(ds_path.split("/")[-1])

        pane["ax"] = ax
        pane["canvas"].draw()

if __name__ == "__main__":
    H5MultiViewer().mainloop()
