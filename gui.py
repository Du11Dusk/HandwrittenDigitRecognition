import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk, ImageOps

from predict import predict_from_pil


class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("900x520")
        self.root.resizable(False, False)

        self.canvas_size = 280
        self.brush_size = 16
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.build_ui()

    def build_ui(self):
        title = tk.Label(self.root, text="手写数字识别", font=("Microsoft YaHei", 20, "bold"))
        title.pack(pady=(12, 8))

        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=16, pady=8)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=12)

        self.canvas = tk.Canvas(left_frame, width=self.canvas_size, height=self.canvas_size, bg="white", highlightbackground="black")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        button_frame = tk.Frame(left_frame)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="清空", width=10, command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="上传图片", width=12, command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="识别", width=10, command=self.predict).pack(side=tk.LEFT, padx=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=16)

        self.preview_label = tk.Label(right_frame, text="绘制或上传图片", font=("Microsoft YaHei", 12))
        self.preview_label.pack(pady=(0, 8))

        self.preview_canvas = tk.Label(right_frame, bg="white", width=280, height=280, relief=tk.SUNKEN)
        self.preview_canvas.pack()

        result_frame = tk.LabelFrame(right_frame, text="识别结果", padx=12, pady=12)
        result_frame.pack(fill=tk.BOTH, pady=(12, 0))
        self.result_var = tk.StringVar(value="等待识别")
        tk.Label(result_frame, textvariable=self.result_var, font=("Microsoft YaHei", 18, "bold"), fg="blue").pack()

        self.confidence_var = tk.StringVar(value="")
        tk.Label(result_frame, textvariable=self.confidence_var, font=("Microsoft YaHei", 11)).pack(pady=(4, 0))

        self.status_var = tk.StringVar(value="")
        tk.Label(result_frame, textvariable=self.status_var, font=("Microsoft YaHei", 10), fg="gray").pack(pady=(6, 0))

        self.update_preview()

    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw_on_canvas(self, event):
        if not getattr(self, "drawing", False):
            return
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=self.brush_size, fill="black", capstyle=tk.ROUND, smooth=True)
        self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=0, width=self.brush_size)
        self.last_x = event.x
        self.last_y = event.y
        self.update_preview()

    def stop_draw(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_var.set("等待识别")
        self.confidence_var.set("")
        self.status_var.set("")
        self.update_preview()

    def update_preview(self):
        tk_image = ImageTk.PhotoImage(self.image.resize((280, 280)))
        self.preview_canvas.configure(image=tk_image)
        self.preview_canvas.image = tk_image

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")])
        if not path:
            return
        try:
            img = Image.open(path).convert("L")
            img = ImageOps.autocontrast(img)
            img = img.resize((self.canvas_size, self.canvas_size), Image.Resampling.LANCZOS)
            self.image = img
            self.draw = ImageDraw.Draw(self.image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=ImageTk.PhotoImage(img.resize((self.canvas_size, self.canvas_size))))
            self.update_preview()
            self.status_var.set(os.path.basename(path))
        except Exception as exc:
            messagebox.showerror("Error", f"Unable to open image: {exc}")

    def predict(self):
        try:
            prediction, confidence = predict_from_pil(self.image)
            self.result_var.set(f"预测结果：{prediction}")
            self.confidence_var.set(f"置信度：{confidence:.2%}")
            self.status_var.set("识别完成")
        except Exception as exc:
            messagebox.showerror("Error", f"Prediction failed: {exc}")


def main():
    root = tk.Tk()
    DigitRecognizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()