import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
from run3 import SignLanguageRecognizer

# ---- Theme: light / white background ----
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# App setup
app = ctk.CTk()
app.title("SignVision")
app.geometry("900x780")
app.configure(fg_color="white")  # whole window white

# Recognizer
recognizer = SignLanguageRecognizer()

# Header
header = ctk.CTkLabel(app, text="SignVision", font=("Arial", 32, "bold"), text_color="#111827")
header.pack(pady=(10, 0))

subheader = ctk.CTkLabel(app, text="Real-time Sign Language Translation", font=("Arial", 16), text_color="#4B5563")
subheader.pack(pady=(0, 20))

# Camera frame
cam_label = ctk.CTkLabel(app, text="", fg_color="white")
cam_label.pack(pady=10)

# Prediction output
output_label = ctk.CTkLabel(app, text="Current Prediction:", font=("Arial", 18, "bold"), text_color="#111827")
output_label.pack(pady=(20, 5))

output_box = ctk.CTkTextbox(
    app, height=40, width=360, font=("Arial", 16),
    fg_color="#F7F7F9", text_color="#111827",
    border_width=1, border_color="#D0D5DD"
)
output_box.pack(pady=5)
output_box.configure(state="disabled")  # make read-only

# Sentence history
history_label = ctk.CTkLabel(app, text="Sentence:", font=("Arial", 18, "bold"), text_color="#111827")
history_label.pack(pady=(20, 5))

history_box = ctk.CTkTextbox(
    app, height=120, width=640, font=("Arial", 16),
    fg_color="#F7F7F9", text_color="#111827",
    border_width=1, border_color="#D0D5DD",
    wrap="word"
)
history_box.pack(pady=5)
history_box.configure(state="disabled")  # make read-only

# ---- Buttons Section (VISIBLE on light bg) ----
button_frame = ctk.CTkFrame(app, fg_color="#F0F0F0")
button_frame.pack(pady=30, fill="x", expand=True)

def confirm_sentence(event=None):
    sentence = recognizer.sentence.strip()
    if sentence:
        recognizer.speak(sentence)
        history_box.configure(state="normal")
        history_box.insert("end", sentence + "\n")
        history_box.configure(state="disabled")
        recognizer.sentence = ""

def delete_sentence(event=None):
    if recognizer.sentence:
        recognizer.sentence = recognizer.sentence[:-1]
        history_box.configure(state="normal")
        history_box.delete("1.0", "end")
        history_box.insert("end", recognizer.sentence)
        history_box.configure(state="disabled")

confirm_button = ctk.CTkButton(
    button_frame,
    text="Confirm",
    width=200,
    height=55,
    corner_radius=12,
    font=("Arial", 16, "bold"),
    fg_color="#4CAF50",
    hover_color="#388E3C",
    bg_color="#F0F0F0",
    border_width=2,
    border_color="#2E7D32",
    text_color="white",
    command=confirm_sentence
)
confirm_button.grid(row=0, column=0, padx=30, pady=20)

delete_button = ctk.CTkButton(
    button_frame,
    text="Delete",
    width=200,
    height=55,
    corner_radius=12,
    font=("Arial", 16, "bold"),
    fg_color="#E74C3C",
    hover_color="#922B21",
    bg_color="#F0F0F0",
    border_width=2,
    border_color="#922B21",
    text_color="white",
    command=delete_sentence
)
delete_button.grid(row=0, column=1, padx=30, pady=20)

# Bind keys
app.bind('<Return>', confirm_sentence)
app.bind('<BackSpace>', delete_sentence)

# Footer
footer = ctk.CTkLabel(app, text="BETA 1.0", font=("Arial", 12), text_color="#6B7280")
footer.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

# Update function
def update_frame():
    frame, prediction, sentence = recognizer.process_frame()
    if frame is not None:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        cam_label.imgtk = imgtk
        cam_label.configure(image=imgtk)

        # Prediction output
        output_box.configure(state="normal")
        output_box.delete("1.0", "end")
        output_box.insert("end", prediction if prediction else "?")
        output_box.configure(state="disabled")

        # Show current sentence being built (live)
        history_box.configure(state="normal")
        history_box.delete("1.0", "end")
        history_box.insert("end", sentence if sentence else "")
        history_box.configure(state="disabled")

    app.after(30, update_frame)  # ~30 FPS for efficiency

# Handle proper closing & cleanup
def on_closing():
    recognizer.cleanup()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_closing)

# Start
update_frame()
app.mainloop()
