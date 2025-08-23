import customtkinter as ctk

# Initialize App
ctk.set_appearance_mode("dark")  # "dark" or "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

class SignVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SignVision - Beta")
        self.geometry("1000x700")

        # ================= HEADER =================
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", pady=10)

        # Title
        title = ctk.CTkLabel(header_frame, text="SignVision",
                             font=("Arial", 28, "bold"))
        title.pack()

        # Subtitle
        subtitle = ctk.CTkLabel(header_frame, text="Real-time Sign Language Translation",
                                font=("Arial", 14))
        subtitle.pack(pady=(0, 15))

        # ================= MAIN CONTENT =================
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Left - Camera Feed
        left_frame = ctk.CTkFrame(main_frame, width=400, height=400, corner_radius=15)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        cam_label = ctk.CTkLabel(left_frame, text="Camera Feed Here", font=("Arial", 14))
        cam_label.pack(pady=20)

        start_btn = ctk.CTkButton(left_frame, text="Start Camera")
        start_btn.pack(pady=10)

        # Right - Translation Output
        right_frame = ctk.CTkFrame(main_frame, width=400, height=400, corner_radius=15)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        output_label = ctk.CTkLabel(right_frame, text="Current Translation:", font=("Arial", 14))
        output_label.pack(pady=10)

        self.output_box = ctk.CTkTextbox(right_frame, width=350, height=100)
        self.output_box.pack(pady=10)

        # Translation history
        history_label = ctk.CTkLabel(right_frame, text="Translation History:", font=("Arial", 14))
        history_label.pack(pady=5)

        self.history_box = ctk.CTkTextbox(right_frame, width=350, height=200)
        self.history_box.pack(pady=10)

        # ================= BOTTOM CONTROLS =================
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(fill="x", padx=20, pady=10)

        controls = ctk.CTkLabel(bottom_frame,
                                text="Keyboard Controls: Enter = Complete | Backspace = Delete | Esc = Stop",
                                font=("Arial", 12))
        controls.pack()

        # ================= FOOTER =================
        footer = ctk.CTkLabel(self, text="BETA 1.0", font=("Arial", 10, "italic"))
        footer.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)  # Bottom-right corner

if __name__ == "__main__":
    app = SignVisionApp()
    app.mainloop()
