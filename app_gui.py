import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
from app_logic import evaluate_code, quality_system


class CodeQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("System Oceny Jakości Kodu (Fuzzy Logic)")
        self.root.geometry("600x450")

        # Konfiguracja kolorów
        self.bg_color = "#f0f0f0"
        self.drop_color = "#e0e0e0"
        self.highlight_color = "#d0e8f2"
        self.root.configure(bg=self.bg_color)

        # 1. Nagłówek
        self.header = tk.Label(root, text="Ocena czytelności kodu Python",
                               font=("Arial", 16, "bold"), bg=self.bg_color, fg="#333")
        self.header.pack(pady=20)

        # 2. Obszar Drag & Drop
        self.drop_frame = tk.Frame(root, bg=self.drop_color, bd=2, relief="groove", cursor="hand2")
        self.drop_frame.pack(fill="both", expand=True, padx=40, pady=20)

        self.drop_label = tk.Label(self.drop_frame, text="[ Przeciągnij plik .py tutaj ]",
                                   font=("Arial", 12), bg=self.drop_color, fg="#666")
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")

        # Rejestracja zdarzeń Drag&Drop
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_frame.dnd_bind('<<DragEnter>>', lambda e: self.drop_frame.config(bg=self.highlight_color))
        self.drop_frame.dnd_bind('<<DragLeave>>', lambda e: self.drop_frame.config(bg=self.drop_color))

        # 3. Wyniki
        self.result_frame = tk.Frame(root, bg=self.bg_color)
        self.result_frame.pack(fill="x", padx=40, pady=20)

        # Etykiety wyników
        self.lbl_filename = tk.Label(self.result_frame, text="Plik: -", font=("Arial", 10), bg=self.bg_color,
                                     anchor="w")
        self.lbl_filename.pack(fill="x")

        self.lbl_metrics = tk.Label(self.result_frame, text="Metryki: CC: - | Density: -", font=("Arial", 10),
                                    bg=self.bg_color, anchor="w")
        self.lbl_metrics.pack(fill="x", pady=5)

        self.lbl_score = tk.Label(self.result_frame, text="OCENA: - / 100", font=("Arial", 24, "bold"),
                                  bg=self.bg_color, fg="#999")
        self.lbl_score.pack(pady=10)

    def handle_drop(self, event):
        self.drop_frame.config(bg=self.drop_color)
        filepath = event.data

        # Windows czasem dodaje klamry {} do ścieżek ze spacjami
        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1]

        if not filepath.lower().endswith('.py'):
            self.lbl_score.config(text="To nie jest plik .py!", fg="red")
            return

        # Wyświetl nazwę pliku
        filename = os.path.basename(filepath)
        self.lbl_filename.config(text=f"Plik: {filename}")

        # --- OBLICZENIA ---

        # 1. Pobierz surowe metryki
        density, cc = evaluate_code(filepath)

        # 2. Przepuść przez system rozmyty
        final_score = quality_system.calculate_score(density, cc)

        # --- AKTUALIZACJA GUI ---
        self.lbl_metrics.config(text=f"Metryki: Density: {density:.2f} | CC: {cc:.2f}")

        self.lbl_score.config(text=f"{final_score:.1f} / 100")

        # Kolorowanie oceny
        if final_score >= 80:
            self.lbl_score.config(fg="#27ae60")  # Zielony
        elif final_score >= 50:
            self.lbl_score.config(fg="#f39c12")  # Pomarańczowy
        else:
            self.lbl_score.config(fg="#c0392b")  # Czerwony


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = CodeQualityApp(root)
    root.mainloop()