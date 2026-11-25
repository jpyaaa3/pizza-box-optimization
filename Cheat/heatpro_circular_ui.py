"""HeatPro desktop interface adapted for the circular top-hat GA."""

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent
MAX_WORKERS = min(61, os.cpu_count() or 1)
GEOMETRY_OPTIONS = (
    ("min_start", "시작 높이 최소", "15.10"),
    ("max_start", "시작 높이 최대", "90"),
    ("min_gap", "내부 틈 최소", "0.05"),
    ("max_gap", "내부 틈 최대", "35"),
    ("min_brim", "챙 너비 최소", "0"),
    ("max_radius", "챙 반지름 최대", "260"),
)
BASE_MM = 150.10
BOARD_MM = 3.0
BACKGROUND = "#eff3f6"
PAPER = "#ffffff"
INK = "#172935"
MUTED = "#536977"
ACCENT = "#176c68"
FONT = "맑은 고딕"


class CircularHatApp:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.events = queue.Queue()
        self.output = None
        self.pending_stop = False
        self.closing = False
        self.options = {}
        self.models = {}
        self.settings = {}
        self.failed = 0
        self.last_error = ""
        self.status = tk.StringVar(value="실행 대기")
        self.details = tk.StringVar(value="설정을 입력하고 실행을 누르세요.")
        self.leaderboard_label = tk.StringVar(value="세대 상위 후보")
        self.preview_label = tk.StringVar(value="형상 미리보기")
        root.title("HeatPro · 원형 탑햇 GA")
        root.geometry("1110x760")
        root.minsize(920, 600)
        root.configure(background=BACKGROUND)
        root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=INK, font=(FONT, 10))
        style.configure("Title.TLabel", font=(FONT, 24, "bold"))
        style.configure("Section.TLabel", font=(FONT, 12, "bold"))
        style.configure("Sub.TLabel", foreground=MUTED)
        style.configure("TButton", font=(FONT, 10), padding=(12, 7))
        style.configure("Accent.TButton", background=ACCENT, foreground=PAPER, font=(FONT, 10, "bold"))
        style.map("Accent.TButton", background=[("disabled", "#d3dcdf"), ("active", "#0e5956")],
                  foreground=[("disabled", MUTED)])
        style.configure("TProgressbar", background=ACCENT, troughcolor="#dbe5ea", borderwidth=0, thickness=9)
        style.configure("Treeview", rowheight=32, background=PAPER, fieldbackground=PAPER,
                        foreground=INK, font=(FONT, 10), borderwidth=0)
        style.configure("Treeview.Heading", background="#dfe9ee", foreground=INK, font=(FONT, 10, "bold"))
        style.map("Treeview", background=[("selected", "#d3ebe7")], foreground=[("selected", INK)])

        layout = ttk.Frame(root, padding=(26, 19))
        layout.pack(fill="both", expand=True)
        header = ttk.Frame(layout)
        header.pack(fill="x")
        ttk.Label(header, text="HeatPro", style="Title.TLabel").pack(side="left")
        ttk.Label(layout, text="원형 탑햇  ·  축대칭 FEM  ·  3 mm 판재", style="Sub.TLabel").pack(anchor="w", pady=(1, 15))

        geometry = ttk.Frame(layout)
        geometry.pack(fill="x", pady=(0, 13))
        self.geometry_options = {}
        for column, (key, label, default) in enumerate(GEOMETRY_OPTIONS):
            ttk.Label(geometry, text=f"{label} (mm)").grid(row=0, column=column, sticky="w", padx=(0, 16))
            entry = ttk.Entry(geometry, width=13, font=(FONT, 10))
            entry.grid(row=1, column=column, sticky="ew", padx=(0, 16), pady=(5, 0), ipady=4)
            entry.insert(0, default)
            self.geometry_options[key] = entry

        controls = ttk.Frame(layout)
        controls.pack(fill="x", pady=(0, 17))
        for column, (key, label, default) in enumerate([
            ("population", "개체 수", "20"), ("children", "세대당 새 후보", "40"),
            ("generations", "세대 수", "25"), ("workers", "CPU 워커", str(min(8, MAX_WORKERS))),
            ("seed", "시드 (선택)", ""),
        ]):
            ttk.Label(controls, text=label).grid(row=0, column=column, sticky="w", padx=(0, 16))
            entry = ttk.Entry(controls, width=11, font=(FONT, 10))
            entry.grid(row=1, column=column, sticky="ew", padx=(0, 16), pady=(5, 0), ipady=4)
            entry.insert(0, default)
            self.options[key] = entry
        self.start_button = ttk.Button(controls, text="실행", style="Accent.TButton", command=self.start)
        self.start_button.grid(row=1, column=5, padx=(0, 7))
        self.stop_button = ttk.Button(controls, text="중지", command=self.stop, state="disabled")
        self.stop_button.grid(row=1, column=6, padx=(0, 7))
        self.open_button = ttk.Button(controls, text="결과 폴더", command=self.open_results, state="disabled")
        self.open_button.grid(row=1, column=7)

        ttk.Label(layout, textvariable=self.status, style="Section.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(layout, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 4))
        ttk.Label(layout, textvariable=self.details, style="Sub.TLabel").pack(anchor="w", pady=(0, 15))

        content = ttk.Frame(layout)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(1, weight=1)
        ttk.Label(content, textvariable=self.leaderboard_label, style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 7))
        ttk.Label(content, textvariable=self.preview_label, style="Section.TLabel").grid(row=0, column=1, sticky="w", padx=(16, 0), pady=(0, 7))
        table = ttk.Frame(content)
        table.grid(row=1, column=0, sticky="nsew")
        fields = ("id", "mass", "heat", "water", "sum")
        self.tree = ttk.Treeview(table, columns=fields, show="headings", selectmode="browse")
        for field, title, width in zip(fields, ("후보", "질량 (g) ↓", "온도 (°C) ↑", "수분 함량 (%) ↓", "등수 합"),
                                       (65, 115, 115, 149, 75)):
            self.tree.heading(field, text=title)
            self.tree.column(field, width=width, minwidth=58, anchor="center")
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.draw_selected)
        self.canvas = tk.Canvas(content, background=PAPER, highlightbackground="#cbd9df", highlightthickness=1)
        self.canvas.grid(row=1, column=1, sticky="nsew", padx=(16, 0))
        self.canvas.bind("<Configure>", self.draw_selected)
        root.after(100, self.poll)

    def start(self):
        try:
            settings = {key: int(entry.get()) for key, entry in self.options.items() if entry.get().strip()}
            if min(settings.get("population", 0), settings.get("children", 0)) < 1 or settings.get("generations", -1) < 0:
                raise ValueError
            if settings.get("seed", 0) < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("설정 확인", "개체 수, 새 후보 수, 세대 수, CPU 워커와 시드를 정수로 입력하세요.")
            return
        if not 1 <= settings.get("workers", 0) <= MAX_WORKERS:
            messagebox.showerror("CPU 워커 설정", f"이 PC에는 CPU 워커를 1~{MAX_WORKERS}개로 설정할 수 있습니다.")
            return
        try:
            limits = {key: float(entry.get()) for key, entry in self.geometry_options.items()}
            if not (15 < limits["min_start"] < limits["max_start"]
                    and 0 < limits["min_gap"] < limits["max_gap"]
                    and 0 <= limits["min_brim"] < limits["max_radius"] - BASE_MM
                    and all(abs(number) < float("inf") for number in limits.values())):
                raise ValueError
        except ValueError:
            messagebox.showerror("형상 범위 설정", "시작 높이는 15 mm 초과, 틈은 양수, 챙 너비는 0 mm 이상이어야 합니다. 최소·최대값을 확인하세요.")
            return
        settings.update(limits)
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:5]
        self.output = ROOT / "tophat_runs" / name
        command = [sys.executable, "-B", "-u", str(ROOT / "heatpro_circular_ga.py"), "--out", str(self.output)]
        for key, value in settings.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8", errors="replace",
                                            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as error:
            messagebox.showerror("실행 오류", str(error))
            return
        self.settings = settings
        self.failed = 0
        self.last_error = ""
        self.pending_stop = False
        self.models = {}
        self.leaderboard_label.set("세대 상위 후보")
        self.tree.delete(*self.tree.get_children())
        self.draw_selected()
        self.progress.configure(value=0, maximum=settings["population"] + settings["children"] * settings["generations"])
        self.status.set("실행 준비 중")
        self.details.set("FEM 실행기를 시작하고 있습니다.")
        self.set_running(True)
        self.open_button.configure(state="disabled")
        threading.Thread(target=self.read_worker, args=(self.process,), daemon=True).start()

    def read_worker(self, process):
        with process.stdout:
            for line in process.stdout:
                try:
                    self.events.put(json.loads(line))
                except json.JSONDecodeError:
                    self.events.put({"event": "log", "text": line.strip()})
        self.events.put({"event": "exit", "code": process.wait()})

    def poll(self):
        while True:
            try:
                message = self.events.get_nowait()
            except queue.Empty:
                break
            event = message["event"]
            if event == "started":
                self.open_button.configure(state="normal")
                self.details.set(f"시드 {message['seed']} · CPU 워커 {message['workers']}개 · 초기 개체 평가 중")
                if self.pending_stop:
                    self.request_stop()
            elif event == "evaluation":
                generation = message["generation"]
                completed = message["accepted"]
                if generation:
                    completed += self.settings["population"] + (generation - 1) * self.settings["children"]
                self.progress.configure(value=completed)
                if message["status"] != "accepted":
                    self.failed += 1
                if not self.pending_stop:
                    self.status.set(f"세대 {generation} / {self.settings['generations']} 평가 중")
                self.details.set(f"유효 후보 {message['accepted']} / {message['target']}  ·  누적 평가 {message['count']}"
                                 f"  ·  탈락 {self.failed}  ·  {message['elapsed_s']:.0f}초 경과")
            elif event == "generation":
                self.show_leaders(message["leaders"])
                if "pool_size" in message:
                    self.leaderboard_label.set(f"세대 상위 후보 · 선발 전 {message['pool_size']}개 기준")
                self.details.set(f"세대 {message['generation']} 저장  ·  누적 평가 {message['count']}"
                                 f"  ·  Pareto 후보 {message['archive']}  ·  {message['elapsed_s']:.0f}초 경과")
            elif event in ("stopped", "finished"):
                self.status.set("중지 · 진행 결과 저장됨" if event == "stopped" else "완료 · 결과 저장됨")
            elif event == "log" and message["text"]:
                self.last_error = message["text"]
            elif event == "exit":
                self.process = None
                self.set_running(False)
                if message["code"] != 0:
                    self.status.set("실행 오류")
                    self.details.set(self.last_error or f"종료 코드 {message['code']}")
                if self.closing:
                    self.root.destroy()
                    return
        self.root.after(100, self.poll)

    def set_running(self, running):
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        for entry in (*self.options.values(), *self.geometry_options.values()):
            entry.configure(state="disabled" if running else "normal")

    def show_leaders(self, leaders):
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        self.models = {str(item["id"]): item for item in leaders}
        for key, item in self.models.items():
            self.tree.insert("", "end", iid=key, values=(
                key, f"{item['mass_g']:.3f} ({item['mass_rank']:g}위)",
                f"{item['pizza_c']:.4f} ({item['heat_rank']:g}위)",
                f"{item['water_fraction_percent']:.5f} ({item['water_rank']:g}위)", f"{item['rank_sum']:g}",
            ))
        if self.models:
            key = selected[0] if selected and selected[0] in self.models else next(iter(self.models))
            self.tree.selection_set(key)
        self.draw_selected()

    def draw_selected(self, _event=None):
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        selected = self.tree.selection()
        if not selected or selected[0] not in self.models:
            self.preview_label.set("중심 단면 미리보기")
            self.canvas.create_text(width / 2, height / 2, text="후보가 평가되면 여기에 표시됩니다.",
                                    fill=MUTED, font=(FONT, 10))
            return

        item = self.models[selected[0]]
        self.preview_label.set(f"후보 {item['id']} · 중심 단면")
        start, gap, rim = (item[key] for key in ("start_mm", "gap_mm", "rim_radius_mm"))
        top = start + gap
        outer_base, outer_rim = BASE_MM + BOARD_MM, rim + BOARD_MM
        canvas = self.canvas
        has_brim = rim > BASE_MM
        summary = (f"챙 시작 {start:.2f} mm  ·  내부 틈 {gap:.2f} mm\n"
                   f"챙 안쪽 반지름 {rim:.2f} mm  ·  무게 {item['mass_g']:.2f} g") if has_brim else (
                   f"챙 없음  ·  안쪽 반지름 {BASE_MM:.2f} mm\n"
                   f"내부 높이 {top:.2f} mm  ·  무게 {item['mass_g']:.2f} g")
        canvas.create_text(
            width / 2, 20, text=summary, justify="center", fill=INK, font=(FONT, 10),
        )
        divider = max(104, height * 0.47) if has_brim else height - 12
        main_scale = min(max(1, width - 42) / (2 * outer_rim),
                         max(1, divider - 68) / (top + 2 * BOARD_MM))

        def full(x, z):
            return width / 2 + x * main_scale, (48 + divider) / 2 + (top / 2 - z) * main_scale

        def coordinates(transform, polygon):
            return [coordinate for point in polygon for coordinate in transform(*point)]

        outer = [(-outer_base, -BOARD_MM), (outer_base, -BOARD_MM),
                 (outer_base, start - BOARD_MM), (outer_rim, start - BOARD_MM),
                 (outer_rim, top + BOARD_MM), (-outer_rim, top + BOARD_MM),
                 (-outer_rim, start - BOARD_MM), (-outer_base, start - BOARD_MM)]
        inside = [(-BASE_MM, 0), (BASE_MM, 0), (BASE_MM, start), (rim, start),
                  (rim, top), (-rim, top), (-rim, start), (-BASE_MM, start)]
        canvas.create_polygon(coordinates(full, outer), fill="#8bbac7", outline=ACCENT)
        canvas.create_polygon(coordinates(full, inside), fill=PAPER)
        canvas.create_rectangle(*full(-150, 15), *full(150, 0), fill="#efc789", outline="#a76427")
        if not has_brim:
            return
        canvas.create_line(*full(-rim, start), *full(-rim, top), fill="#c87830", width=2)
        canvas.create_line(*full(rim, start), *full(rim, top), fill="#c87830", width=2)
        canvas.create_line(12, divider, width - 12, divider, fill="#dbe5ea")
        canvas.create_text(13, divider + 16, text="오른쪽 챙 확대", anchor="w", fill=MUTED, font=(FONT, 9))

        lo_x, hi_x = BASE_MM - 6, outer_rim + 6
        lo_z, hi_z = max(0, start - 8), top + 6
        detail_h = max(1, height - divider - 50)
        detail_scale = min(max(1, width - 40) / (hi_x - lo_x), detail_h / (hi_z - lo_z))

        def detail(x, z):
            return (width / 2 + (x - (lo_x + hi_x) / 2) * detail_scale,
                    divider + 34 + detail_h / 2 + ((lo_z + hi_z) / 2 - z) * detail_scale)

        canvas.create_polygon(coordinates(detail, [
            (BASE_MM, lo_z), (outer_base, lo_z), (outer_base, start - BOARD_MM),
            (outer_rim, start - BOARD_MM), (outer_rim, top + BOARD_MM),
            (lo_x, top + BOARD_MM), (lo_x, top), (rim, top), (rim, start),
            (BASE_MM, start),
        ]), fill="#8bbac7", outline=ACCENT)
        canvas.create_line(*detail(BASE_MM, start), *detail(rim, start),
                           *detail(rim, top), fill="#c87830", width=2)
        if lo_z < 15:
            canvas.create_rectangle(*detail(lo_x, 15), *detail(150, lo_z), fill="#efc789", outline="#a76427")

    def request_stop(self):
        if self.output is not None and self.output.is_dir():
            (self.output / "stop.request").touch()

    def stop(self):
        self.pending_stop = True
        self.status.set("진행 중인 후보 평가 후 중지")
        self.stop_button.configure(state="disabled")
        self.request_stop()

    def open_results(self):
        if self.output is not None and self.output.is_dir():
            os.startfile(self.output)

    def close(self):
        if self.process is None:
            self.root.destroy()
        elif messagebox.askyesno("실행 중", "평가 중인 후보를 마치고 결과를 저장한 뒤 종료할까요?"):
            self.closing = True
            self.stop()


if __name__ == "__main__":
    window = tk.Tk()
    CircularHatApp(window)
    window.mainloop()
